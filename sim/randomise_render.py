"""Re-render recorded demos with randomised visuals, as a LeRobot dataset (domain randomisation).

The demos store the full simulator state of every frame (demo_io.py), so each one can be
replayed and drawn again with a different look while the robot motion and the actions (the
training labels) stay exactly the same. Each demo becomes --copies episodes, each with its own
random look. Physics is never stepped and nothing physical changes: extra lights and stray
objects are visual only (no collisions), and the arm/cube are drawn exactly where they were.

    cd sim
    # look first: a grid of randomised frames next to real ones (calib/pairs)
    /opt/anaconda3/envs/lerobot/bin/python randomise_render.py --preview 12
    # the dataset (sharded: each shard writes every Nth (demo, copy) episode to its own root)
    /opt/anaconda3/envs/lerobot/bin/python randomise_render.py "demos/cube/trimmed/*.npz" --copies 4 \\
        --root lerobot_dataset/so101-cube-dr

What varies, and how much (ranges from the measured real/sim gap, calib/measure_gap.py,
2026-11-23; "ep" = once per episode, "fr" = every frame, as Jin et al. found frame-wise helps):
  lighting    ep: total light 0.85-1.08 of today's, 45-80% of it ambient (lower = darker
              shadows), the rest split between the headlight, the ceiling light and 0-3 extra
              shadow-casting lights at random positions (several lights = softer, overlapping
              shadows, like the real room); fr: each light's strength x N(1, 3%)
  materials   ep: white PLA arm x0.70-1.05 (real arm measured 0.74-0.86 of sim), black servos
              x0.6-1.4, cube grey 0-0.10 (real cube darker than sim's 0.08), table x0.92-1.08,
              floor texture x0.85-1.10, container alpha/tint, tape tint; small colour tints
  cameras     side ep: +-1 cm, +-1 deg, fovy +-1 deg; wrist ep: +-5 mm, +-3 deg (the wrist-view
              mismatch that no fixed correction explained, calib/refit_*.py); fr: wrist
              +-0.5 mm / 0.3 deg, side 0.1 deg
  distractors ep: 0-4 visual-only objects on the table, away from the cube, the container and
              the path between them; 15%: a second black cube at one end of the container (seen in
              the real capture)
  shadows     ep: MuJoCo draws hard shadows only; every frame is drawn with and without them
              and the shadow ratio is blurred (side 2-10 px, wrist 6-25 px) and weakened
              (x0.35-1.0) before it is put back: soft, faint shadows like the real room's
  image       ep: uneven illumination (smooth random field, up to +-12%), wrist vignetting,
              gain 0.9-1.1, gamma 0.9-1.1, white balance +-3%, saturation 0.85-1.15,
              blur (side 0-0.8 px, wrist 0-2.5 px), noise (side 0.5-2, wrist 0-1 grey levels),
              JPEG quality 70-95 (70% of episodes); fr: gain x N(1, 1%), blur +- 0.3 px

Each episode's parameters go to <root>/meta/randomisation.jsonl (episode index, demo, copy, seed,
parameters), so any episode can be reproduced.
"""
import argparse
import glob
import json
import pathlib
import shutil
import time

import cv2
import mujoco
import numpy as np

import joint_units
from demo_io import STATE_SPEC, load_demo
from export_lerobot import CAMERAS, build_features
from wrist_lens import WristLens, orient_wrist

SIM = pathlib.Path(__file__).resolve().parent
N_EXTRA_LIGHTS = 3
N_DISTRACTORS = 4
WHITE_PLA = np.array([0.70, 0.69, 0.645])
CONTAINER_XY, CONTAINER_HALF = np.array([0.2095, -0.2245]), np.array([0.10, 0.065])
BASE_XY = np.array([0.02, 0.0])


# -- scene with spare lights and spare distractor geoms ---------------------------------------

def build_model(scene):
    """The demo scene plus N_EXTRA_LIGHTS lights (off) and N_DISTRACTORS + 1 visual-only geoms
    (parked far away). Neither changes the state size, so demo states replay unchanged."""
    spec = mujoco.MjSpec.from_file(str(scene))
    for i in range(N_EXTRA_LIGHTS):
        spec.worldbody.add_light(name=f"dr_light{i}", pos=[0, 0, 2], dir=[0, 0, -1], castshadow=True,
                                 diffuse=[0, 0, 0], specular=[0, 0, 0])
    for i in range(N_DISTRACTORS + 1):                       # the last one: cube in the container
        spec.worldbody.add_geom(name=f"dr_obj{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01],
                                pos=[10 + i, 10, 0], contype=0, conaffinity=0, group=0, rgba=[0.5, 0.5, 0.5, 1])
    return spec.compile()


class Base:
    """The model's original visual values, to randomise around."""

    def __init__(self, m):
        self.mat_rgba = m.mat_rgba.copy()
        self.light_diffuse = m.light_diffuse.copy()
        self.light_pos = m.light_pos.copy()
        self.light_dir = m.light_dir.copy()
        self.head_ambient = m.vis.headlight.ambient.copy()
        self.head_diffuse = m.vis.headlight.diffuse.copy()
        self.cam_pos = m.cam_pos.copy()
        self.cam_quat = m.cam_quat.copy()
        self.cam_fovy = m.cam_fovy.copy()


def mat_ids(m):
    names = [m.mat(i).name for i in range(m.nmat)]
    white = [i for i in range(m.nmat) if np.allclose(m.mat_rgba[i, :3], WHITE_PLA, atol=0.02)]
    black = [i for i, n in enumerate(names) if n.startswith("sts3215")]
    get = names.index
    return {"white": white, "black": black, "cube": get("cube_mat"), "table": get("table_white"),
            "floor": get("floor_real_mat"), "floor_plain": get("floor_plain"),
            "container": get("container_mat"), "tape": get("container_tape_mat")}


def small_rotation(rng, sigma_deg, clip_deg):
    v = np.clip(rng.normal(0, np.radians(sigma_deg), 3), -np.radians(clip_deg), np.radians(clip_deg))
    a = np.linalg.norm(v)
    return np.array([1.0, 0, 0, 0]) if a < 1e-12 else np.r_[np.cos(a / 2), np.sin(a / 2) * v / a]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


# -- episode parameters -----------------------------------------------------------------------

def sample_episode(rng, cube_xy):
    U = rng.uniform
    p = {}
    # lighting: keep the total near today's (headlight 0.7 + 0.15, ceiling 0.17), vary the mix
    total = U(0.85, 1.08) * (0.7 + 0.15 + 0.17)
    amb = total * U(0.45, 0.80)
    n_extra = int(rng.integers(0, N_EXTRA_LIGHTS + 1))
    w = rng.dirichlet(np.ones(2 + n_extra))                  # headlight diffuse, ceiling, extras
    rest = total - amb
    p["head_ambient"] = amb
    u = U(0.3, 1.0)                                          # the headlight casts no shadows: part of
    p["head_diffuse"] = rest * w[0] * u                      # its share goes to the ceiling light
    p["ceiling"] = rest * (w[1] + w[0] * (1 - u))
    p["extra_lights"] = []
    for i in range(n_extra):
        r, az, h = U(0.3, 1.3), U(0, 2 * np.pi), U(0.8, 2.0)
        pos = np.array([0.25 + r * np.cos(az), r * np.sin(az), h])
        target = np.array([U(0.1, 0.4), U(-0.25, 0.25), 0.0])
        d = target - pos
        p["extra_lights"].append({"pos": pos.tolist(), "dir": (d / np.linalg.norm(d)).tolist(),
                                  "diffuse": float(rest * w[2 + i]), "bulbradius": float(U(0.05, 0.6))})
    p["ceiling_bulbradius"] = float(U(0.2, 0.8))
    p["light_tint"] = (1 + rng.normal(0, 0.02, 3)).tolist()
    # materials
    p["white_gain"] = float(U(0.70, 1.05))
    p["white_tint"] = (1 + rng.normal(0, 0.015, 3)).tolist()
    p["black_gain"] = float(U(0.6, 1.4))
    p["cube_grey"] = float(U(0.0, 0.10))
    p["table_gain"] = float(U(0.92, 1.08))
    p["table_tint"] = (1 + rng.normal(0, 0.01, 3)).tolist()
    p["floor_gain"] = float(U(0.85, 1.10))
    p["floor_tint"] = (1 + rng.normal(0, 0.03, 3)).tolist()
    p["container_alpha"] = float(U(0.15, 0.40))
    p["container_tint"] = (1 + rng.normal(0, 0.03, 3)).tolist()
    p["tape_tint"] = (1 + rng.normal(0, 0.06, 3)).tolist()
    # cameras (episode part)
    p["side_dpos"] = np.clip(rng.normal(0, 0.005, 3), -0.01, 0.01).tolist()
    p["side_drot"] = small_rotation(rng, 0.5, 1.0).tolist()
    p["side_dfovy"] = float(U(-1, 1))
    p["wrist_dpos"] = np.clip(rng.normal(0, 0.0025, 3), -0.005, 0.005).tolist()
    p["wrist_drot"] = small_rotation(rng, 1.5, 3.0).tolist()
    # distractors
    p["distractors"] = []
    n = int(rng.integers(0, N_DISTRACTORS + 1))
    tries = 0
    while len(p["distractors"]) < n and tries < 200:
        tries += 1
        xy = np.array([U(0.0, 0.55), U(-0.40, 0.40)])
        if (np.linalg.norm(xy - BASE_XY) < 0.12 or np.linalg.norm(xy - cube_xy) < 0.08
                or seg_dist(xy, cube_xy, CONTAINER_XY) < 0.07
                or np.all(np.abs(xy - CONTAINER_XY) < CONTAINER_HALF + 0.03)
                or any(np.linalg.norm(xy - np.array(o["pos"][:2])) < 0.06 for o in p["distractors"])):
            continue
        kind = str(rng.choice(["box", "cylinder", "sphere", "capsule"]))
        s = U(0.008, 0.035)
        size = [s, U(0.6, 1.4) * s, U(0.5, 1.5) * s]
        z = size[2] if kind in ("box", "cylinder", "capsule") else s
        if kind == "capsule":
            z = size[0] + size[1]
        col = rng.uniform(0.05, 0.95, 3)
        if col.max() < 0.25:                                   # no black objects on the table:
            col = col + 0.3                                    # only the target cube is black
        p["distractors"].append({"kind": kind, "pos": [float(xy[0]), float(xy[1]), float(z)],
                                 "size": [float(v) for v in size], "yaw": float(U(0, np.pi)),
                                 "rgba": [*map(float, col), 1.0]})
    p["cube_in_container"] = bool(U() < 0.15)
    if p["cube_in_container"]:                                 # at one end: the demos drop near the middle
        off = np.array([rng.choice([-1, 1]) * U(0.045, CONTAINER_HALF[0] - 0.03), U(-1, 1) * (CONTAINER_HALF[1] - 0.03)])
        p["container_cube_xy"] = (CONTAINER_XY + off).tolist()
        p["container_cube_yaw"] = float(U(0, np.pi / 2))
    # image look, per camera. Shadows: MuJoCo only draws hard ones, the real room's are soft and
    # faint, so each frame is drawn with and without shadows and the shadow ratio is blurred
    # (penumbra, px) and weakened (strength) before it is put back.
    for cam, blur, noise, pen in (("side", (0.0, 0.8), (0.5, 2.0), (2.0, 10.0)),
                                  ("wrist", (0.0, 2.5), (0.0, 1.0), (6.0, 25.0))):
        p[f"{cam}_img"] = {
            "shadow_blur": float(U(*pen)), "shadow_strength": float(U(0.35, 1.0)),
            "field_amp": float(U(0.0, 0.12)), "field_seed": int(rng.integers(1 << 31)),
            "vignette": float(U(0.0, 0.15)) if cam == "wrist" else 0.0,
            "gain": float(U(0.9, 1.1)), "gamma": float(U(0.9, 1.1)),
            "wb": rng.uniform(0.97, 1.03, 3).tolist(), "sat": float(U(0.85, 1.15)),
            "blur": float(U(*blur)), "noise": float(U(*noise)),
            "jpeg": int(rng.integers(70, 96)) if U() < 0.7 else None}
    return p


# -- applying them ----------------------------------------------------------------------------

class Randomiser:
    def __init__(self, scene, width=640, height=480):
        self.m = build_model(scene)
        self.d = mujoco.MjData(self.m)
        self.base = Base(self.m)
        self.mats = mat_ids(self.m)
        self.light0 = self.m.light("ceiling").id
        self.extra = [self.m.light(f"dr_light{i}").id for i in range(N_EXTRA_LIGHTS)]
        self.obj = [self.m.geom(f"dr_obj{i}").id for i in range(N_DISTRACTORS + 1)]
        self.cam = {c: self.m.camera(c).id for c in CAMERAS}
        self.r = mujoco.Renderer(self.m, height, width)
        self.lens = WristLens()
        self.W, self.H = width, height
        self.p = None
        self.fields = {}

    def set_episode(self, p):
        m, b, M = self.m, self.base, self.mats
        self.p = p
        # lights
        tint = np.array(p["light_tint"])
        m.vis.headlight.ambient[:] = p["head_ambient"] * tint
        m.vis.headlight.diffuse[:] = p["head_diffuse"] * tint
        m.light_diffuse[self.light0] = p["ceiling"] * tint
        m.light_bulbradius[self.light0] = p["ceiling_bulbradius"]
        for i, lid in enumerate(self.extra):
            if i < len(p["extra_lights"]):
                e = p["extra_lights"][i]
                m.light_pos[lid], m.light_dir[lid] = e["pos"], e["dir"]
                m.light_diffuse[lid] = e["diffuse"] * tint
                m.light_bulbradius[lid] = e["bulbradius"]
                m.light_active[lid] = 1
            else:
                m.light_diffuse[lid] = 0
                m.light_active[lid] = 0
        self.light_diffuse_ep = m.light_diffuse.copy()
        self.head_ep = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        # materials
        rgba = b.mat_rgba.copy()
        for i in M["white"]:
            rgba[i, :3] = np.clip(b.mat_rgba[i, :3] * p["white_gain"] * np.array(p["white_tint"]), 0, 1)
        for i in M["black"]:
            rgba[i, :3] = np.clip(b.mat_rgba[i, :3] * p["black_gain"], 0, 1)
        rgba[M["cube"], :3] = p["cube_grey"]
        rgba[M["table"], :3] = np.clip(b.mat_rgba[M["table"], :3] * p["table_gain"] * np.array(p["table_tint"]), 0, 1)
        for k in ("floor", "floor_plain"):
            rgba[M[k], :3] = np.clip(b.mat_rgba[M[k], :3] * p["floor_gain"] * np.array(p["floor_tint"]), 0, 1)
        rgba[M["container"], :3] = np.clip(b.mat_rgba[M["container"], :3] * np.array(p["container_tint"]), 0, 1)
        rgba[M["container"], 3] = p["container_alpha"]
        rgba[M["tape"], :3] = np.clip(b.mat_rgba[M["tape"], :3] * np.array(p["tape_tint"]), 0, 1)
        m.mat_rgba[:] = rgba
        # distractors
        kinds = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}
        for i, gid in enumerate(self.obj[:-1]):
            if i < len(p["distractors"]):
                o = p["distractors"][i]
                m.geom_type[gid] = kinds[o["kind"]]
                size = list(o["size"])
                if o["kind"] in ("cylinder", "capsule"):
                    size = [size[0], size[2] if o["kind"] == "cylinder" else size[1], 0]
                m.geom_size[gid] = size
                m.geom_pos[gid] = o["pos"]
                m.geom_quat[gid] = [np.cos(o["yaw"] / 2), 0, 0, np.sin(o["yaw"] / 2)]
                m.geom_rgba[gid] = o["rgba"]
            else:
                m.geom_pos[gid] = [10 + i, 10, 0]
        gid = self.obj[-1]
        if p["cube_in_container"]:
            m.geom_type[gid] = mujoco.mjtGeom.mjGEOM_BOX
            m.geom_size[gid] = [0.02, 0.02, 0.02]
            m.geom_pos[gid] = [*p["container_cube_xy"], 0.022]
            y = p["container_cube_yaw"]
            m.geom_quat[gid] = [np.cos(y / 2), 0, 0, np.sin(y / 2)]
            m.geom_rgba[gid] = [p["cube_grey"]] * 3 + [1]
        else:
            m.geom_pos[gid] = [20, 10, 0]
        # cameras: episode offsets (frame jitter is added per frame)
        self.cam_ep = {}
        for cam, tag in (("ext_cam", "side"), ("wrist_cam", "wrist")):
            c = self.cam[cam]
            self.cam_ep[cam] = (b.cam_pos[c] + np.array(p[f"{tag}_dpos"]),
                                quat_mul(b.cam_quat[c], np.array(p[f"{tag}_drot"])))
        m.cam_fovy[self.cam["ext_cam"]] = b.cam_fovy[self.cam["ext_cam"]] + p["side_dfovy"]
        self.fields = {}

    def _field(self, tag, shape):
        """Smooth multiplicative illumination field (uneven room light), fixed per episode."""
        if tag not in self.fields:
            q = self.p[f"{tag}_img"]
            g = np.random.default_rng(q["field_seed"]).normal(0, 1, (3, 4))
            f = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
            f = 1 + q["field_amp"] * f / max(np.abs(f).max(), 1e-6)
            if q["vignette"] > 0:
                yy, xx = np.mgrid[:shape[0], :shape[1]]
                r2 = ((xx - shape[1] / 2) ** 2 + (yy - shape[0] / 2) ** 2) / ((shape[1] / 2) ** 2 + (shape[0] / 2) ** 2)
                f = f * (1 - q["vignette"] * r2)
            self.fields[tag] = f.astype(np.float32)[..., None]
        return self.fields[tag]

    def look(self, img, tag, rng):
        """Camera/ISP effects on an RGB uint8 frame."""
        q = self.p[f"{tag}_img"]
        x = img.astype(np.float32) / 255.0
        x = x * self._field(tag, x.shape) * q["gain"] * rng.normal(1, 0.01) * np.array(q["wb"], np.float32)
        x = np.clip(x, 0, 1) ** q["gamma"]
        grey = x.mean(2, keepdims=True)
        x = np.clip(grey + (x - grey) * q["sat"], 0, 1)
        s = q["blur"] + rng.normal(0, 0.3)
        if s > 0.3:
            x = cv2.GaussianBlur(x, (0, 0), s)
        x = x * 255.0 + rng.normal(0, q["noise"], x.shape)
        out = np.clip(x, 0, 255).astype(np.uint8)
        if q["jpeg"]:
            ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q["jpeg"]])
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        return np.ascontiguousarray(out)

    def soft_shadows(self, cam, tag, cast):
        """Render with and without shadows; blur and weaken the shadow ratio; recombine."""
        q = self.p[f"{tag}_img"]
        m = self.m
        self.r.update_scene(self.d, camera=cam)
        lit = self.r.render().astype(np.float32)
        m.light_castshadow[:] = 0
        self.r.update_scene(self.d, camera=cam)
        flat = self.r.render().astype(np.float32)
        m.light_castshadow[:] = cast
        ratio = np.clip(lit.mean(2) / np.maximum(flat.mean(2), 1.0), 0, 1)
        ratio = cv2.GaussianBlur(ratio, (0, 0), q["shadow_blur"])
        shade = 1 - q["shadow_strength"] * (1 - ratio)
        return np.clip(flat * shade[..., None], 0, 255).astype(np.uint8)

    def frame(self, state, rng):
        """Both camera images for one demo state, with frame-wise jitter."""
        m = self.m
        mujoco.mj_setState(m, self.d, state, STATE_SPEC)
        m.light_diffuse[:] = self.light_diffuse_ep * rng.normal(1, 0.03, (m.nlight, 1))
        m.vis.headlight.ambient[:] = self.head_ep[0] * rng.normal(1, 0.02)
        for cam, (pos, quat) in self.cam_ep.items():
            c = self.cam[cam]
            if cam == "wrist_cam":
                m.cam_pos[c] = pos + rng.normal(0, 0.0005, 3)
                m.cam_quat[c] = quat_mul(quat, small_rotation(rng, 0.3, 0.6))
            else:
                m.cam_pos[c] = pos
                m.cam_quat[c] = quat_mul(quat, small_rotation(rng, 0.1, 0.2))
        mujoco.mj_forward(m, self.d)
        out = {}
        cast = m.light_castshadow.copy()
        for cam in CAMERAS:
            tag = "wrist" if cam == "wrist_cam" else "side"
            img = self.soft_shadows(cam, tag, cast)
            if cam == "wrist_cam":
                img = orient_wrist(self.lens.distort(img))
                out[cam] = self.look(img, "wrist", rng)
            else:
                out[cam] = self.look(img, "side", rng)
        return out


# -- preview ----------------------------------------------------------------------------------

def preview(R, demo_paths, n, seed, out_path, pairs_dir):
    """Rows of randomised (side | wrist) frames from random demos and moments, with real frames
    from a capture_pairs folder at the top for comparison."""
    rng = np.random.default_rng(seed)
    rows = []
    real = sorted(pathlib.Path(pairs_dir).glob("*_real_side.png")) if pairs_dir else []
    for f in [real[i] for i in np.linspace(0, len(real) - 1, 3).astype(int)] if real else []:
        side = cv2.imread(str(f))[..., ::-1]
        wrist = cv2.imread(str(f).replace("_side", "_wrist"))[..., ::-1]
        rows.append(tile(side, wrist, "REAL"))
    for k in range(n):
        path = demo_paths[int(rng.integers(len(demo_paths)))]
        demo = load_demo(path)
        R.set_episode(sample_episode(rng, demo.obj_pos[0][:2]))
        t = int(rng.integers(len(demo)))
        imgs = R.frame(demo.mj_state[t], rng)
        rows.append(tile(imgs["ext_cam"], imgs["wrist_cam"], f"{pathlib.Path(path).stem} t={t}"))
    grid = []
    for i in range(0, len(rows), 3):
        r = rows[i:i + 3]
        r += [np.zeros_like(rows[0])] * (3 - len(r))
        grid.append(np.hstack(r))
    cv2.imwrite(str(out_path), cv2.cvtColor(np.vstack(grid), cv2.COLOR_RGB2BGR))
    print("wrote", out_path)


def tile(side, wrist, label):
    w = cv2.resize(wrist, (round(wrist.shape[1] * side.shape[0] / wrist.shape[0]), side.shape[0]))
    t = np.ascontiguousarray(np.hstack([side, w]))
    t = cv2.resize(t, None, fx=0.5, fy=0.5)
    cv2.putText(t, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(t, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return t


# -- dataset ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="*", default=["demos/cube/trimmed/*.npz"])
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", default="local/so101-cube-dr")
    ap.add_argument("--root", default="lerobot_dataset/so101-cube-dr")
    ap.add_argument("--scene", default=None, help="override the scene recorded in the demos")
    ap.add_argument("--shard", default=None, metavar="I/N")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--vcodec", default="auto")
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="write a preview grid of N frames and stop")
    ap.add_argument("--pairs", default="calib/pairs/20261123_202747", help="real frames for the preview")
    a = ap.parse_args()

    paths = sorted(p for pat in a.demos for p in glob.glob(str(SIM / pat) if not pathlib.Path(pat).is_absolute() else pat))
    if not paths:
        ap.error("no demos matched")
    first = load_demo(paths[0])
    scene = pathlib.Path(a.scene or first.meta["scene_path"])
    scene = scene if scene.is_absolute() else SIM / scene
    R = Randomiser(scene)
    if a.preview:
        preview(R, paths, a.preview, a.seed, SIM / "eval" / "dr_preview.png", SIM / a.pairs if a.pairs else None)
        return 0

    jobs = [(p, c) for c in range(a.copies) for p in paths]       # copy-major: shards stay balanced
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = jobs[i::n]
    root = SIM / a.root
    if root.exists():
        if not a.overwrite:
            ap.error(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    jmap = joint_units.JointMap.load()
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset.create(repo_id=a.repo_id, root=str(root), fps=round(first.meta["fps"]),
                               features=build_features(joint_units.REAL_NAMES, 640, 480, wrist_rotated=True),
                               robot_type="so101", use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec=a.vcodec))
    log = []
    t0, frames = time.time(), 0
    for k, (path, copy) in enumerate(jobs):
        demo = load_demo(path)
        seed = int(np.random.SeedSequence([a.seed, copy, int(pathlib.Path(path).stem.split("_")[-1])]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        p = sample_episode(rng, demo.obj_pos[0][:2])
        R.set_episode(p)
        for t in range(len(demo)):
            imgs = R.frame(demo.mj_state[t], rng)
            ds.add_frame({"observation.state": np.asarray(jmap.sim_to_real(demo.joint_pos[t]), np.float32),
                          "action": np.asarray(jmap.sim_to_real(demo.action[t]), np.float32),
                          "task": demo.meta["task"],
                          "observation.images.ext_cam": imgs["ext_cam"],
                          "observation.images.wrist_cam": imgs["wrist_cam"]})
        ds.save_episode()
        frames += len(demo)
        log.append({"episode_index": k, "demo": pathlib.Path(path).name, "copy": copy, "seed": seed, "params": p})
        el = time.time() - t0
        print(f"[{k + 1}/{len(jobs)}] {pathlib.Path(path).name} copy {copy}: {len(demo)} frames "
              f"({frames / el:.1f} fps, {el / 60:.0f} min, ~{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left)",
              flush=True)
    ds.finalize()
    shutil.copy(joint_units.MAP_PATH, root / "joint_map.json")
    (root / "meta" / "randomisation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    print(f"done: {len(jobs)} episodes, {frames} frames, {(time.time() - t0) / 60:.0f} min -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
