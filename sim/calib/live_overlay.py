"""Live overlay of the sim side camera (ext_cam) on the real side camera, with the
real arm driving the sim arm. Tweak the sim camera and the LeRobot->MuJoCo joint
offsets with the keyboard until the two line up. Every change is saved at once.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/live_overlay.py                # leader drives follower
    /opt/anaconda3/envs/lerobot/bin/python calib/live_overlay.py --no-leader    # follower limp, move it by hand
    /opt/anaconda3/envs/lerobot/bin/python calib/live_overlay.py --snapshot calib/live_overlay_test.png
                                                    # one frame, no torque, save all view modes, exit

Wrist camera (--cam wrist): the same, for the real wrist camera ('icspring')
against the sim wrist_cam. The camera keys then move/rotate wrist_cam in its
own frame (w/s along the view, a/d sideways, r/f or t/b up/down in the image).
Only the camera element moves: the mount, PCB and lens geoms stay where they
are, so the side camera sees the gripper exactly as before. --cube X Y puts the
sim cube at robot-frame (X, Y) metres: put the real cube there to compare.
X writes wrist_cam into so101_real.xml (included by both real-look scenes).

    /opt/anaconda3/envs/lerobot/bin/python calib/live_overlay.py --cam wrist --cube 0.20 0

Placing the real container (side camera): the sim container's base footprint is drawn
in orange and the tape band's edges in red (both projected from the scene, so they're
exact even where the transparent shell covers them). Move the real container and tape
until they sit on the lines. --no-arm runs without the arms (sim arm in its 'ready'
pose); the container's tape corners are also printed in robot-frame millimetres.

    /opt/anaconda3/envs/lerobot/bin/python calib/live_overlay.py --no-arm

The sim pose always comes from the follower's *measured* joints (not the
leader's commands), converted with joint_units.py. Every save also refreshes
calib/joint_map.json, which the dataset export and eval use.

Keys (the overlay window must have focus):
  Tab       switch the movement keys: CAMERA -> LIGHT -> CONTAINER (shown at the top
            of the panel). Table, joint and general keys work in all of them.
  camera (moves keep the aim; rotations are about the camera's own axes)
    w / s   toward / away from the robot, parallel to the floor
    a / d   left / right, parallel to the floor
    t / b   straight up / down
    r / f   up / down along the image's up direction (diagonal: follows the tilt)
    j / l   pan left / right    i / k   tilt up / down      u / o   roll
    z / x   fovy - / +
  light (after Tab; the scene's one shadow-casting light, "ceiling")
    w s a d t b r f   move the light, same directions as the camera keys,
                      10x the camera step. Its position sets where shadows fall.
    i / k   tilt the beam away from / toward the camera
    j / l   tilt the beam left / right
    z / x   light brightness - / +      (directional light: shadow contrast)
    u / o   ambient - / +               (fills shadows; brightens everything)
    y / h   shadow softer / sharper     (bulb radius x1.25 / /1.25)
  container (after Tab twice; resize the sim container live, one edge at a time)
    w/s/a/d pick the edge as seen in the side image: top (back end, near the base) /
            bottom (front end) / left (outer side) / right (inner side, toward the robot)
    x / z   move that edge out (bigger) / in (smaller) by 1 mm ([ ] change the step).
            The opposite edge stays put; the tape follows. Each step rebuilds the container
            (container/build_container.py: meshes, collision, success test, scene XML) and
            saves the size into that script's LENGTH / WIDTH / PLACED_* variables.
            Bksp undoes resizes too.
  table
    v / c   table's back edge toward / away from the camera. The robot base's back
            is at x = -0.031, so a table edge there means the base is flush with it.
  joint offsets
    1..6    select joint (pan, lift, elbow, wrist_flex, wrist_roll, gripper)
    - / =   offset of the selected joint - / +
  general
    [ / ]   all step sizes /2 / x2
    m       view mode: outline -> blend -> real -> sim -> separate
            (separate = real in the main window, sim in a second window)
    g       grab: save the real frame + measured joints to calib/live_grabs/
    Bksp    undo the last change (any number of steps, this session)
    J       (shift-j) reset only the joint offsets, to side_cam_fit.json (camera untouched)
    0       reset everything to the values this session started with
    F       (shift-f) reset to the original fit: the camera before this tool
            first wrote the XML, plus the offsets in side_cam_fit.json
    X       (shift-x) write the camera, table and light into pick_place_scene_real.xml
    q / Esc quit. In teleop mode the follower's torque is released, so put the
            arms at rest first.

Values are kept in calib/live_overlay.json and loaded on the next start (use
--fresh to start from the scene XML + side_cam_fit.json instead).
"""
import argparse
import json
import re
import shutil
import threading
import time
from pathlib import Path

import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joint_units  # noqa: E402  (sim/joint_units.py: the one real<->sim joint conversion)
from cameras import NAMES, Grabber  # noqa: E402  (calib/cameras.py: open cameras by name)

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
STATE = HERE / "live_overlay.json"
ROBOT_XML = HERE.parent / "mujoco_menagerie/robotstudio_so101/so101_real.xml"
WRIST_PAT = re.compile(r'<camera name="wrist_cam"[^>]*/>', re.S)
FIT = HERE / "side_cam_fit.json"
GRABS = HERE / "live_grabs"
W, H = 640, 480
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FOLLOWER_PORT = "/dev/tty.usbmodem5B790809571"
LEADER_PORT = "/dev/tty.usbmodem5B790804261"
MODES = ["outline", "blend", "real", "sim", "separate"]


# ---------------------------------------------------------------- geometry
def rodrigues(axis, ang):
    axis = np.asarray(axis, float)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k


def orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def xyaxes_to_R(xy):
    x = np.asarray(xy[:3], float); x /= np.linalg.norm(x)
    y = np.asarray(xy[3:], float); y -= x * (x @ y); y /= np.linalg.norm(y)
    return np.c_[x, y, np.cross(x, y)]      # columns = camera x, y, z axes in world


def euler_readout(R):
    """pitch-down, yaw (heading of the view direction), roll, in degrees."""
    f = -R[:, 2]                           # MuJoCo cameras look along -z
    pitch = np.degrees(np.arcsin(np.clip(-f[2], -1, 1)))
    yaw = np.degrees(np.arctan2(f[1], f[0]))
    roll = np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    return pitch, yaw, roll


# ---------------------------------------------------------------- state
def table_edge_x(xml):
    """x of the table's back edge (the edge behind the robot) in a scene XML text."""
    body = re.search(r'<body name="table"([^>]*)>', xml).group(1)
    bpos = re.search(r'pos="([^"]*)"', body)
    top = re.search(r'<geom name="table_top"[^>]*/>', xml).group(0)
    tx = float(re.search(r'pos="([^"]*)"', top).group(1).split()[0])
    sx = float(re.search(r'size="([^"]*)"', top).group(1).split()[0])
    return (float(bpos.group(1).split()[0]) if bpos else 0.0) + tx - sx


LIGHT_PAT = re.compile(r'<light name="ceiling"[^>]*/>', re.S)
HEAD_PAT = re.compile(r'<headlight[^>]*/>', re.S)


def _attr(el, k):
    return [float(v) for v in re.search(rf'\b{k}="([^"]*)"', el).group(1).split()]


def light_from_xml(xml):
    """The 'ceiling' light and the headlight ambient, from a scene XML text."""
    el, head = LIGHT_PAT.search(xml).group(0), HEAD_PAT.search(xml).group(0)
    return dict(pos=_attr(el, "pos"), dir=_attr(el, "dir"), diffuse=_attr(el, "diffuse")[0],
                radius=_attr(el, "bulbradius")[0], ambient=_attr(head, "ambient")[0])


def quat_to_R(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def R_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def wrist_from_xml(xml):
    el = WRIST_PAT.search(xml).group(0)
    if re.search(r'\bfovy="', el):
        fovy = _attr(el, "fovy")[0]
    else:                                   # calibrated intrinsics (focalpixel) instead of fovy
        fovy = float(np.degrees(2 * np.arctan(_attr(el, "resolution")[1] / 2 / _attr(el, "focalpixel")[1])))
    return dict(pos=_attr(el, "pos"), quat=_attr(el, "quat"), fovy=fovy)


class Params:
    def __init__(self, pos, R, fovy, offs, table_x, light, wrist=None):
        self.pos, self.R, self.fovy, self.offs = np.array(pos, float), np.array(R, float), float(fovy), dict(offs)
        self.table_x = float(table_x)          # x of the table's back edge (robot base back is at -0.031)
        self.light = dict(light, pos=np.array(light["pos"], float),
                          dir=np.array(light["dir"], float) / np.linalg.norm(light["dir"]))
        w = wrist or wrist_from_xml(ROBOT_XML.read_text())   # wrist_cam, in the camera_mount body frame
        self.wrist = dict(pos=np.array(w["pos"], float),
                          R=quat_to_R(w["quat"]) if "quat" in w else np.array(w["R"], float),
                          fovy=float(w["fovy"]))

    def copy(self):
        return Params(self.pos.copy(), self.R.copy(), self.fovy, self.offs, self.table_x,
                      dict(self.light, pos=self.light["pos"].copy(), dir=self.light["dir"].copy()),
                      dict(pos=self.wrist["pos"].copy(), R=self.wrist["R"].copy(), fovy=self.wrist["fovy"]))

    def xyaxes(self):
        return np.r_[self.R[:, 0], self.R[:, 1]]

    def to_json(self):
        return dict(pos=[round(v, 5) for v in self.pos.tolist()],
                    xyaxes=[round(v, 5) for v in self.xyaxes().tolist()],
                    fovy=round(self.fovy, 3),
                    joint_offsets_deg={j: round(self.offs[j], 3) for j in JOINTS},
                    table_back_edge_x=round(self.table_x, 4),
                    light=dict(pos=[round(v, 4) for v in self.light["pos"].tolist()],
                               dir=[round(v, 4) for v in self.light["dir"].tolist()],
                               diffuse=round(self.light["diffuse"], 4), radius=round(self.light["radius"], 4),
                               ambient=round(self.light["ambient"], 4)),
                    wrist_cam=dict(pos=[round(v, 5) for v in self.wrist["pos"].tolist()],
                                   quat=[round(v, 6) for v in R_to_quat(self.wrist["R"]).tolist()],
                                   fovy=round(self.wrist["fovy"], 3)),
                    updated=time.strftime("%Y-%m-%d %H:%M:%S"))

    def save(self):
        STATE.write_text(json.dumps(self.to_json(), indent=2))
        joint_units.update_map(quiet=True)     # keep calib/joint_map.json (export/eval) in sync


def initial_params(m, fresh):
    if STATE.exists() and not fresh:
        s = json.loads(STATE.read_text())
        print("loaded", STATE.name, "from", s.get("updated", "?"))
        xml = SCENE.read_text()
        return Params(s["pos"], xyaxes_to_R(s["xyaxes"]), s["fovy"], s["joint_offsets_deg"],
                      s.get("table_back_edge_x", table_edge_x(xml)), s.get("light", light_from_xml(xml)),
                      s.get("wrist_cam"))
    cam = m.camera("ext_cam").id
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, m.cam_quat[cam])
    offs = {j: 0.0 for j in JOINTS}
    if FIT.exists():
        offs.update(json.loads(FIT.read_text())["joint_offsets_deg"])
    print("starting from the scene XML camera and", FIT.name)
    xml = SCENE.read_text()
    return Params(m.cam_pos[cam].copy(), Rm.reshape(3, 3), m.cam_fovy[cam], offs, table_edge_x(xml),
                  light_from_xml(xml))


def original_params():
    """Cameras, table and light as they were before this tool ever wrote the XMLs, plus the fitted offsets."""
    bak = SCENE.with_suffix(".xml.before_live_overlay")
    xml = (bak if bak.exists() else SCENE).read_text()
    el = re.search(r'<camera name="ext_cam"[^>]*/>', xml, re.S).group(0)
    attr = lambda k: [float(v) for v in re.search(rf'{k}="([^"]*)"', el).group(1).split()]
    offs = {j: 0.0 for j in JOINTS}
    if FIT.exists():
        offs.update(json.loads(FIT.read_text())["joint_offsets_deg"])
    rbak = ROBOT_XML.with_suffix(".xml.before_live_overlay")
    wrist = wrist_from_xml((rbak if rbak.exists() else ROBOT_XML).read_text())
    return Params(attr("pos"), xyaxes_to_R(attr("xyaxes")), attr("fovy")[0], offs, table_edge_x(xml),
                  light_from_xml(xml), wrist)


def write_xml(p):
    xml = SCENE.read_text()
    pat = re.compile(r'<camera name="ext_cam"[^>]*/>', re.S)
    if not pat.search(xml):
        print("ext_cam element not found in", SCENE.name, "- nothing written")
        return
    bak = SCENE.with_suffix(".xml.before_live_overlay")
    if not bak.exists():
        shutil.copy(SCENE, bak)
        print("backup:", bak.name)
    pos = " ".join(f"{v:.4f}" for v in p.pos)
    xy = p.xyaxes()
    xys = " ".join(f"{v:.4f}" for v in xy[:3]) + "  " + " ".join(f"{v:.4f}" for v in xy[3:])
    new = (f'<camera name="ext_cam" mode="fixed" pos="{pos}"\n'
           f'      xyaxes="{xys}" fovy="{p.fovy:.2f}"/>')
    xml = pat.sub(new, xml, count=1)
    # move the whole table body so its back edge lands at p.table_x (the geoms themselves stay put)
    shift = p.table_x - (table_edge_x(xml) - _table_body_x(xml))
    xml = re.sub(r'<body name="table"[^>]*>', f'<body name="table" pos="{shift:.4f} 0 0">', xml, count=1)
    L = p.light
    el = LIGHT_PAT.search(xml).group(0)
    for k, v in (("pos", " ".join(f"{x:.4f}" for x in L["pos"])), ("dir", " ".join(f"{x:.4f}" for x in L["dir"])),
                 ("diffuse", " ".join([f"{L['diffuse']:.4f}"] * 3)), ("bulbradius", f"{L['radius']:.4f}")):
        el = re.sub(rf'\b{k}="[^"]*"', f'{k}="{v}"', el)
    xml = LIGHT_PAT.sub(lambda _: el, xml, count=1)
    head = re.sub(r'\bambient="[^"]*"', 'ambient="' + " ".join([f"{L['ambient']:.4f}"] * 3) + '"',
                  HEAD_PAT.search(xml).group(0))
    xml = HEAD_PAT.sub(lambda _: head, xml, count=1)
    SCENE.write_text(xml)
    print("light:", el.replace("\n", " "))
    print("wrote ext_cam to", SCENE.name, ":", pos, "|", xys, "| fovy", f"{p.fovy:.2f}")
    print(f"table back edge at x = {p.table_x:.4f} (table body pos x {shift:+.4f})")


def write_wrist_xml(p):
    xml = ROBOT_XML.read_text()
    if not WRIST_PAT.search(xml):
        print("wrist_cam element not found in", ROBOT_XML.name, "- nothing written")
        return
    bak = ROBOT_XML.with_suffix(".xml.before_live_overlay")
    if not bak.exists():
        shutil.copy(ROBOT_XML, bak)
        print("backup:", bak.name)
    w = p.wrist
    pos = " ".join(f"{v:.5f}" for v in w["pos"])
    quat = " ".join(f"{v:.6f}" for v in R_to_quat(w["R"]))
    el = WRIST_PAT.search(xml).group(0)     # only pos + quat change: the calibrated lens attributes stay
    new = re.sub(r'\bpos="[^"]*"', f'pos="{pos}"', el)
    new = re.sub(r'\bquat="[^"]*"', f'quat="{quat}"', new)
    ROBOT_XML.write_text(xml.replace(el, new, 1))
    print("wrote wrist_cam pose to", ROBOT_XML.name, ":", pos, "|", quat)


def _table_body_x(xml):
    m = re.search(r'pos="([^"]*)"', re.search(r'<body name="table"([^>]*)>', xml).group(1))
    return float(m.group(1).split()[0]) if m else 0.0


# ---------------------------------------------------------------- sim
class Sim:
    def __init__(self, view="side", cube=None, quiet=False):
        self.quiet = quiet
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, H)
        self.d = mujoco.MjData(self.m)
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.cam = self.m.camera("ext_cam").id
        self.wcam = self.m.camera("wrist_cam").id
        self.view = view
        # the real wrist lens (calib/intrinsics/wrist.json): wrist_cam has its intrinsics in
        # so101_real.xml, the distortion is applied to the render here
        self.lens = None
        if self.m.cam_sensorsize[self.wcam][0] > 0:
            from wrist_lens import WristLens
            self.lens = WristLens()
        self.rgb = mujoco.Renderer(self.m, H, W)
        self.seg = mujoco.Renderer(self.m, H, W)
        self.seg.enable_segmentation_rendering()
        body = self.m.geom_bodyid
        lo, hi = self.m.body("base").id, self.m.body("moving_jaw_so101_v1").id
        self.robot_geoms = np.flatnonzero((body >= lo) & (body <= hi))
        self.table_geom = self.m.geom("table_top").id
        self.table_body = self.m.body("table").id
        self.light = self.m.light("ceiling").id
        g = self.table_geom                    # back edge x when the table body sits at x = 0
        self.table_edge0 = self.m.geom_pos[g][0] - self.m.geom_size[g][0]
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS]
        self.grip_span = joint_units.build_map()["gripper_span_deg"]
        # the task object: at --cube X Y [yaw deg] if given (where the real one is), else far away
        self.obj_geoms = np.array([], int)
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = self.m.jnt_qposadr[j]
                if cube is None:
                    self.d.qpos[a] = 5.0
                else:
                    x, y = cube[0], cube[1]
                    yaw = np.radians(cube[2]) if len(cube) > 2 else 0.0
                    self.d.qpos[a:a + 7] = [x, y, self.d.qpos[a + 2], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                    self.obj_geoms = np.flatnonzero(self.m.geom_bodyid == self.m.jnt_bodyid[j])

        self.container = self._container_outlines()

    def _container_outlines(self):
        """World-frame outlines of the container (sim/container/build_container.py): its base
        footprint and the tape band's outer and inner edges, from the mesh vertices."""
        if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "container") < 0:
            return []
        mujoco.mj_kinematics(self.m, self.d)
        b = self.m.body("container").id
        bpos, bmat = self.d.xpos[b], self.d.xmat[b].reshape(3, 3)

        def verts(name):
            g = self.m.geom(name).id
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            w = self.d.geom_xpos[g] + v @ self.d.geom_xmat[g].reshape(3, 3).T
            return w, (w - bpos) @ bmat                    # world, container frame

        def ring(w, loc, sel):
            pts, l2 = w[sel], loc[sel]
            o = np.argsort(np.arctan2(l2[:, 1], l2[:, 0]))
            return np.unique(np.round(pts[o], 6), axis=0, return_index=True)[1], pts[o]

        out = []
        w, loc = verts("container_shell")
        base = (np.abs(loc[:, 2]) < 1e-5) & (np.linalg.norm(loc[:, :2], axis=1) > 1e-4)
        idx, pts = ring(w, loc, base)
        out.append((pts[np.sort(idx)], (0, 165, 255)))                 # base footprint: orange
        w, loc = verts("container_tape")
        top = loc[:, 2] > loc[:, 2].max() - 1e-5
        r = np.linalg.norm(loc[:, :2], axis=1)
        for sel in (top & (r > np.median(r[top])), top & (r <= np.median(r[top]))):
            idx, pts = ring(w, loc, sel)
            out.append((pts[np.sort(idx)], (0, 0, 255)))                # tape edges: red
        if not self.quiet:
          print("container (robot frame, mm): centre", [int(round(v * 1000)) for v in bpos[:2]],
              "| tape outer corners", [(int(round(c[0] * 1000)), int(round(c[1] * 1000))) for c in out[1][0]])
        return out[1:] + out[:1]                   # footprint drawn last, on top of the tape's inner edge

    def project(self, pts, fovy):
        """World points -> side-camera pixels (pinhole, principal point at the centre)."""
        p, R = self.d.cam_xpos[self.cam], self.d.cam_xmat[self.cam].reshape(3, 3)
        c = (pts - p) @ R
        f = (H / 2) / np.tan(np.radians(fovy) / 2)
        return np.stack([W / 2 + f * c[:, 0] / -c[:, 2], H / 2 - f * c[:, 1] / -c[:, 2]], 1)

    def close(self):
        self.rgb.close()
        self.seg.close()

    def render(self, p, lerobot):
        self.m.cam_pos[self.cam] = p.pos
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, p.R.flatten())
        self.m.cam_quat[self.cam] = q
        self.m.cam_fovy[self.cam] = p.fovy
        self.m.cam_pos[self.wcam] = p.wrist["pos"]          # only the camera element: no geom moves
        self.m.cam_quat[self.wcam] = R_to_quat(p.wrist["R"])
        self.m.cam_fovy[self.wcam] = p.wrist["fovy"]
        if lerobot is not None:
            q = joint_units.JointMap(p.offs, self.grip_span).real_dict_to_sim(lerobot)
            for a, v in zip(self.qadr, q):
                self.d.qpos[a] = v
        self.m.body_pos[self.table_body][0] = p.table_x - self.table_edge0
        L, li = p.light, self.light
        self.m.light_pos[li] = L["pos"]
        self.m.light_dir[li] = L["dir"]
        self.m.light_diffuse[li] = [L["diffuse"]] * 3
        self.m.light_bulbradius[li] = L["radius"]
        self.m.vis.headlight.ambient[:] = [L["ambient"]] * 3
        mujoco.mj_forward(self.m, self.d)
        cam = self.wcam if self.view == "wrist" else self.cam
        self.rgb.update_scene(self.d, camera=cam)
        img = self.rgb.render()[..., ::-1].copy()          # BGR for OpenCV
        self.seg.update_scene(self.d, camera=cam)
        s = self.seg.render()
        geom = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        if self.view == "wrist" and self.lens is not None:
            img = self.lens.distort(img)
            geom = self.lens.distort(geom.astype(np.float32), nearest=True).astype(int)
        self.obj_mask = np.isin(geom, self.obj_geoms)
        self.lines = ([] if self.view == "wrist" else
                      [(self.project(pts, p.fovy), col) for pts, col in self.container])
        return img, np.isin(geom, self.robot_geoms), geom == self.table_geom


# ---------------------------------------------------------------- camera
# ---------------------------------------------------------------- robot
def configure_with_retry(robot, tries=3):
    """robot.configure() writes every motor once with no retry; a single dropped status packet
    ('There is no status packet!') then aborts start-up. Retry a few times before giving up."""
    for k in range(tries):
        try:
            robot.configure()
            return
        except ConnectionError as e:
            if k == tries - 1:
                raise
            print(f"motor bus glitch during configure ({e}); retrying")
            time.sleep(0.3)


class Arms:
    def __init__(self, follower_port, leader_port, teleop, max_step):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.teleop = teleop
        self.robot = SO101Follower(SO101FollowerConfig(port=follower_port, id="so101_follower",
                                                       max_relative_target=max_step))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit("follower motors don't match its calibration file -- run lerobot-calibrate first")
        self.leader = None
        if teleop:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
            self.leader = SO101Leader(SO101LeaderConfig(port=leader_port, id="so101_leader"))
            self.leader.bus.connect()
            if not self.leader.bus.is_calibrated:
                self.close()
                raise SystemExit("leader motors don't match its calibration file -- run lerobot-calibrate first")
            self.leader.configure()                       # leader: torque off
            self.bus.disable_torque()
            p0 = self.bus.sync_read("Present_Position")
            self.bus.sync_write("Goal_Position", p0)      # torque comes back on holding exactly here
            configure_with_retry(self.robot)
            print(f"teleop on: leader {leader_port} -> follower {follower_port}, "
                  f"max {max_step} deg per step")
        else:
            self.bus.disable_torque()
            print("follower torque OFF: move it by hand")

    def step(self):
        if self.leader is not None:
            self.robot.send_action(self.leader.get_action())
        return self.bus.sync_read("Present_Position", num_retry=2)

    def close(self):
        if self.leader is not None:
            self.leader.bus.disconnect(disable_torque=True)
        self.bus.disconnect(disable_torque=True)


class ContainerEdit:
    """Container mode (Tab): resize the sim container one edge at a time, live. Each change
    rebuilds it with sim/container/build_container.py (meshes, tape, collision, success test,
    written into the scene) and stores the size in that script's LENGTH / WIDTH / PLACED_*
    variables. The edge opposite the one you move stays put."""
    # key -> (name, axis, direction of "outward"): edges as seen in the side camera image
    EDGES = {"w": ("back end (top, near base)", 0, -1), "s": ("front end (bottom)", 0, +1),
             "a": ("outer side (left)", 1, -1), "d": ("inner side (right, robot)", 1, +1)}
    MIN_SIZE = 0.05

    def __init__(self, sim):
        sys.path.insert(0, str(HERE.parent / "container"))
        import build_container
        self.bc = build_container
        self.sel = "s"
        base = sim.container[-1][0]                       # base footprint, world frame
        self.lo = np.round(base[:, :2].min(0), 4)         # (xmin, ymin)
        self.hi = np.round(base[:, :2].max(0), 4)         # (xmax, ymax)

    def edges(self):
        return self.lo.copy(), self.hi.copy()

    def set_edges(self, e):
        self.lo, self.hi = e[0].copy(), e[1].copy()

    def size(self):
        return self.hi - self.lo                          # (length, width)

    def nudge(self, outward, step):
        _, ax, d = self.EDGES[self.sel]
        delta = step * (1 if outward else -1)
        if d > 0:
            self.hi[ax] = max(self.hi[ax] + delta, self.lo[ax] + self.MIN_SIZE)
        else:
            self.lo[ax] = min(self.lo[ax] - delta, self.hi[ax] - self.MIN_SIZE)

    def apply(self):
        """Rebuild the container in the scene at the current edges (the Sim must then reload)."""
        a = self.bc.default_args()
        a.length, a.width = map(float, self.size())
        a.pos = tuple(float(v) for v in (self.lo + self.hi) / 2)
        self.bc.build(a)
        self.bc.save_size(a.length, a.width, a.pos)


class NoArms:
    """--no-arm: no robot connected; the sim arm stays in the scene's 'ready' pose."""
    teleop = False

    def step(self):
        return None

    def close(self):
        pass


# ---------------------------------------------------------------- drawing
def compose(mode, real, sim, robot_m, table_m, scale, obj_m=None, lines=()):
    dw, dh = int(W * scale), int(H * scale)
    real = cv2.resize(real, (dw, dh), interpolation=cv2.INTER_LINEAR)
    sim = cv2.resize(sim, (dw, dh), interpolation=cv2.INTER_LINEAR)
    if mode in ("real", "separate"):
        return real
    if mode == "sim":
        return sim
    if mode == "blend":
        out = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
        draw_lines(out, lines, scale)
        return out
    out = real.copy()
    rm = cv2.resize(robot_m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    tm = cv2.resize(table_m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    tint = out.copy()
    tint[rm > 0] = (0, 200, 0)
    out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)
    outlines = [(tm, (255, 200, 0)), (rm, (0, 255, 0))]
    if obj_m is not None and obj_m.any():
        outlines.append((cv2.resize(obj_m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST), (255, 0, 255)))
    for mask, col in outlines:
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 1, cv2.LINE_AA)
    draw_lines(out, lines, scale)
    return out


def draw_lines(img, lines, scale):
    """Closed outlines in camera pixels (e.g. the container footprint and tape edges)."""
    for pts, col in lines:
        q = np.round(np.asarray(pts) * scale * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [q], True, col, 1, cv2.LINE_AA, shift=2)


def brightness(real, sim, robot_m, table_m):
    """Mean grey level of real vs sim, over the sim's table (minus robot) and robot pixels."""
    out = {}
    for name, m in (("table", table_m & ~robot_m), ("robot", robot_m)):
        if m.sum() > 200:
            out[name] = (float(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)[m].mean()),
                         float(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY)[m].mean()))
    return out


def panel(height, p, sel, steps, mode, fps, joints, teleop, msg, target, bright, view="side", p_orig=None,
          cont=None):
    joints = joints or {}                       # --no-arm: no measured joints
    img = np.full((height, 360, 3), 24, np.uint8)
    y = [22]

    def line(txt, col=(220, 220, 220), dy=20):
        cv2.putText(img, txt, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        y[0] += dy

    pitch, yaw, roll = euler_readout(p.R)
    line(f"KEYS: {target.upper()}   (Tab to switch)",
         {"camera": (0, 255, 255), "light": (80, 180, 255)}.get(target, (0, 165, 255)))
    line(f"mode: {mode}   {fps:4.1f} fps", (120, 220, 255))
    line("TELEOP leader->follower" if teleop else "follower limp (by hand)", (120, 255, 120) if teleop else (0, 200, 255))
    line("")
    if view == "wrist":
        w = p.wrist
        line("WRIST CAMERA wrist_cam (in camera_mount frame)", (255, 220, 120))
        line(f" pos mm  x {w['pos'][0]*1000:+.1f} y {w['pos'][1]*1000:+.1f} z {w['pos'][2]*1000:+.1f}")
        line(f" lens: calibrated (intrinsics + distortion), fovy {w['fovy']:.1f} fixed")
        if p_orig is not None:
            o = p_orig.wrist
            dpos = np.linalg.norm(w["pos"] - o["pos"]) * 1000
            dang = np.degrees(np.arccos(np.clip((np.trace(o["R"].T @ w["R"]) - 1) / 2, -1, 1)))
            line(f" vs original fit: moved {dpos:.1f} mm, turned {dang:.2f} deg, fovy {w['fovy'] - o['fovy']:+.2f}")
    else:
        line("CAMERA ext_cam", (255, 220, 120))
        line(f" pos  x {p.pos[0]:+.4f} y {p.pos[1]:+.4f} z {p.pos[2]:+.4f}")
        line(f" pitch-down {pitch:6.2f}  yaw {yaw:7.2f}")
        line(f" roll {roll:+6.2f}    fovy {p.fovy:6.2f}")
    line(f" table back edge x {p.table_x:+.4f}  (flush: -0.031)")
    line(f" steps: move {steps['pos']*1000:.1f} mm, rot {steps['ang']:.2f} deg")
    L = p.light
    tilt = np.degrees(np.arccos(np.clip(-L["dir"][2], -1, 1)))
    line("LIGHT ceiling", (255, 220, 120))
    line(f" pos  x {L['pos'][0]:+.3f} y {L['pos'][1]:+.3f} z {L['pos'][2]:+.3f}")
    line(f" beam {tilt:4.1f} deg off vertical   step {steps['pos']*10*1000:.0f} mm")
    line(f" bright {L['diffuse']:.3f}  ambient {L['ambient']:.3f}  soft {L['radius']:.3f}")
    if bright:
        line("  grey real/sim: " + "  ".join(f"{k} {r:.0f}/{s_:.0f}" for k, (r, s_) in bright.items()))
    if cont is not None:
        (x0, y0), (x1, y1) = cont.lo, cont.hi
        L_, W_ = cont.size()
        line(f"CONTAINER {L_ * 1000:.1f} x {W_ * 1000:.1f} mm  step {steps['pos'] / 5 * 1000:g}",
             (0, 165, 255) if target == "container" else (255, 220, 120))
        line(f" x {x0 * 1000:.1f}..{x1 * 1000:.1f}  y {y0 * 1000:.1f}..{y1 * 1000:.1f} mm")
        if target == "container":
            line(f" moving: {cont.EDGES[cont.sel][0]}", (0, 255, 255))
    line("")
    line("JOINT OFFSETS (deg)   offset | real", (255, 220, 120))
    for i, j in enumerate(JOINTS):
        cur = i == sel
        line(f"{'>' if cur else ' '}{i+1} {j:<13} {p.offs[j]:+7.2f} | {joints.get(j, float('nan')):7.1f}",
             (0, 255, 255) if cur else (220, 220, 220))
    line(f" step {steps['joint']:.2f} deg")
    line("")
    move_help = ((["WRIST: w/s along view  a/d side  r/f up/down", "j/l i/k u/o  pan tilt roll   z/x fovy"]
                  if view == "wrist" else
                  ["w/s fwd/back  a/d side  t/b up/down (level)", "r/f up/down along image (diagonal)",
                   "j/l i/k u/o  pan tilt roll   z/x fovy"]) if target == "camera" else
                 ["CONTAINER: w/s/a/d pick the edge (as in image)", "x / z  edge out (bigger) / in (smaller)"]
                 if target == "container" else
                 ["LIGHT: wsad t/b r/f move (10x step)", "i/k beam away/toward cam  j/l left/right",
                  "z/x bright  u/o ambient  y/h softer/sharper"])
    for t in move_help + ["1-6 joint  -/= offset",
              "[ ] step   m mode   g grab", "Bksp undo  0 reset session  F reset fit",
              "J reset joint offsets only",
              "c/v table edge   X write XML   q quit"]:
        line(t, (150, 150, 150), 18)
    line("")
    line("green = sim robot, cyan = sim table, magenta = cube", (150, 150, 150), 18)
    line("orange = container base, red = tape edges", (150, 150, 150), 18)
    if msg:
        line(msg, (0, 255, 255))
    return img


def apply_key(k, p, sel, steps, target="camera", view="side"):
    """Returns (changed, new_sel)."""
    c = chr(k) if 0 < k < 256 else ""
    s, a = steps["pos"], np.radians(steps["ang"])
    if view == "wrist" and target == "camera" and c and c in "wsadrftbijkluozx":
        w = p.wrist                             # everything in the camera's own frame, 1/5 of the side steps
        Rw, sw = w["R"], s / 5
        mv = {"d": Rw[:, 0], "a": -Rw[:, 0], "r": Rw[:, 1], "t": Rw[:, 1], "f": -Rw[:, 1], "b": -Rw[:, 1],
              "w": -Rw[:, 2], "s": Rw[:, 2]}
        rt = {"j": ([0, 1, 0], a), "l": ([0, 1, 0], -a), "i": ([1, 0, 0], a), "k": ([1, 0, 0], -a),
              "u": ([0, 0, 1], a), "o": ([0, 0, 1], -a)}
        if c in mv:
            w["pos"] = w["pos"] + mv[c] * sw
        elif c in rt:
            w["R"] = orthonormal(Rw @ rodrigues(*rt[c]))
        else:
            return False, sel                  # the lens is calibrated: field of view is not a tuning knob
        return True, sel
    R = p.R
    # "forward along the floor": view direction + image-up direction, flattened. Both point
    # toward what the camera looks at for any tilt, and the sum stays defined looking straight down.
    fwd = (R[:, 1] - R[:, 2]) * [1, 1, 0]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])           # horizontal, image-right
    moves = {"d": right * s, "a": -right * s, "r": R[:, 1] * s, "f": -R[:, 1] * s,
             "w": fwd * s, "s": -fwd * s,
             "t": np.array([0, 0, s]), "b": np.array([0, 0, -s])}     # world vertical, aim unchanged
    if c in "cv" and c:                        # table back edge toward the camera (v) / away (c)
        p.table_x += s if c == "v" else -s
        return True, sel
    if target == "light" and c and c in "wsadtbrfijklzxuoyh":
        L = p.light
        if c in moves:
            L["pos"] = L["pos"] + moves[c] * 10
        elif c in "ijkl":                      # beam tilt, about horizontal axes tied to the camera view
            axis, ang = {"i": (right, 4 * a), "k": (right, -4 * a), "j": (fwd, 4 * a), "l": (fwd, -4 * a)}[c]
            L["dir"] = rodrigues(axis, ang) @ L["dir"]
            L["dir"] /= np.linalg.norm(L["dir"])
        elif c in "zx":
            L["diffuse"] = float(np.clip(L["diffuse"] + (0.01 if c == "x" else -0.01), 0, 1))
        elif c in "uo":
            L["ambient"] = float(np.clip(L["ambient"] + (0.01 if c == "o" else -0.01), 0, 1))
        elif c in "yh":
            L["radius"] = float(np.clip(L["radius"] * (1.25 if c == "y" else 0.8), 0.005, 2.0))
        return True, sel
    rots = {"j": ([0, 1, 0], a), "l": ([0, 1, 0], -a), "i": ([1, 0, 0], a), "k": ([1, 0, 0], -a),
            "u": ([0, 0, 1], a), "o": ([0, 0, 1], -a)}
    if c in moves:
        p.pos += moves[c]
    elif c in rots:
        p.R = orthonormal(R @ rodrigues(*rots[c]))       # rotation about the camera's own axes
    elif c in "zx" and c:
        p.fovy = float(np.clip(p.fovy + (steps["ang"] if c == "x" else -steps["ang"]), 10, 120))
    elif c in "-=_+" and c:
        p.offs[JOINTS[sel]] += steps["joint"] * (1 if c in "=+" else -1)
    elif c in "123456" and c:
        return False, int(c) - 1
    else:
        return False, sel
    return True, sel


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", choices=("side", "wrist"), default="side", help="which camera to overlay")
    ap.add_argument("--camera", default=None,
                    help="camera name fragment or OpenCV index (default: 'C920' for side, "
                         "'icspring' for wrist; python calib/cameras.py lists them)")
    ap.add_argument("--cube", type=float, nargs="+", metavar="X_Y_[YAW]",
                    help="put the sim cube at robot-frame X Y metres (optional yaw, deg); else hidden")
    ap.add_argument("--follower-port", default=FOLLOWER_PORT)
    ap.add_argument("--leader-port", default=LEADER_PORT)
    ap.add_argument("--no-leader", action="store_true", help="no teleop: follower torque off, move it by hand")
    ap.add_argument("--no-arm", action="store_true",
                    help="no arms connected: sim arm in its 'ready' pose (e.g. for placing the container)")
    ap.add_argument("--max-step", type=float, default=4.0,
                    help="teleop: max follower move per loop step, deg (about 30 steps/s)")
    ap.add_argument("--scale", type=float, default=1.5, help="display size of the 640x480 view")
    ap.add_argument("--fresh", action="store_true", help=f"ignore {STATE.name}, start from XML + fit")
    ap.add_argument("--snapshot", help="save one frame in all view modes to this PNG and exit (no torque)")
    args = ap.parse_args()

    sim = Sim(args.cam, args.cube)
    cont = ContainerEdit(sim) if sim.container else None
    p = initial_params(sim.m, args.fresh)
    p0 = p.copy()
    p_orig = original_params()
    cam = Grabber(args.camera if args.camera is not None else NAMES[args.cam])
    print("camera:", cam.name)
    arms = (NoArms() if args.no_arm else
            Arms(args.follower_port, args.leader_port, teleop=not (args.no_leader or args.snapshot),
                 max_step=args.max_step))

    try:
        if args.snapshot:
            joints = arms.step()
            real, _ = cam.get()
            simg, rm, tm = sim.render(p, joints)
            tiles = [compose(md, real, simg, rm, tm, 1.0, sim.obj_mask, sim.lines) for md in MODES[:4]]
            for t, md in zip(tiles, MODES[:4]):
                cv2.putText(t, md, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            cv2.imwrite(args.snapshot, grid)
            print("joints", {k: round(v, 1) for k, v in (joints or {}).items()})
            print("saved", args.snapshot)
            return

        win = ("live overlay - sim wrist_cam on real wrist cam" if args.cam == "wrist"
               else "live overlay - sim ext_cam on real side cam")
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        sim_win, sim_open = "sim ext_cam", False
        steps = {"pos": 0.005, "ang": 0.25, "joint": 0.5}
        sel, mode, msg, msg_t = 1, 0, "", 0.0
        t_last, fps = time.time(), 0.0
        last_n = -1
        history = []                            # Params before each change, for undo
        target = "camera"                       # what the movement keys drive; Tab toggles
        while True:
            joints = arms.step()
            real, n = cam.get()
            if n == last_n:                     # wait for a new camera frame
                time.sleep(0.003)
            last_n = n
            simg, rm, tm = sim.render(p, joints)
            view = compose(MODES[mode], real, simg, rm, tm, args.scale, sim.obj_mask, sim.lines)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-3)
            t_last = now
            if msg and now - msg_t > 4:
                msg = ""
            hud = panel(view.shape[0], p, sel, steps, MODES[mode], fps, joints, arms.teleop, msg,
                        target, brightness(real, simg, rm, tm), args.cam, p_orig, cont)
            cv2.imshow(win, np.hstack([view, hud]))
            if MODES[mode] == "separate":
                cv2.imshow(sim_win, compose("sim", real, simg, rm, tm, args.scale))
                sim_open = True
            elif sim_open:
                cv2.destroyWindow(sim_win)
                sim_open = False

            k = cv2.waitKey(1) & 0xFF
            if k == 255:
                continue
            if k in (ord("q"), 27):
                break
            if k == 9:                          # Tab: camera -> light -> container -> camera
                order = ["camera", "light"] + (["container"] if cont is not None else [])
                target = order[(order.index(target) + 1) % len(order)]
                msg, msg_t = f"keys now move the {target.upper()}", now
                continue
            c = chr(k)
            if target == "container" and c in "wsadzx":
                if c in "wsad":
                    cont.sel = c
                    msg, msg_t = f"moving the {cont.EDGES[c][0]}", now
                    continue
                history.append(("container", cont.edges()))
                cont.nudge(c == "x", steps["pos"] / 5)
                cont.apply()
                sim.close()
                sim = Sim(args.cam, args.cube, quiet=True)
                L_, W_ = cont.size()
                msg, msg_t = f"container {L_ * 1000:.1f} x {W_ * 1000:.1f} mm (saved)", now
                continue
            if target == "container" and c in "rftbijkluoyh":
                continue                        # camera / light movement keys do nothing here
            before = p.copy()
            changed, sel = apply_key(k, p, sel, steps, target, args.cam)
            if changed:
                history.append(before)
                p.save()
                continue
            if k in (8, 127):                   # Backspace: undo
                if history and isinstance(history[-1], tuple):          # a container resize
                    cont.set_edges(history.pop()[1])
                    cont.apply()
                    sim.close()
                    sim = Sim(args.cam, args.cube, quiet=True)
                    msg, msg_t = f"undo container ({len(history)} left)", now
                elif history:
                    p = history.pop()
                    p.save()
                    msg, msg_t = f"undo ({len(history)} left)", now
                else:
                    msg, msg_t = "nothing to undo", now
                continue
            if c == "J":                        # offsets only, back to the Nov 18 fit
                history.append(before)
                p.offs = dict(original_params().offs)
                p.save()
                msg, msg_t = "joint offsets reset to side_cam_fit.json", now
                continue
            if c in "0F":
                history.append(before)
                p = p0.copy() if c == "0" else original_params()
                p.save()
                msg, msg_t = "reset to session start" if c == "0" else "reset to original fit", now
                continue
            if c in "[]":
                f = 0.5 if c == "[" else 2.0
                steps = {kk: v * f for kk, v in steps.items()}
            elif c == "m":
                mode = (mode + 1) % len(MODES)
            elif c == "g":
                GRABS.mkdir(exist_ok=True)
                i = len(list(GRABS.glob("*.png")))
                cv2.imwrite(str(GRABS / f"{i:03d}.png"), real)
                jf = GRABS / "joints.json"
                allj = json.loads(jf.read_text()) if jf.exists() else {}
                allj[f"{i:03d}"] = {kk: round(v, 2) for kk, v in (joints or {}).items()}
                jf.write_text(json.dumps(allj, indent=2))
                msg, msg_t = f"grabbed live_grabs/{i:03d}.png", now
            elif c == "X":
                if args.cam == "wrist":
                    write_wrist_xml(p)
                    msg, msg_t = "wrist_cam written to so101_real.xml", now
                else:
                    write_xml(p)
                    msg, msg_t = "camera, table, light written to scene XML", now
    finally:
        arms.close()
        cam.close()
        cv2.destroyAllWindows()
        if STATE.exists():
            print("values saved in", STATE)


if __name__ == "__main__":
    main()
