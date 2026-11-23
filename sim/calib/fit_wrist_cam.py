"""Fit the sim wrist camera to the real one from the gripper-jaw silhouette.

The jaws are rigidly attached to the wrist camera, so their outline in the
image is the same in every arm pose -- only the camera's mount pose and lens
decide it. This script segments the jaws in a real wrist frame, renders a
segmentation mask of the sim jaws, and optimises the camera's pose (in the
camera_mount body frame) and fovy to maximise mask overlap (IoU).

    python calib/fit_wrist_cam.py calib/real/wrist_pair_1080.png

The real frame may be 1920x1080: it is centre-cropped to 4:3 and resized to
640x480, which is exactly what the icspring camera's own 640x480 mode returns.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene.xml"
W, H = 640, 480


def to_640x480(img):
    w, h = img.size
    if (w, h) != (W, H):
        cw = h * 4 // 3                      # centre 4:3 crop, as the camera does
        img = img.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h)).resize((W, H), Image.LANCZOS)
    return img


def real_jaw_mask(path):
    """Jaws are cream PLA, the table is neutral grey: threshold on warmth (R-B)."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(float)
    warm = Image.fromarray(np.clip((a[..., 0] - a[..., 2]) * 8, 0, 255).astype(np.uint8))
    warm = warm.filter(ImageFilter.GaussianBlur(radius=img.size[0] / 200))
    mask = to_640x480(warm)
    m = np.asarray(mask) > 40              # R-B > 5
    # keep blobs connected to the bottom edge (the jaws enter from the bottom)
    from scipy import ndimage
    lab, n = ndimage.label(m)
    keep = set(lab[-1][lab[-1] > 0])
    m = np.isin(lab, list(keep))
    return ndimage.binary_fill_holes(m), np.asarray(to_640x480(img))


class SimWrist:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.cam = self.m.camera("wrist_cam").id
        self.r = mujoco.Renderer(self.m, H, W)
        jaw_bodies = {self.m.body("gripper").id, self.m.body("moving_jaw_so101_v1").id,
                      self.m.body("camera_mount").id}
        self.jaw_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in jaw_bodies])

    def set(self, p, gripper):
        pos, rv, fovy = p[:3], p[3:6], p[6]
        q = R.from_rotvec(rv).as_quat()        # x y z w
        self.m.cam_pos[self.cam] = pos
        self.m.cam_quat[self.cam] = [q[3], q[0], q[1], q[2]]
        self.m.cam_fovy[self.cam] = fovy
        self.d.qpos[:] = 0
        self.d.qpos[self.m.joint("gripper").qposadr[0]] = gripper
        mujoco.mj_forward(self.m, self.d)

    def mask(self, p, gripper):
        self.set(p, gripper)
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=self.cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], self.jaw_geoms) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)

    def rgb(self, p, gripper):
        self.set(p, gripper)
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render()


def lookat_rotvec(eye, target, up_hint):
    """Rotation (as rotvec) of a MuJoCo camera at eye looking at target (-z forward, y up)."""
    f = target - eye; f /= np.linalg.norm(f)
    x = np.cross(f, up_hint); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return R.from_matrix(np.column_stack([x, y, -f])).as_rotvec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real", nargs="?", default=str(HERE / "real/wrist_pair_1080.png"))
    ap.add_argument("--gripper", type=float, default=-0.17, help="sim gripper angle of the real frame (closed)")
    args = ap.parse_args()

    real, real_rgb = real_jaw_mask(args.real)
    Image.fromarray((real * 255).astype(np.uint8)).save(HERE / "wrist_real_mask.png")
    sim = SimWrist()

    def cost(p):
        s = sim.mask(p, args.gripper)
        inter = (s & real).sum(); union = (s | real).sum()
        return 1 - inter / max(union, 1)

    # Initial guesses: camera at the official lens position, looking at points
    # around the jaw tips, with a sweep of roll (the real camera may be mounted
    # rotated relative to the official design).
    lens = np.array([-0.00298, 0.05669, -0.04954])
    tip = np.array([0.012, 0.0, -0.098])
    best = (np.inf, None)
    for dz in [0.0, -0.03, -0.06]:
        for dy in [0.0, 0.02, -0.02]:
            target = tip + np.array([0, dy, dz])
            f = target - lens
            for roll in np.radians(np.arange(0, 360, 30)):
                up = R.from_rotvec(f / np.linalg.norm(f) * roll).apply(np.cross(f, [1, 0, 0]))
                for fovy in [60, 75, 90]:
                    p = np.r_[lens, lookat_rotvec(lens, target, up), fovy]
                    c = cost(p)
                    if c < best[0]:
                        best = (c, p)
    print(f"coarse best IoU {1 - best[0]:.3f}")

    p = best[1]
    for _ in range(2):                  # one restart: Nelder-Mead often stalls early
        res = minimize(cost, p, method="Nelder-Mead",
                       options=dict(initial_simplex=None, maxiter=1500, xatol=1e-5, fatol=1e-5,
                                    adaptive=True))
        p = res.x
        print(f"refined IoU {1 - res.fun:.3f}")

    q = R.from_rotvec(p[3:6]).as_quat()
    out = dict(pos=[round(v, 5) for v in p[:3]],
               quat=[round(v, 6) for v in [q[3], q[0], q[1], q[2]]],
               fovy=round(float(p[6]), 2), iou=round(1 - cost(p), 4), gripper=args.gripper)
    print(json.dumps(out))
    (HERE / "wrist_cam_fit.json").write_text(json.dumps(out, indent=2))

    s = sim.mask(p, args.gripper)
    over = real_rgb.copy()
    over[s & ~real] = [255, 0, 0]          # sim only
    over[real & ~s] = [0, 0, 255]          # real only
    over[s & real] = (over[s & real] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(over).save(HERE / "wrist_fit_overlay.png")
    Image.fromarray(sim.rgb(p, args.gripper)).save(HERE / "wrist_fit_sim.png")


if __name__ == "__main__":
    main()
