"""Refit the wrist camera's POSE (position + orientation in the camera_mount frame) with
its lens fixed to the checkerboard calibration (calib/intrinsics/wrist.json).

The old fit (fit_wrist_cam.py) fitted the pose and the field of view together from
the jaw outline alone, which can't tell "further away + narrower" from "closer +
wider": it found 87.6 deg. The calibrated lens is 57.6 deg, so the pose has to be
refitted with the lens held fixed.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/fit_wrist_pose.py                 # fit + report
    /opt/anaconda3/envs/lerobot/bin/python calib/fit_wrist_pose.py --write         # also write so101_real.xml

Real jaws: segmented in calib/real/wrist_640x480.png (cream jaws on the grey
table, gripper closed). Sim jaws: segmentation render of wrist_cam (measured
intrinsics), distorted with the real lens (wrist_lens.py). Maximises the mask
overlap (IoU) from two starts: the current wrist_cam pose, and the lens position
the Seeed bracket's geometry predicts. Checks the result on a second, newer real
frame (calib/intrinsics/posefit/wrist_now.png).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
from wrist_lens import WristLens  # noqa: E402

SCENE = SIM / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
ROBOT = SIM / "mujoco_menagerie/robotstudio_so101/so101_real.xml"
REAL = HERE / "real/wrist_640x480.png"
CHECK = HERE / "intrinsics/posefit/wrist_now.png"
OUT = HERE / "intrinsics/posefit"
SEEED_LENS = (np.array([0.0025, 0.07251, -0.00096]), np.array([0.976296, -0.216440, 0, 0]))
W, H = 640, 480
S = 2                                   # fit at half resolution (320x240)


def real_jaw_mask(path):
    """Cream PLA jaws on the neutral grey table: warmth (R - B), blobs touching the bottom edge."""
    img = cv2.imread(str(path))
    warm = img[..., 2].astype(np.float32) - img[..., 0].astype(np.float32)
    warm = cv2.GaussianBlur(warm, (0, 0), 3)
    m = (warm > 5).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    keep = set(np.unique(lab[-1])) - {0}
    m = np.isin(lab, list(keep)).astype(np.uint8)
    ff = m.copy()                                         # fill holes
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
    m = m | (1 - ff)
    return m.astype(bool), img


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotvec_quat(r):
    a = np.linalg.norm(r)
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * r / a]


class JawSim:
    def __init__(self, lens):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[self.m.jnt_qposadr[self.m.joint("gripper").id]] = -0.174533      # closed
        mujoco.mj_forward(self.m, self.d)
        self.cam = self.m.camera("wrist_cam").id
        jaw = {self.m.body("gripper").id, self.m.body("moving_jaw_so101_v1").id}
        self.jaw = np.flatnonzero(np.isin(self.m.geom_bodyid, list(jaw)))
        self.r = mujoco.Renderer(self.m, H // S, W // S)
        self.r.enable_segmentation_rendering()
        self.rgb = mujoco.Renderer(self.m, H, W)
        self.lens = lens

    def set(self, pos, quat):
        self.m.cam_pos[self.cam] = pos
        self.m.cam_quat[self.cam] = quat / np.linalg.norm(quat)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)            # camera world poses are computed here, not in kinematics

    def mask(self, pos, quat):
        self.set(pos, quat)
        self.r.update_scene(self.d, camera=self.cam)
        seg = self.r.render()[..., 0]
        m = np.isin(seg, self.jaw).astype(np.uint8)
        return self.lens.distort(m, nearest=True).astype(bool)

    def image(self, pos, quat):
        self.set(pos, quat)
        self.rgb.update_scene(self.d, camera=self.cam)
        return self.lens.distort(self.rgb.render()[..., ::-1].copy())


def nelder_mead(f, x0, step, iters=600, tol=1e-6):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the fitted pose into so101_real.xml")
    args = ap.parse_args()

    lens = WristLens()
    real, real_img = real_jaw_mask(REAL)
    real_s = cv2.resize(real.astype(np.uint8), (W // S, H // S), interpolation=cv2.INTER_NEAREST).astype(bool)
    sim = JawSim(lens)
    pos_now, quat_now = sim.m.cam_pos[sim.cam].copy(), sim.m.cam_quat[sim.cam].copy()

    def iou(pos, quat, full=False):
        s = sim.mask(pos, quat)
        r = real if full else real_s
        if full:
            s = cv2.resize(s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return (s & r).sum() / max((s | r).sum(), 1)

    def unpack(p, base):
        return base[0] + p[:3], quat_mul(base[1], rotvec_quat(p[3:]))

    results = []
    for name, base in (("current wrist_cam", (pos_now, quat_now)), ("Seeed bracket lens", SEEED_LENS)):
        print(f"start {name}: IoU {iou(*base):.3f}")
        best = None
        for rounds in range(3):
            x0 = np.zeros(6) if best is None else best[0]
            step = np.r_[[0.006] * 3, [0.06] * 3] * (0.5 ** rounds)
            x, v = nelder_mead(lambda p: 1 - iou(*unpack(p, base)), x0, step, iters=400)
            best = (x, v)
        pos, quat = unpack(best[0], base)
        results.append((1 - best[1], name, pos, quat))
        print(f"   -> IoU {1 - best[1]:.3f}  pos {np.round(pos * 1000, 1)} mm")
    score, name, pos, quat = max(results, key=lambda r: r[0])
    full = iou(pos, quat, full=True)
    d_seeed = np.linalg.norm(pos - SEEED_LENS[0]) * 1000
    print(f"\nbest (from {name}): IoU {full:.3f} at 640x480; position {np.round(pos * 1000, 2)} mm, "
          f"{d_seeed:.1f} mm from the Seeed bracket's lens estimate")

    # overlays: fit frame and the newer check frame (jaw outline drawn)
    OUT.mkdir(parents=True, exist_ok=True)
    s_full = cv2.resize(sim.mask(pos, quat).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    tiles = []
    for img in (real_img, cv2.imread(str(CHECK)) if CHECK.exists() else None):
        if img is None:
            continue
        o = img.copy()
        cs, _ = cv2.findContours(s_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (0, 255, 0), 2)
        tiles.append(o)
    sim_img = sim.image(pos, quat)
    tiles.append(sim_img)
    cv2.imwrite(str(OUT / "pose_fit_overlay.png"), np.hstack(tiles))
    res = dict(pos=[round(float(v), 5) for v in pos], quat=[round(float(v), 6) for v in quat / np.linalg.norm(quat)],
               iou=round(float(full), 4), start=name, mm_from_seeed_lens=round(float(d_seeed), 1),
               lens=str(lens.calib.get("date")), real_frame=str(REAL.name))
    (OUT / "pose_fit.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / "pose_fit.json", "and pose_fit_overlay.png")

    if args.write:
        s = ROBOT.read_text()
        el = re.search(r'<camera name="wrist_cam"[^>]*/>', s, re.S).group(0)
        new = re.sub(r'pos="[^"]*"', 'pos="' + " ".join(f"{v:.5f}" for v in pos) + '"', el)
        new = re.sub(r'quat="[^"]*"', 'quat="' + " ".join(f"{v:.6f}" for v in res["quat"]) + '"', new)
        bak = ROBOT.with_suffix(".xml.before_wrist_pose")
        if not bak.exists():
            shutil.copy(ROBOT, bak)
        ROBOT.write_text(s.replace(el, new))
        mujoco.MjModel.from_xml_path(str(SCENE))
        print("wrote wrist_cam pose into", ROBOT.name, "(backup", bak.name + ")")


if __name__ == "__main__":
    main()
