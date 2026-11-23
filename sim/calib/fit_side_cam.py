"""Fit the sim side camera (ext_cam) to the real one from several known arm poses.

Input: calib/poses/<name>.png side-cam frames and calib/poses/joints.json with
the joint angles measured at each (from capture_poses.py).

1. Real robot masks: the arm is in a different place in every frame, so the
   per-pixel median of all frames is the empty scene (except the base, which
   never moves). Robot = pixels that differ from that background; soft
   shadows (uniformly darker, same hue) are rejected.
2. Sim masks: segmentation renders of the robot with the measured joint angles.
3. The table's back edge, measured in the background image
   (poses/table_edge.json): a long straight line, which pins down camera roll
   and pitch better than the arm does. Its x position in the sim is fitted too.
4. Optimise camera pose (6) + per-joint offsets between LeRobot degrees and
   MuJoCo radians (5) + table edge x (1) to maximise the mean IoU over all
   frames, minus a small penalty per pixel of table-edge error. The base and
   shoulder, which sit in the median background, are left out of the IoU.

    python calib/fit_side_cam.py            # writes side_cam_fit.json + overlays
    python calib/fit_side_cam.py --free-fovy
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
W, H = 320, 240           # fit at half resolution; same FOV as 640x480
EDGE_WEIGHT = 0.004       # cost per pixel of table-edge rms error (2 px ~ 0.008 IoU)


def real_masks(names, folder=HERE / "poses"):
    ims = np.stack([np.asarray(Image.open(Path(folder) / f"{n}.png").convert("RGB").resize((W, H), Image.BILINEAR))
                    .astype(float) for n in names])
    bg = np.median(ims, axis=0)
    masks = []
    for im in ims:
        diff = np.abs(im - bg).sum(-1)
        ratio = (im.sum(-1) + 1) / (bg.sum(-1) + 1)
        chroma = np.abs((im - im.mean(-1, keepdims=True)) - (bg - bg.mean(-1, keepdims=True))).sum(-1)
        shadow = (ratio > 0.6) & (ratio < 0.97) & (chroma < 18)
        m = (diff > 38) & ~shadow
        m = ndimage.binary_opening(m, iterations=1)
        m = ndimage.binary_closing(m, iterations=2)
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)                       # keep the arm: the largest blob
        if n:                                           # and anything right next to it
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            main = lab == 1 + np.argmax(sizes)
            near = ndimage.binary_dilation(main, iterations=12)
            keep = [i for i in range(1, n + 1) if sizes[i - 1] >= 30 and (near & (lab == i)).any()]
            m = np.isin(lab, keep)
        masks.append(m)
    return np.stack(masks), ims.astype(np.uint8), bg


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, 640)
        self.d = mujoco.MjData(self.m)
        self.cam = self.m.camera("ext_cam").id
        self.r = mujoco.Renderer(self.m, H, W)
        body = self.m.geom_bodyid
        self.arm_geoms = np.flatnonzero((body >= self.m.body("upper_arm").id) & (body <= self.m.body("moving_jaw_so101_v1").id))
        self.base_geoms = np.flatnonzero((body >= self.m.body("base").id) & (body <= self.m.body("shoulder").id))
        self.r.enable_segmentation_rendering()
        self.c0 = (self.m.cam_pos[self.cam].copy(), self.m.cam_quat[self.cam].copy(), float(self.m.cam_fovy[self.cam]))

    def set_cam(self, p):
        pos0, quat0, fovy0 = self.c0
        self.m.cam_pos[self.cam] = pos0 + p[:3]
        q0 = R.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]])
        q = (R.from_rotvec(p[3:6]) * q0).as_quat()     # world-frame perturbation
        self.m.cam_quat[self.cam] = [q[3], q[0], q[1], q[2]]
        self.m.cam_fovy[self.cam] = fovy0 + (p[12] if len(p) > 12 else 0.0)

    def edge_rows(self, p, u):
        """Image rows (640x480 frame) where the table back edge at x=p[11] projects, at columns u."""
        self.set_cam(p)
        pos = self.m.cam_pos[self.cam]
        xm = np.zeros(9)
        mujoco.mju_quat2Mat(xm, self.m.cam_quat[self.cam])
        xm = xm.reshape(3, 3)
        f = 240 / np.tan(np.radians(self.m.cam_fovy[self.cam]) / 2)
        ys = np.linspace(-0.6, 0.6, 121)
        c = (np.c_[np.full_like(ys, p[11]), ys, np.zeros_like(ys)] - pos) @ xm   # camera coords
        uu = 320 + f * c[:, 0] / -c[:, 2]
        vv = 240 - f * c[:, 1] / -c[:, 2]
        return np.interp(u, uu, vv)

    def masks(self, p, joints):
        self.set_cam(p)
        out, ign = [], []
        for q in joints:
            self.d.qpos[:] = 0
            self.d.qpos[6] = 5                           # eraser out of the way
            self.d.qpos[:5] = np.radians(q + p[6:11])
            self.d.qpos[5] = -0.17
            mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera=self.cam)
            seg = self.r.render()[..., 0]
            out.append(np.isin(seg, self.arm_geoms))
            ign.append(np.isin(seg, self.base_geoms))
        return np.stack(out), np.stack(ign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-fovy", action="store_true", help="also fit the vertical FOV")
    args = ap.parse_args()

    joints_all = json.loads((HERE / "poses/joints.json").read_text())
    names = list(joints_all)
    joints = np.array([[joints_all[n][j] for j in ARM] for n in names])
    real, ims, bg = real_masks(names)
    sim = Sim()
    edge = json.loads((HERE / "poses/table_edge.json").read_text())
    edge_u, edge_v = np.array(edge["cols"], float), np.array(edge["ys"], float)

    def edge_rms(p):
        return float(np.sqrt(np.mean((sim.edge_rows(p, edge_u) - edge_v) ** 2)))

    def iou(p):
        s, ign = sim.masks(p, joints)
        keep = ~ndimage.binary_dilation(ign, iterations=2, structure=np.ones((1, 3, 3)))
        inter = (s & real & keep).sum((1, 2)); union = ((s | real) & keep).sum((1, 2))
        return np.mean(inter / np.maximum(union, 1))

    def cost(p):
        return 1 - iou(p) + EDGE_WEIGHT * edge_rms(p)

    n = 13 if args.free_fovy else 12
    p = np.zeros(n)
    p[11] = -0.06                                         # table edge x, initial guess
    print(f"start: mean IoU {iou(p):.3f}, edge rms {edge_rms(p):.1f}px")
    scales = np.r_[[0.03] * 3, [0.05] * 3, [5.0] * 5, [0.02], [3.0] * (n - 12)]
    for it in range(3):
        simplex = np.vstack([p] + [p + np.eye(n)[i] * scales[i] * (0.5 ** it) for i in range(n)])
        res = minimize(cost, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxiter=2500, xatol=1e-5, fatol=1e-5, adaptive=True))
        p = res.x
        print(f"round {it + 1}: mean IoU {iou(p):.3f}, edge rms {edge_rms(p):.2f}px")

    sim.set_cam(p)
    cam = sim.cam
    s, ign = sim.masks(p, joints)
    keep = ~ndimage.binary_dilation(ign, iterations=2, structure=np.ones((1, 3, 3)))
    ious = ((s & real & keep).sum((1, 2)) / np.maximum(((s | real) & keep).sum((1, 2)), 1))
    xmat = np.zeros(9)
    mujoco.mju_quat2Mat(xmat, sim.m.cam_quat[cam])
    xmat = xmat.reshape(3, 3)
    pitch = np.degrees(np.arcsin(-(-xmat[:, 2])[2]))   # camera looks along -z
    out = dict(pos=[round(float(v), 4) for v in sim.m.cam_pos[cam]],
               xyaxes=[round(float(v), 4) for v in np.r_[xmat[:, 0], xmat[:, 1]]],
               fovy=round(float(sim.m.cam_fovy[cam]), 2), pitch_down_deg=round(float(pitch), 1),
               joint_offsets_deg=dict(zip(ARM, [round(float(v), 2) for v in p[6:11]])),
               table_back_edge_x=round(float(p[11]), 4), table_edge_rms_px=round(edge_rms(p), 2),
               iou_per_frame=dict(zip(names, [round(float(v), 3) for v in ious])),
               mean_iou=round(float(ious.mean()), 3))
    print(json.dumps(out, indent=2))
    (HERE / "side_cam_fit.json").write_text(json.dumps(out, indent=2))

    tiles = []
    for k in range(len(names)):
        o = ims[k].copy()
        o[s[k] & ~real[k]] = [255, 0, 0]
        o[real[k] & ~s[k]] = [0, 0, 255]
        both = s[k] & real[k]
        o[both] = (o[both] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        tiles.append(o)
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])).save(HERE / "side_fit_overlay.png")


if __name__ == "__main__":
    main()
