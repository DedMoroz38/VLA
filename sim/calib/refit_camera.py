"""Refit ONLY the sim side camera (ext_cam) to real frames, with the joint offsets
held fixed at the touch-test-verified values (calib/joint_map.json) and the table
where the scene puts it.

Why: the camera and the joint offsets used to be tuned together, so camera error
could hide joint error and vice versa. The offsets are now checked with a ruler
(touch_test.py), so the camera is fitted to them, not the other way round.

Input: a folder from capture_poses.py (rest.png, P1..P8.png, joints.json with the
measured joints in the CURRENT calibration's LeRobot degrees).

    cd sim
    /opt/anaconda3/envs/mujoco-so101/bin/python calib/refit_camera.py calib/poses_20261119

Fits camera position (3) + orientation (3), fovy fixed, maximising the arm
silhouette overlap (IoU) over all frames, plus the table's back edge found
automatically in the median background. Writes <folder>/camera_refit.json and
<folder>/refit_overlay.png (green = both, red = sim only, blue = real only).
Applying it is a separate step (see the printout).
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import joint_units  # noqa: E402
from fit_side_cam import real_masks  # noqa: E402

SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
W, H = 320, 240            # half resolution, same FOV as 640x480
EDGE_WEIGHT = 0.004        # cost per px of table-edge rms error (as in fit_side_cam.py)


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[self.m.jnt_qposadr[j]] = 5.0
        self.cam = self.m.camera("ext_cam").id
        self.r = mujoco.Renderer(self.m, H, W)
        self.r.enable_segmentation_rendering()
        body = self.m.geom_bodyid
        self.arm_geoms = np.flatnonzero((body >= self.m.body("upper_arm").id) & (body <= self.m.body("moving_jaw_so101_v1").id))
        self.base_geoms = np.flatnonzero((body >= self.m.body("base").id) & (body <= self.m.body("shoulder").id))
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in joint_units.JOINTS]
        g = self.m.geom("table_top").id
        mujoco.mj_forward(self.m, self.d)
        self.table_x = float(self.d.geom_xpos[g][0] - self.m.geom_size[g][0])     # back edge, as the scene has it
        self.c0 = (self.m.cam_pos[self.cam].copy(), self.m.cam_quat[self.cam].copy())

    def set_cam(self, p):
        pos0, quat0 = self.c0
        self.m.cam_pos[self.cam] = pos0 + p[:3]
        q = (R.from_rotvec(p[3:6]) * R.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]])).as_quat()
        self.m.cam_quat[self.cam] = [q[3], q[0], q[1], q[2]]

    def cam_matrix(self):
        xm = np.zeros(9)
        mujoco.mju_quat2Mat(xm, self.m.cam_quat[self.cam])
        return xm.reshape(3, 3)

    def project(self, pts):
        """World points -> pixel (u, v) at W x H."""
        xm, pos = self.cam_matrix(), self.m.cam_pos[self.cam]
        f = (H / 2) / np.tan(np.radians(self.m.cam_fovy[self.cam]) / 2)
        c = (np.asarray(pts) - pos) @ xm
        return np.c_[W / 2 + f * c[:, 0] / -c[:, 2], H / 2 - f * c[:, 1] / -c[:, 2]]

    def edge_rows(self, p, u):
        self.set_cam(p)
        ys = np.linspace(-0.6, 0.6, 241)
        uv = self.project(np.c_[np.full_like(ys, self.table_x), ys, np.zeros_like(ys)])
        return np.interp(u, uv[:, 0], uv[:, 1])

    def masks(self, p, q_sims):
        self.set_cam(p)
        out, ign = [], []
        for q in q_sims:
            for a, v in zip(self.qadr, q):
                self.d.qpos[a] = v
            mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera=self.cam)
            seg = self.r.render()[..., 0]
            out.append(np.isin(seg, self.arm_geoms))
            ign.append(np.isin(seg, self.base_geoms))
        return np.stack(out), np.stack(ign)


def find_table_edge(bg, sim, real_any):
    """Columns + rows of the table's back edge in the median background: the strongest
    vertical brightness step near where the current sim camera puts the edge, away from
    the arm, then a robust straight-line fit."""
    g = ndimage.gaussian_filter(bg.mean(-1), 1.0)
    grad = np.abs(np.diff(g, axis=0))
    cols = np.arange(4, W - 4, 2)
    guess = sim.edge_rows(np.zeros(6), cols)
    arm_cols = real_any.any(0)
    us, vs = [], []
    for u, v0 in zip(cols, guess):
        if arm_cols[max(0, u - 3):u + 4].any():
            continue
        lo, hi = int(max(1, v0 - 25)), int(min(H - 2, v0 + 25))
        if hi <= lo:
            continue
        us.append(u)
        vs.append(lo + int(np.argmax(grad[lo:hi, u])) + 0.5)
    us, vs = np.array(us, float), np.array(vs, float)
    best = None
    rng = np.random.default_rng(0)
    for _ in range(300):                                     # RANSAC line
        i, j = rng.choice(len(us), 2, replace=False)
        if us[i] == us[j]:
            continue
        a = (vs[j] - vs[i]) / (us[j] - us[i])
        inl = np.abs(vs - (vs[i] + a * (us - us[i]))) < 1.5
        if best is None or inl.sum() > best.sum():
            best = inl
    return us[best], vs[best], len(us)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    args = ap.parse_args()
    folder = Path(args.folder)

    jm = joint_units.JointMap.load()
    joints = json.loads((folder / "joints.json").read_text())
    names = list(joints)
    q_sims = [jm.real_dict_to_sim({**{"gripper": 10.0}, **joints[n]}) for n in names]
    real, ims, bg = real_masks(names, folder)
    sim = Sim()
    eu, ev, n_cols = find_table_edge(bg, sim, real)
    print(f"{len(names)} frames; table edge: {len(eu)} of {n_cols} columns on one line")
    print(f"joint offsets held fixed: {jm.meta['offsets_deg']}  (joint_map {jm.meta['updated']})")
    print(f"table back edge held at x = {sim.table_x*1000:.1f} mm, fovy {sim.m.cam_fovy[sim.cam]:.2f}")

    def ious(p):
        s, ign = sim.masks(p, q_sims)
        keep = ~ndimage.binary_dilation(ign, iterations=2, structure=np.ones((1, 3, 3)))
        return (s & real & keep).sum((1, 2)) / np.maximum(((s | real) & keep).sum((1, 2)), 1), s

    def edge_rms(p):
        return float(np.sqrt(np.mean((sim.edge_rows(p, eu) - ev) ** 2)))

    def cost(p):
        return 1 - ious(p)[0].mean() + EDGE_WEIGHT * edge_rms(p)

    p = np.zeros(6)
    i0, e0 = ious(p)[0], edge_rms(p)
    print(f"current camera: mean IoU {i0.mean():.3f}, table edge {e0:.2f} px rms (at {W}x{H})")
    scales = np.r_[[0.02] * 3, [0.03] * 3]
    for it in range(3):
        simplex = np.vstack([p] + [p + np.eye(6)[i] * scales[i] * 0.5 ** it for i in range(6)])
        res = minimize(cost, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxiter=1500, xatol=1e-6, fatol=1e-6, adaptive=True))
        p = res.x
        print(f"  round {it + 1}: mean IoU {ious(p)[0].mean():.3f}, table edge {edge_rms(p):.2f} px")
    i1, s = ious(p)
    sim.set_cam(p)
    xm = sim.cam_matrix()
    pitch = float(np.degrees(np.arcsin(xm[2, 2])))       # view dir is -z, so pitch-down = asin(z_z)
    out = dict(folder=str(folder), joint_map_updated=jm.meta["updated"], offsets_deg=jm.meta["offsets_deg"],
               pos=[round(float(v), 5) for v in sim.m.cam_pos[sim.cam]],
               xyaxes=[round(float(v), 5) for v in np.r_[xm[:, 0], xm[:, 1]]],
               fovy=round(float(sim.m.cam_fovy[sim.cam]), 3), pitch_down_deg=round(pitch, 2),
               moved_mm=[round(float(v) * 1000, 1) for v in p[:3]],
               rotated_deg=round(float(np.degrees(np.linalg.norm(p[3:6]))), 2),
               iou_before=dict(zip(names, np.round(i0, 3).tolist())), iou_after=dict(zip(names, np.round(i1, 3).tolist())),
               mean_iou_before=round(float(i0.mean()), 3), mean_iou_after=round(float(i1.mean()), 3),
               edge_rms_px_before=round(e0, 2), edge_rms_px_after=round(edge_rms(p), 2))
    (folder / "camera_refit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("pos", "pitch_down_deg", "moved_mm", "rotated_deg",
                                          "mean_iou_before", "mean_iou_after", "edge_rms_px_before",
                                          "edge_rms_px_after")}, indent=2))

    tiles = []
    for k in range(len(names)):
        o = ims[k].copy()
        o[s[k] & ~real[k]] = [255, 0, 0]
        o[real[k] & ~s[k]] = [0, 0, 255]
        both = s[k] & real[k]
        o[both] = (o[both] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        uv = np.c_[eu, sim.edge_rows(p, eu)].astype(int)
        o[np.clip(uv[:, 1], 0, H - 1), np.clip(uv[:, 0], 0, W - 1)] = [255, 255, 0]
        tiles.append(o)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])).save(folder / "refit_overlay.png")
    print("wrote", folder / "camera_refit.json", "and refit_overlay.png")


if __name__ == "__main__":
    main()
