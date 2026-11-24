"""Why does the real cube appear shifted in the wrist view? Three candidate causes:

  mount      the sim wrist camera sits/points differently on the gripper than the real one
             -> the JAWS are shifted in the image too (they are rigid to the camera)
  arm        the real gripper is somewhere slightly different for the same joint readings
             -> jaws not shifted; the cube error is constant in the GRIPPER's frame
  placement  the real cube sits slightly off its sim pose (hand placement, or a bias of the side
             camera that drew the outline) -> jaws not shifted; error constant in the WORLD frame

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/wrist_offset_cause.py calib/pairs/20261123_202747

Per usable pair (cube visible in the wrist view, and the side camera says the cube is on its
outline): (1) the jaw shift: the real jaws (cream on grey, as in fit_wrist_pose.py) against the
sim jaws at the measured gripper angle; (2) the world-frame (x, y) cube shift that makes the sim
cube outline land on the real wrist edges. Then: is that shift steadier in world or gripper axes?
"""
import argparse
import json
import pathlib
import sys

import cv2
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE))

import joint_units  # noqa: E402
import measure_gap as mg  # noqa: E402
from pick_place_task import reset_to_ready, set_eraser_pose  # noqa: E402
from run_policy_real import H, W, SimView  # noqa: E402
from wrist_lens import orient_wrist  # noqa: E402

SIDE_OK_PX = 5


def unrotate(img):
    return np.ascontiguousarray(np.rot90(img, -1))          # undo orient_wrist


def real_jaw_mask(img_rot):
    """Cream PLA jaws on the grey table: warmth (R - B), blobs touching the (unrotated) bottom edge."""
    img = unrotate(img_rot)
    warm = cv2.GaussianBlur(img[..., 2].astype(np.float32) - img[..., 0].astype(np.float32), (0, 0), 3)
    n, lab = cv2.connectedComponents((warm > 5).astype(np.uint8))
    keep = set(np.unique(lab[-1])) - {0}
    return np.isin(lab, list(keep))


class Scene:
    def __init__(self, sim):
        self.s = sim
        self.r = mujoco.Renderer(sim.m, H, W)
        self.r.enable_segmentation_rendering()
        sim._render(sim.d, "ext_cam")                           # creates sim.lens
        m = sim.m
        self.cube = np.flatnonzero(m.geom_bodyid == m.body(sim.obj).id)
        self.jaw = np.flatnonzero(np.isin(m.geom_bodyid, [m.body("gripper").id, m.body("moving_jaw_so101_v1").id]))
        self.grip_site = m.site("gripperframe").id

    def pose(self, q_sim, x, y, yaw):
        s = self.s
        reset_to_ready(s.m, s.d)
        s.d.qpos[:6] = q_sim
        set_eraser_pose(s.m, s.d, x, y, yaw, object_body=s.obj)
        mujoco.mj_forward(s.m, s.d)

    def seg(self):
        """Wrist segmentation, distorted, NOT rotated (the jaw test works unrotated)."""
        self.r.update_scene(self.s.d, camera="wrist_cam")
        return self.s.lens.distort(self.r.render()[..., 0].copy(), nearest=True)

    def gripper_xy_axes(self):
        """World xy unit vectors of the gripper's forward direction and its left, on the table."""
        R = self.s.d.site_xmat[self.grip_site].reshape(3, 3)
        f = R[:, 2][:2]                                        # gripperframe z: along the jaws
        if np.linalg.norm(f) < 0.2:                            # pointing straight down: use x
            f = R[:, 0][:2]
        f = f / np.linalg.norm(f)
        return f, np.array([-f[1], f[0]])


def cube_cost(sc, q_sim, pose, dist, dx, dy):
    sc.pose(q_sim, pose[0] + dx, pose[1] + dy, pose[2])
    m = np.isin(orient_wrist(sc.seg()), sc.cube)
    pts = np.argwhere(mg.outline(m))
    if len(pts) < 40:
        return np.inf
    return float(np.minimum(dist[pts[:, 0], pts[:, 1]], 12.0).mean())


def fit_cube_shift(sc, q_sim, pose, dist):
    best = (0.0, 0.0, cube_cost(sc, q_sim, pose, dist, 0, 0))
    base = best[2]
    for span, step in ((0.012, 0.002), (0.002, 0.0005)):
        cx, cy = best[:2]
        for dy in np.arange(cy - span, cy + span + 1e-9, step):
            for dx in np.arange(cx - span, cx + span + 1e-9, step):
                c = cube_cost(sc, q_sim, pose, dist, dx, dy)
                if c < best[2]:
                    best = (float(dx), float(dy), c)
    return best, base


def jaw_shift(sc, real_rot, q_sim, pose):
    """Pixel shift (unrotated image) and IoU of the real jaw mask vs the sim one."""
    real = real_jaw_mask(real_rot)
    sc.pose(q_sim, *pose)
    sim = np.isin(sc.seg(), sc.jaw)
    if real.sum() < 2000 or sim.sum() < 2000:
        return None
    best = None
    for dy in range(-30, 31, 2):
        for dx in range(-30, 31, 2):
            s = mg.shift(sim, dx, dy)
            iou = (s & real).sum() / max((s | real).sum(), 1)
            if best is None or iou > best[2]:
                best = (dx, dy, iou)
    iou0 = (sim & real).sum() / max((sim | real).sum(), 1)
    return {"dx": best[0], "dy": best[1], "iou_best": round(float(best[2]), 3), "iou_now": round(float(iou0), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    a = ap.parse_args()
    folder = SIM / a.pairs
    idx = json.loads((folder / "pairs.json").read_text())
    gap = {r["n"]: r for r in json.loads((folder / "gap.json").read_text())["pairs"]}
    sim = SimView(SIM / idx["scene"])
    sc = Scene(sim)
    jmap = joint_units.JointMap.load()
    rows = []
    for e in idx["pairs"]:
        g = gap[e["n"]]
        side, wc = g["side"]["geometry"]["cube"], g["wrist"]["geometry"]["cube"]
        if not side or abs(side["dx"]) > SIDE_OK_PX or abs(side["dy"]) > SIDE_OK_PX or not wc or wc["resid_px"] > 5:
            continue
        q_sim = jmap.real_to_sim(np.array(e["joints_real"], np.float32))
        pose = tuple(e["pose"])
        real = cv2.imread(str(folder / e["files"]["real_wrist"]))
        dist = mg.real_edges(real)
        (dx, dy, c), c0 = fit_cube_shift(sc, q_sim, pose, dist)
        sc.pose(q_sim, *pose)
        f, l = sc.gripper_xy_axes()
        j = jaw_shift(sc, real, q_sim, pose)
        rows.append({"n": e["n"], "pose_id": e["pose_id"], "world_mm": [dx * 1000, dy * 1000],
                     "gripper_mm": [float(np.dot([dx, dy], f) * 1000), float(np.dot([dx, dy], l) * 1000)],
                     "cost_before": round(c0, 2), "cost_after": round(c, 2), "jaw": j})
        print(f"pair {e['n']:2d} pose {e['pose_id']}: cube shift world ({dx * 1000:+5.1f}, {dy * 1000:+5.1f}) mm, "
              f"gripper fwd/left ({rows[-1]['gripper_mm'][0]:+5.1f}, {rows[-1]['gripper_mm'][1]:+5.1f}) mm, "
              f"edge cost {c0:.1f} -> {c:.1f} px; jaw {j}", flush=True)
    W_ = np.array([r["world_mm"] for r in rows])
    G_ = np.array([r["gripper_mm"] for r in rows])
    jaws = [r["jaw"] for r in rows if r["jaw"]]
    summary = {
        "pairs_used": len(rows),
        "world_mm_median": np.median(W_, 0).round(1).tolist(), "world_mm_spread": W_.std(0).round(1).tolist(),
        "gripper_mm_median": np.median(G_, 0).round(1).tolist(), "gripper_mm_spread": G_.std(0).round(1).tolist(),
        "jaw_shift_px_median": [float(np.median([j["dx"] for j in jaws])), float(np.median([j["dy"] for j in jaws]))]
        if jaws else None,
        "jaw_iou_now_median": float(np.median([j["iou_now"] for j in jaws])) if jaws else None,
        "jaw_iou_best_median": float(np.median([j["iou_best"] for j in jaws])) if jaws else None,
    }
    by_pose = {}
    for r in rows:
        by_pose.setdefault(r["pose_id"], []).append(r["world_mm"])
    summary["world_mm_median_by_pose"] = {p: np.median(v, 0).round(1).tolist() for p, v in by_pose.items()}
    (folder / "wrist_offset_cause.json").write_text(json.dumps({"summary": summary, "pairs": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
