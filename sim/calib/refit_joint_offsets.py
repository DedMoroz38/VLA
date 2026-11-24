"""Fit small constant joint offsets (real -> sim) that explain the wrist-view cube error, while
keeping the arm aligned in the side view. Alternative to refit_wrist_cam.py (mount pose).

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/refit_joint_offsets.py calib/pairs/20261123_202747

Model: q_sim = joint_map(q_real) + delta, delta = one offset (deg) per arm joint (pan, lift, elbow,
wrist flex, wrist roll), the same in every pair. A joint offset moves the gripper (and the camera
with it) relative to the cube, so the jaws stay put in the wrist view while the cube shifts, which
is what the pairs show; the side view constrains it (an arm error across the side camera's line of
sight would show there).

Cost per pair: wrist cube outline -> real wrist edges (px, capped at 12)
             + side arm outline -> real side edges (px, capped at 12) x SIDE_WEIGHT
Checks: leave-one-pose-out. Writes joint_offsets_fit.json and joint_offsets_fit.png; changes nothing else.
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
from fit_wrist_pose import nelder_mead  # noqa: E402
from run_policy_real import H, W, SimView  # noqa: E402
from wrist_lens import orient_wrist  # noqa: E402
from wrist_offset_cause import SIDE_OK_PX, Scene  # noqa: E402

SIDE_WEIGHT = 1.0
MAX_DEG = 6.0
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]


class Fit:
    def __init__(self, folder):
        idx = json.loads((folder / "pairs.json").read_text())
        gap = {r["n"]: r for r in json.loads((folder / "gap.json").read_text())["pairs"]}
        self.sim = SimView(SIM / idx["scene"])
        self.sc = Scene(self.sim)
        self.side_r = mujoco.Renderer(self.sim.m, H, W)
        self.side_r.enable_segmentation_rendering()
        m = self.sim.m
        self.arm = np.flatnonzero(np.isin(m.geom_bodyid, [m.body(b).id for b in
                                  ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")]))
        jmap = joint_units.JointMap.load()
        self.items = []
        for e in idx["pairs"]:
            g = gap[e["n"]]
            side, wc = g["side"]["geometry"]["cube"], g["wrist"]["geometry"]["cube"]
            if not side or abs(side["dx"]) > SIDE_OK_PX or abs(side["dy"]) > SIDE_OK_PX or not wc or wc["resid_px"] > 5:
                continue
            self.items.append({
                "n": e["n"], "pose_id": e["pose_id"], "pose": tuple(e["pose"]),
                "q": jmap.real_to_sim(np.array(e["joints_real"], np.float32)).astype(float),
                "wdist": mg.real_edges(cv2.imread(str(folder / e["files"]["real_wrist"]))),
                "sdist": mg.real_edges(cv2.imread(str(folder / e["files"]["real_side"])))})

    @staticmethod
    def chamfer(dist, mask):
        pts = np.argwhere(mg.outline(mask))
        if len(pts) < 40:
            return 12.0
        return float(np.minimum(dist[pts[:, 0], pts[:, 1]], 12.0).mean())

    def score(self, it, delta_deg):
        q = it["q"].copy()
        q[:5] += np.radians(delta_deg)
        self.sc.pose(q, *it["pose"])
        wseg = orient_wrist(self.sc.seg())
        w = self.chamfer(it["wdist"], np.isin(wseg, self.sc.cube))
        self.side_r.update_scene(self.sim.d, camera="ext_cam")
        s = self.chamfer(it["sdist"], np.isin(self.side_r.render()[..., 0], self.arm))
        return w, s

    def cost(self, p, items):
        if np.abs(p).max() > MAX_DEG:
            return 1e3
        return float(np.mean([w + SIDE_WEIGHT * s for w, s in (self.score(it, p) for it in items)]))

    def fit(self, items):
        best = (np.zeros(5), self.cost(np.zeros(5), items))
        for rnd in range(3):
            x, v = nelder_mead(lambda p: self.cost(p, items), best[0], np.full(5, 1.5 * 0.5 ** rnd), iters=300, tol=1e-4)
            if v < best[1]:
                best = (x, v)
        return best

    def report(self, p, items):
        s = np.array([self.score(it, p) for it in items])
        return {"wrist_cube_px": round(float(np.median(s[:, 0])), 2), "side_arm_px": round(float(np.median(s[:, 1])), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    a = ap.parse_args()
    folder = SIM / a.pairs
    F = Fit(folder)
    items, zero = F.items, np.zeros(5)
    print(f"{len(items)} usable pairs")
    before = F.report(zero, items)
    print("before:", before, flush=True)
    loo = {}
    for pid in sorted({i["pose_id"] for i in items}):
        x, _ = F.fit([i for i in items if i["pose_id"] != pid])
        test = [i for i in items if i["pose_id"] == pid]
        loo[pid] = {"before": F.report(zero, test), "after": F.report(x, test), "offsets_deg": x.round(2).tolist()}
        print(f"held-out pose {pid}: {loo[pid]['before']} -> {loo[pid]['after']}  offsets {dict(zip(NAMES, loo[pid]['offsets_deg']))}",
              flush=True)
    x, _ = F.fit(items)
    after = F.report(x, items)
    print(f"\nall pairs: {before} -> {after}\noffsets (deg, sim = map(real) + offset): {dict(zip(NAMES, x.round(2)))}")
    res = {"before": before, "after": after, "offsets_deg": dict(zip(NAMES, x.round(3).tolist())),
           "leave_one_pose_out": loo, "side_weight": SIDE_WEIGHT, "n_pairs": len(items)}
    (folder / "joint_offsets_fit.json").write_text(json.dumps(res, indent=1))

    tiles = []
    for it in sorted(items, key=lambda i: i["n"])[:: max(len(items) // 6, 1)][:6]:
        real = cv2.imread(str(folder / f"{it['n']:03d}_real_wrist.png"))
        for p, col in ((zero, (0, 0, 255)), (x, (0, 255, 0))):
            q = it["q"].copy()
            q[:5] += np.radians(p)
            F.sc.pose(q, *it["pose"])
            real[mg.outline(np.isin(orient_wrist(F.sc.seg()), F.sc.cube))] = col
        cv2.putText(real, f"n{it['n']} pose {it['pose_id']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(real)
    cv2.imwrite(str(folder / "joint_offsets_fit.png"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    print("wrote joint_offsets_fit.json, joint_offsets_fit.png (red: now, green: with offsets)")


if __name__ == "__main__":
    main()
