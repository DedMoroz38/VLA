"""Refit the wrist camera's mount pose (position + orientation in the camera_mount frame) from
matched real / sim pairs. The lens (intrinsics + distortion) stays the checkerboard calibration.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/refit_wrist_cam.py calib/pairs/20261123_202747            # fit + check
    /opt/anaconda3/envs/lerobot/bin/python calib/refit_wrist_cam.py calib/pairs/20261123_202747 --write    # + so101_real.xml

Why: in the pairs the real cube sits ~21 px left of the sim one in the wrist view, while the side
camera shows the cube on its outline (<= 5 px, ~1 px/mm) and the arm within ~2 px, and the real
jaws are shifted too (calib/wrist_offset_cause.py). So the sim wrist camera sits/points slightly
differently on the gripper than the real one. The previous fit (fit_wrist_pose.py) used only the
jaw silhouette of one frame; near objects alone can't separate a small turn of the camera from a
small shift. The cube, 5-20 cm away, can.

Cost over the usable pairs (cube visible, cube on its outline in the side view):
  cube term  mean distance (px, capped at 12) from the sim cube outline to real edges
  jaw term   1 - IoU of the sim jaws with the real jaw mask (cream on grey), x JAW_WEIGHT
Checks: leave-one-pose-out (fit without a pose, score it), and the change in mm / degrees.
"""
import argparse
import json
import pathlib
import re
import shutil
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
from fit_wrist_pose import nelder_mead, quat_mul, rotvec_quat  # noqa: E402
from run_policy_real import SimView  # noqa: E402
from wrist_lens import orient_wrist  # noqa: E402
from wrist_offset_cause import SIDE_OK_PX, Scene, real_jaw_mask  # noqa: E402

ROBOT = SIM / "mujoco_menagerie/robotstudio_so101/so101_real.xml"
JAW_WEIGHT = 10.0
MAX_SHIFT_M = 0.008          # a mount can't plausibly be off by more than this; guards the search


class Fit:
    def __init__(self, folder):
        idx = json.loads((folder / "pairs.json").read_text())
        gap = {r["n"]: r for r in json.loads((folder / "gap.json").read_text())["pairs"]}
        self.sim = SimView(SIM / idx["scene"])
        self.sc = Scene(self.sim)
        jmap = joint_units.JointMap.load()
        m = self.sim.m
        self.cam = m.camera("wrist_cam").id
        self.pos0, self.quat0 = m.cam_pos[self.cam].copy(), m.cam_quat[self.cam].copy()
        self.items = []
        for e in idx["pairs"]:
            g = gap[e["n"]]
            side, wc = g["side"]["geometry"]["cube"], g["wrist"]["geometry"]["cube"]
            if not side or abs(side["dx"]) > SIDE_OK_PX or abs(side["dy"]) > SIDE_OK_PX or not wc or wc["resid_px"] > 5:
                continue
            real = cv2.imread(str(folder / e["files"]["real_wrist"]))
            jaw = real_jaw_mask(real)
            self.items.append({"n": e["n"], "pose_id": e["pose_id"], "pose": tuple(e["pose"]),
                               "q": jmap.real_to_sim(np.array(e["joints_real"], np.float32)),
                               "dist": mg.real_edges(real), "jaw": jaw if jaw.sum() > 2000 else None})

    def unpack(self, p):
        return self.pos0 + p[:3], quat_mul(self.quat0, rotvec_quat(p[3:]))

    def set_cam(self, pos, quat):
        self.sim.m.cam_pos[self.cam] = pos
        self.sim.m.cam_quat[self.cam] = quat / np.linalg.norm(quat)

    def score(self, it):
        """(cube px, 1 - jaw IoU or None) for one pair at the current camera."""
        self.sc.pose(it["q"], *it["pose"])
        seg = self.sc.seg()
        cube = np.isin(orient_wrist(seg), self.sc.cube)
        pts = np.argwhere(mg.outline(cube))
        c = float(np.minimum(it["dist"][pts[:, 0], pts[:, 1]], 12.0).mean()) if len(pts) >= 40 else 12.0
        j = None
        if it["jaw"] is not None:
            s = np.isin(seg, self.sc.jaw)
            j = 1 - (s & it["jaw"]).sum() / max((s | it["jaw"]).sum(), 1)
        return c, j

    def cost(self, p, items):
        if np.linalg.norm(p[:3]) > MAX_SHIFT_M:
            return 1e3
        self.set_cam(*self.unpack(p))
        tot = 0.0
        for it in items:
            c, j = self.score(it)
            tot += c + (JAW_WEIGHT * j if j is not None else 0.0)
        return tot / len(items)

    def fit(self, items):
        best = (np.zeros(6), self.cost(np.zeros(6), items))
        for rnd in range(3):
            step = np.r_[[0.003] * 3, [0.03] * 3] * (0.5 ** rnd)
            x, v = nelder_mead(lambda p: self.cost(p, items), best[0], step, iters=300, tol=1e-4)
            if v < best[1]:
                best = (x, v)
        return best

    def report(self, p, items):
        self.set_cam(*self.unpack(p))
        s = [self.score(it) for it in items]
        c = np.array([x[0] for x in s])
        j = np.array([1 - x[1] for x in s if x[1] is not None])
        return {"cube_px_median": round(float(np.median(c)), 2), "cube_px_mean": round(float(c.mean()), 2),
                "jaw_iou_median": round(float(np.median(j)), 3) if len(j) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    folder = SIM / a.pairs
    F = Fit(folder)
    items = F.items
    print(f"{len(items)} usable pairs; poses {sorted({i['pose_id'] for i in items})}")
    zero = np.zeros(6)
    before = F.report(zero, items)
    print("before:", before)

    # leave one pose out: does a fit without a pose improve that pose?
    loo = {}
    for pid in sorted({i["pose_id"] for i in items}):
        train = [i for i in items if i["pose_id"] != pid]
        test = [i for i in items if i["pose_id"] == pid]
        x, _ = F.fit(train)
        loo[pid] = {"before": F.report(zero, test), "after": F.report(x, test),
                    "shift_mm": (x[:3] * 1000).round(1).tolist(), "turn_deg": np.degrees(x[3:]).round(2).tolist()}
        print(f"held-out pose {pid}: {loo[pid]['before']} -> {loo[pid]['after']}  "
              f"(shift {loo[pid]['shift_mm']} mm, turn {loo[pid]['turn_deg']} deg)", flush=True)

    x, v = F.fit(items)
    after = F.report(x, items)
    pos, quat = F.unpack(x)
    quat = quat / np.linalg.norm(quat)
    res = {"before": before, "after": after, "leave_one_pose_out": loo,
           "shift_mm_in_camera_mount": (x[:3] * 1000).round(2).tolist(),
           "turn_deg_rotvec": np.degrees(x[3:]).round(3).tolist(),
           "turn_deg_total": round(float(np.degrees(np.linalg.norm(x[3:]))), 3),
           "pos_old": F.pos0.round(5).tolist(), "quat_old": F.quat0.round(6).tolist(),
           "pos": [round(float(v), 5) for v in pos], "quat": [round(float(v), 6) for v in quat],
           "pairs": a.pairs, "n_pairs": len(items), "jaw_weight": JAW_WEIGHT}
    print(f"\nall pairs: {before} -> {after}")
    print(f"camera moved {res['shift_mm_in_camera_mount']} mm, turned {res['turn_deg_total']} deg")
    (folder / "wrist_refit.json").write_text(json.dumps(res, indent=1))

    # overlay: real wrist frames with the old (red) and new (green) sim cube + jaw outlines
    tiles = []
    for it in sorted(items, key=lambda i: i["n"])[:: max(len(items) // 6, 1)][:6]:
        real = cv2.imread(str(folder / f"{it['n']:03d}_real_wrist.png"))
        for p, col in ((zero, (0, 0, 255)), (x, (0, 255, 0))):
            F.set_cam(*F.unpack(p))
            F.sc.pose(it["q"], *it["pose"])
            seg = orient_wrist(F.sc.seg())
            real[mg.outline(np.isin(seg, F.sc.cube)) | mg.outline(np.isin(seg, F.sc.jaw))] = col
        cv2.putText(real, f"n{it['n']} pose {it['pose_id']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(real)
    cv2.imwrite(str(folder / "wrist_refit.png"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    print("wrote wrist_refit.json, wrist_refit.png (red: old camera, green: refit)")

    if a.write:
        s = ROBOT.read_text()
        el = re.search(r'<camera name="wrist_cam"[^>]*/>', s, re.S).group(0)
        new = re.sub(r'pos="[^"]*"', 'pos="' + " ".join(f"{v:.5f}" for v in res["pos"]) + '"', el)
        new = re.sub(r'quat="[^"]*"', 'quat="' + " ".join(f"{v:.6f}" for v in res["quat"]) + '"', new)
        bak = ROBOT.with_suffix(".xml.before_wrist_refit")
        if not bak.exists():
            shutil.copy(ROBOT, bak)
        ROBOT.write_text(s.replace(el, new))
        mujoco.MjModel.from_xml_path(str(SIM / json.loads((folder / "pairs.json").read_text())["scene"]))
        print("wrote the refit wrist_cam pose into", ROBOT.name, "(backup", bak.name + ")")


if __name__ == "__main__":
    main()
