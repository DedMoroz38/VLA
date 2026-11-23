"""Cube poses v4 (2026-11-21): add 10 'test' poses in the far region the v3 train poses
cover (the 20 v2 test poses are all in x 0.16-0.26), and mark pose 251 as having its demo
dropped (teleop_071: the arm caught on its own shoulder for 1.5 s).

    cd sim
    python make_cube_poses_v4.py          # updates eval/cube_poses.json (v3 kept as eval/cube_poses_v3.json)

Far test poses: region x 0.24-0.33, y 0.00-0.20 as v3, plus at most 35.5 cm from the base
(the one v3 pose at 38.5 cm was unreachable by hand; the far demos reached 36 cm). Same
checks as v3 (free-pitch reach within 3 mm, whole cube in the side view, >= 20 mm from the
resting arm, clear of the tape). 60 scrambled-Halton candidates; the 10 most spread out
are kept. ids 500-509, split 'test'.
"""
import json
import pathlib

import numpy as np

import make_cube_poses as v1
import make_cube_poses_v3 as v3
from pick_place_task import SAMPLE_YAW

HERE = pathlib.Path(__file__).parent
N_TEST, N_CAND, SEED, MAX_R = 10, 60, 20261124, 0.355


def main():
    path = HERE / "eval/cube_poses.json"
    data = json.loads(path.read_text())
    if any(500 <= p["id"] < 600 for p in data["poses"]):
        raise SystemExit("already has the v4 test poses (ids 500+)")
    rng = np.random.default_rng(SEED)
    shift = rng.uniform(size=3)
    bounds = (v3.REGION_X, v3.REGION_Y, SAMPLE_YAW["cube"])
    box = v1.tape_box(v1.SCENE)
    chk = v3.Checks()
    cand, i = [], 20
    while len(cand) < N_CAND:
        u = (np.array([v3.halton1(i, b) for b in (2, 3, 5)]) + shift) % 1.0
        i += 1
        x, y, yaw = (lo + uu * (hi - lo) for uu, (lo, hi) in zip(u, bounds))
        if (np.hypot(x, y) <= MAX_R and chk.in_view(x, y, yaw) and chk.arm_clearance(x, y, yaw) >= 0.020
                and v1.clear_of(box, x, y, yaw) and v3.reach_error(x, y) <= v3.REACH_TOL):
            cand.append((x, y, yaw))
    cand = np.array(cand)
    q = v1.normalised(cand, bounds)
    best, best_score = None, -1.0
    for _ in range(5000):
        pick = rng.choice(len(cand), N_TEST, replace=False)
        sc = v1.spread(q[pick])
        if sc > best_score:
            best, best_score = pick, sc
    new = [{"id": 500 + k, "x": float(cand[j][0]), "y": float(cand[j][1]), "yaw_rad": float(cand[j][2]),
            "yaw_deg": float(np.degrees(cand[j][2])), "split": "test"} for k, j in enumerate(sorted(best))]
    for p in data["poses"]:
        if p["id"] == 251:
            p["demo_dropped"] = ("teleop_071 dropped 2026-11-21: the fixed jaw caught on the shoulder while "
                                 "unfolding, arm stuck 1.5 s with the command 53 deg ahead")
    tests = [p for p in data["poses"] if p["split"] == "test"]
    others = [p for p in data["poses"] if p["split"] != "test"]
    data["poses"] = [p for p in others if p["split"] == "train"] + tests + new + [p for p in others if p["split"] != "train"]
    data["version"] = 4
    data["description"] += " v4: + 10 far test poses (ids 500-509); pose 251's demo dropped. make_cube_poses_v4.py"
    data["counts"] = {s: sum(p["split"] == s for p in data["poses"]) for s in ("train", "test", "sim_test", "dropped")}
    path.write_text(json.dumps(data, indent=2))
    v1.CUBE_SAMPLE_X, v1.CUBE_SAMPLE_Y = v3.REGION_X, v3.REGION_Y
    v1.draw_map(HERE / "eval/cube_poses.png", [p for p in data["poses"] if p["split"] != "dropped"], box)
    r = [np.hypot(p["x"], p["y"]) * 100 for p in new]
    print(f"added {len(new)} far test poses ({min(r):.1f}-{max(r):.1f} cm from the base), spread {best_score:.3f}; "
          f"counts {data['counts']}")


if __name__ == "__main__":
    main()
