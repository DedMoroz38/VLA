"""Cube poses v2 (2026-11-21): keep the train poses that already have a successful demo,
move everything else further from the robot (towards the side camera), where the arm is
easier to drive by hand.

    cd sim
    python make_cube_poses_v2.py            # eval/cube_poses.json (v1 kept as eval/cube_poses_v1.json)

- kept: every v1 'train' pose with a successful demo in demos/cube/ (same id, same x/y/yaw, so
  the teleop's resume still counts them), first in the train order
- new region: x 0.16-0.26, y 0.08-0.18 (still >= 8 cm to the robot's left), limited to where a
  straight-down grasp at cube height reaches within 3 mm (checked with the arm's IK for every
  pose; the far-left corner beyond ~27 cm from the base is out)
- new train poses (to make 100 in total) + 20 test: a scrambled Halton sequence over the new
  region and yaw -45..45 (evenly spread, like the v1 Latin hypercube), the 20 test poses being
  the most spread subset of the candidates (as in v1); ids 200+ (train), 300+ (test)
- 100 sim_test: independent random poses in the new region, ids 1000+
"""
import datetime
import glob
import json
import pathlib

import mujoco
import numpy as np

import make_cube_poses as v1
from demo_io import task_for
from pick_place_task import APPROACH_Z, FREE_SPACE_TOL, PolicyRunner, SAMPLE_YAW, load_scene

HERE = pathlib.Path(__file__).parent
REGION_X, REGION_Y = (0.16, 0.26), (0.08, 0.18)
GRASP_Z, REACH_TOL = 0.015, 0.003
SEED = 20261122


def reach_error(x, y):
    m, d = load_scene(str(HERE / v1.SCENE))
    adr = m.jnt_qposadr[m.body("cube").jntadr[0]]
    d.qpos[adr:adr + 3] = (0.5, 0.5, 0.02)                 # cube out of the way
    mujoco.mj_forward(m, d)
    r = PolicyRunner(m, d)
    r.move_to(np.array([x, y, APPROACH_Z]), pos_tol=FREE_SPACE_TOL)
    r.move_to(np.array([x, y, GRASP_Z]), chase_pitch=True)
    return float(np.linalg.norm(r.ik.site_pos(d) - [x, y, GRASP_Z]))


def halton(n, base, skip=20):
    out = []
    for i in range(skip, skip + n):
        f, r, k = 1.0, 0.0, i
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out.append(r)
    return np.array(out)


def main():
    old = json.loads((HERE / "eval/cube_poses_v1.json").read_text())
    done = set()
    for f in glob.glob(str(HERE / "demos/cube/teleop_*.npz")):
        meta = json.loads(str(np.load(f)["metadata_json"]))
        if meta.get("success") and meta.get("pose_file") == "cube_poses.json":
            done.add(meta["pose_id"])
    kept = [dict(p) for p in old["poses"] if p["split"] == "train" and p["id"] in done]
    print(f"keeping {len(kept)} train poses with a successful demo: {sorted(done)}")

    rng = np.random.default_rng(SEED)
    bounds = (REGION_X, REGION_Y, SAMPLE_YAW["cube"])
    box = v1.tape_box(v1.SCENE)
    shift = rng.uniform(size=3)
    n_need = 100 - len(kept) + 20
    cand, i = [], 0
    while len(cand) < n_need:                              # scrambled Halton, IK-checked
        u = (np.stack([halton(1, b, 20 + i)[0] for b in (2, 3, 5)]) + shift) % 1.0
        i += 1
        p = np.array([lo + uu * (hi - lo) for uu, (lo, hi) in zip(u, bounds)])
        if v1.clear_of(box, *p) and reach_error(p[0], p[1]) <= REACH_TOL:
            cand.append(p)
    rejected = i - len(cand)
    cand = np.array(cand)
    q = v1.normalised(cand, bounds)
    best, best_score, scores = None, -1.0, []
    for _ in range(5000):
        pick = rng.choice(len(cand), 20, replace=False)
        sc = v1.spread(q[pick])
        scores.append(sc)
        if sc > best_score:
            best, best_score = pick, sc
    test_idx = set(best.tolist())
    entry = lambda i, p, s: {"id": i, "x": float(p[0]), "y": float(p[1]), "yaw_rad": float(p[2]),
                             "yaw_deg": float(np.degrees(p[2])), "split": s}
    new_train = [entry(200 + k, p, "train") for k, p in enumerate(c for j, c in enumerate(cand) if j not in test_idx)]
    test = [entry(300 + k, p, "test") for k, p in enumerate(c for j, c in enumerate(cand) if j in test_idx)]
    extra = []
    while len(extra) < 100:
        p = np.array([rng.uniform(*b) for b in bounds])
        if v1.clear_of(box, *p) and reach_error(p[0], p[1]) <= REACH_TOL:
            extra.append(entry(1000 + len(extra), p, "sim_test"))
    poses = kept + new_train + test + extra
    out = dict(old)
    out.update({
        "description": ("Cube start poses v2: the train poses already demonstrated (v1, region x 0.12-0.22) "
                        "+ the rest of train, all of test and sim_test in a region further from the robot "
                        "(x 0.16-0.26, y 0.08-0.18, reachable top-down within 3 mm). make_cube_poses_v2.py"),
        "created": datetime.date.today().isoformat(), "seed": SEED, "version": 2,
        "region": {"x": list(REGION_X), "y": list(REGION_Y), "yaw_deg": [-45.0, 45.0],
                   "reach": "straight-down grasp at z 0.015 within 3 mm (IK)"},
        "kept_from_v1": sorted(done), "test_spread": float(best_score),
        "counts": {s: sum(p["split"] == s for p in poses) for s in ("train", "test", "sim_test")},
        "poses": poses,
    })
    (HERE / "eval/cube_poses.json").write_text(json.dumps(out, indent=2))
    v1.CUBE_SAMPLE_X, v1.CUBE_SAMPLE_Y = REGION_X, REGION_Y           # draw the new region box
    v1.draw_map(HERE / "eval/cube_poses.png", poses, box)
    print(f"wrote eval/cube_poses.json {out['counts']} (+png); {rejected} candidates rejected as unreachable; "
          f"test spread {best_score:.3f} vs {np.median(scores):.3f} typical")


if __name__ == "__main__":
    main()
