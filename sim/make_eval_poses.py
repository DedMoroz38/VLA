"""Write the fixed test set of eraser start poses that every policy is evaluated on.

Stage 0 of plans/baseline-v2.md. Policies used to be compared on poses drawn
fresh at eval time, 20-50 at a time -- too few to separate a real improvement
from luck (+/-20 points at n=20). From now on every evaluation uses the same
file, so any two results are measured on identical poses.

The poses are sample_eraser_pose() draws from seed 20261022, the seed v1 was
evaluated with, so the first 50 are exactly v1's 50 test poses and the 42%
baseline stays directly comparable on that subset.

Each pose is also annotated with:
  - whether the scripted expert solves it -- the teacher's ceiling, and a flag
    for poses that are hard in themselves (the base/tray corner);
  - its distance to the nearest start pose of each training demo set, so a
    result can be split into "near training data" and "far from it".

Usage:
    python make_eval_poses.py                         # -> eval/test_poses.json
    python make_eval_poses.py -n 100 --seed 20261022 --out eval/test_poses.json
"""
import argparse
import datetime
import json
import pathlib

import numpy as np

from eval_policy import nearest_training_pose, training_poses
from pick_place_task import (
    SCENE_PATH, PolicyRunner, load_scene, object_in_container,
    run_scripted_pick_place, sample_eraser_pose, set_eraser_pose,
)

DEFAULT_OUT = "eval/test_poses.json"
TRAIN_SETS = {
    "teleop": "demos/trimmed/*.npz",
    "mimicgen": "demos/generated_trimmed/*.npz",
    "scripted": "demos/scripted_trimmed/*.npz",
}


def expert_solves(pose, scene_path):
    model, data = load_scene(scene_path)
    set_eraser_pose(model, data, *pose)
    run_scripted_pick_place(PolicyRunner(model, data))
    return object_in_container(model, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scene", default=SCENE_PATH)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    poses = [sample_eraser_pose(rng) for _ in range(args.num)]
    train = {name: training_poses([pattern]) for name, pattern in TRAIN_SETS.items()}

    entries = []
    for i, (x, y, yaw) in enumerate(poses):
        expert = expert_solves((x, y, yaw), args.scene)
        nearest = {name: nearest_training_pose((x, y, yaw), t)
                   for name, t in train.items() if len(t)}
        entries.append({"id": i, "x": float(x), "y": float(y),
                        "yaw_rad": float(yaw), "yaw_deg": float(np.degrees(yaw)),
                        "expert_success": bool(expert), "nearest_training": nearest})
        near = "  ".join(f"{name} {d['xy_m'] * 1000:3.0f} mm"
                         for name, d in nearest.items())
        print(f"[{i:3d}] x={x:.3f} y={y:+.3f} yaw={np.degrees(yaw):+6.1f}°  "
              f"expert {'OK  ' if expert else 'FAIL'}  nearest: {near}", flush=True)

    k = sum(e["expert_success"] for e in entries)
    print(f"\nscripted expert solves {k}/{len(entries)} test poses")
    for name in train:
        d = np.array([e["nearest_training"][name]["xy_m"] for e in entries
                      if name in e["nearest_training"]])
        if len(d):
            print(f"  nearest {name:8s} start pose: median {np.median(d) * 1000:.0f} mm, "
                  f"max {d.max() * 1000:.0f} mm  ({len(train[name])} demos)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "Fixed eval set: eraser start poses every policy is tested on",
        "seed": args.seed, "num": len(entries),
        "created": datetime.date.today().isoformat(),
        "expert_successes": k,
        "poses": entries}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
