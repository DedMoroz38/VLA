"""Headless scripted sanity check of the full pick-and-place pipeline.

Drives the arm (via the same IK the interactive controller uses) through:
ready -> above eraser -> roll to its yaw -> descend -> close -> lift -> above
container -> descend -> open -> retreat, then checks the eraser ends up resting
inside the container. Prints a PASS/FAIL summary.

The motion itself lives in pick_place_task.run_scripted_pick_place, shared with
record_scripted_demo.py so the demo we record is the trajectory we verified.

With -n it becomes the expert's success-rate check: N random eraser poses from
the reachable region, a per-pose line, and a summary by failed stage and region.
The scripted expert should clear 95% before any training data is drawn from it.

Usage:
    python verify_pick_place.py                       # keyframe pose, stage by stage
    python verify_pick_place.py --eraser 0.22 0.05 45 # x y yaw_degrees
    python verify_pick_place.py -n 200 --seed 1       # success rate over random poses
"""
import argparse
import collections
import time

import numpy as np

from pick_place_task import (
    SCENE_PATH, PolicyRunner, load_scene, object_in_container,
    run_scripted_pick_place, sample_eraser_pose, set_eraser_pose,
)


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def run_once(scene_path, pose=None):
    """Run the scripted policy once. Returns (success, stages, frames, model, data)."""
    model, data = load_scene(scene_path)
    if pose is not None:
        set_eraser_pose(model, data, *pose)
    frames = []
    runner = PolicyRunner(model, data, on_control_step=lambda d, a: frames.append(1))
    stages = run_scripted_pick_place(runner)
    # Judged on the outcome, like teleop and generated demos: a waypoint that
    # stopped a few mm outside its tolerance is reported but is not a failure.
    success = object_in_container(model, data)
    return success, stages, len(frames), model, data


def single(scene_path, pose):
    success, stages, frames, model, data = run_once(scene_path, pose)
    for name, ok in stages:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    eraser_bid = model.body("eraser").id
    print(f"\nfinal eraser pos={np.round(data.xpos[eraser_bid], 3)} "
          f"speed={np.linalg.norm(data.cvel[eraser_bid][:3]):.4f}  "
          f"episode {frames} frames")
    print("OVERALL:", "PASS" if success else "FAIL")
    return 0 if success else 1


def sweep(scene_path, n, seed):
    rng = np.random.default_rng(seed)
    results = []
    t0 = time.time()
    for i in range(n):
        pose = sample_eraser_pose(rng)
        success, stages, frames, _, _ = run_once(scene_path, pose)
        failed = next((name for name, ok in stages if not ok),
                      None if success else "not at rest in tray")
        results.append((pose, success, failed, frames))
        x, y, yaw = pose
        note = (f"  failed: {failed}" if not success
                else f"  (missed stage: {failed})" if failed else "")
        print(f"[{i + 1:3d}/{n}] {'OK  ' if success else 'FAIL'} x={x:.3f} y={y:+.3f} "
              f"yaw={np.degrees(yaw):+6.1f}°  {frames:4d} frames{note}", flush=True)

    k = sum(r[1] for r in results)
    lo, hi = wilson_interval(k, n)
    print(f"\nsuccess: {k}/{n} = {100 * k / n:.0f}%  "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%)  "
          f"[{time.time() - t0:.0f}s wall]")
    frames = np.array([r[3] for r in results if r[1]])
    if len(frames):
        print(f"successful episode length: mean {frames.mean():.0f} frames "
              f"({frames.mean() * 0.03:.1f}s), min {frames.min()}, max {frames.max()}")

    failures = collections.Counter(r[2] for r in results if not r[1])
    for name, count in failures.most_common():
        print(f"    {count:3d}  failed at: {name}")
    missed = collections.Counter(r[2] for r in results if r[1] and r[2])
    for name, count in missed.most_common():
        print(f"    {count:3d}  succeeded despite missing stage: {name}")

    def by(label, keep):
        sub = [r for r in results if keep(r[0])]
        if sub:
            ok = sum(r[1] for r in sub)
            print(f"  {label:22s} {ok}/{len(sub)} = {100 * ok / len(sub):.0f}%")

    print("by region:")
    by("y < 0.06", lambda p: p[1] < 0.06)
    by("y >= 0.06", lambda p: p[1] >= 0.06)
    by("|yaw| < 45°", lambda p: abs(p[2]) < np.pi / 4)
    by("|yaw| >= 45°", lambda p: abs(p[2]) >= np.pi / 4)
    return 0 if k == n else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=0,
                    help="run on this many random eraser poses and report the rate")
    ap.add_argument("--seed", type=int, default=0, help="pose sampling seed (with -n)")
    ap.add_argument("--eraser", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                    help="single run at this eraser pose instead of the keyframe")
    ap.add_argument("--scene", default=SCENE_PATH)
    args = ap.parse_args()

    if args.num:
        return sweep(args.scene, args.num, args.seed)
    pose = None
    if args.eraser:
        x, y, yaw_deg = args.eraser
        pose = (x, y, np.radians(yaw_deg))
    return single(args.scene, pose)


if __name__ == "__main__":
    raise SystemExit(main())
