"""Record demonstrations headlessly with the scripted reference policy.

The scripted expert handles any eraser pose in the sampled region (it rolls the
wrist to the eraser's yaw), so with --randomize this is the main source of
clean training data: smooth, pause-free, as many episodes as wanted. Check its
success rate first with `verify_pick_place.py -n 200`.

Episodes where the eraser does not end up in the tray are not saved unless
--keep-failures is given; recording continues until -n successes are saved.

Usage:
    python record_scripted_demo.py                          # one demo, keyframe pose -> demos/
    python record_scripted_demo.py -n 5                     # five demos
    python record_scripted_demo.py -o demos/src             # choose the output dir
    python record_scripted_demo.py -n 200 --randomize --seed 3 -o demos/scripted
    python record_scripted_demo.py -n 50 --pose 0.21 0.07 30 --jitter 3 3 -o demos/onepose
        # one pose +/-3 mm, +/-3 deg: the single-pose overfit test of plans/baseline-v2.md
"""
import argparse
import pathlib

import numpy as np

from demo_io import DemoRecorder, next_demo_path
from pick_place_task import (
    SCENE_PATH, PolicyRunner, get_eraser_pose, load_scene, object_in_container,
    run_scripted_pick_place, sample_eraser_pose, set_eraser_pose,
)

DEFAULT_OUT = pathlib.Path(__file__).parent / "demos"


def record_one(out_path, scene_path=SCENE_PATH, pose=None, keep_failures=True):
    """Record one episode. Returns (path or None if not saved, frames, success, stages)."""
    model, data = load_scene(scene_path)
    if pose is not None:
        set_eraser_pose(model, data, *pose)
    x, y, yaw = get_eraser_pose(model, data)
    recorder = DemoRecorder(model)
    runner = PolicyRunner(model, data, on_control_step=recorder.capture)

    stages = run_scripted_pick_place(runner)
    success = object_in_container(model, data)
    if not success and not keep_failures:
        return None, len(recorder), success, stages

    path = recorder.save(out_path, scene_path=scene_path, success=success,
                         source="scripted",
                         extra_metadata={
                             "stages": [list(s) for s in stages],
                             "eraser_start": {"x": x, "y": y, "yaw_rad": yaw,
                                              "yaw_deg": float(np.degrees(yaw))}})
    return path, len(recorder), success, stages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=1, help="demos to save")
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--scene", default=SCENE_PATH)
    ap.add_argument("--prefix", default="scripted")
    ap.add_argument("--randomize", action="store_true",
                    help="random eraser pose per episode (default: keyframe pose)")
    ap.add_argument("--seed", type=int, default=None, help="pose sampling seed")
    ap.add_argument("--pose", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                    help="use this eraser pose for every episode")
    ap.add_argument("--jitter", type=float, nargs=2, metavar=("MM", "DEG"),
                    default=(0.0, 0.0),
                    help="with --pose: uniform +/- jitter on x, y (mm) and yaw (deg)")
    ap.add_argument("--keep-failures", action="store_true",
                    help="also save episodes that end with the eraser outside the tray")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    saved = failures = 0
    while saved < args.num:
        if args.pose:
            x, y, yaw_deg = args.pose
            mm, deg = args.jitter
            pose = (x + rng.uniform(-mm, mm) / 1000, y + rng.uniform(-mm, mm) / 1000,
                    np.radians(yaw_deg + rng.uniform(-deg, deg)))
        else:
            pose = sample_eraser_pose(rng) if args.randomize else None
        out_path = next_demo_path(args.out, prefix=args.prefix)
        path, frames, success, stages = record_one(
            out_path, args.scene, pose, keep_failures=args.keep_failures)
        if path is not None:
            saved += 1
        if success:
            print(f"{path}  {frames} frames  success")
            continue
        failures += 1
        failed = [name for name, ok in stages if not ok]
        print(f"{path if path else '(not saved)'}  {frames} frames  FAILED"
              + (f"  stages: {', '.join(failed)}" if failed else ""))
        if not (args.randomize or args.pose) and not args.keep_failures:
            break   # the keyframe pose is deterministic; retrying cannot help
    print(f"\n{saved} saved, {failures} failed attempts")
    return 1 if failures and not (args.randomize or args.pose) else 0


if __name__ == "__main__":
    raise SystemExit(main())
