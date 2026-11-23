"""Generate demonstrations automatically from the recorded human ones.

Samples a new eraser pose, picks a source demo, retargets it with
mimicgen.retarget, executes the result in MuJoCo, and keeps the episode if the
eraser ends up in the tray. Generated episodes are written in exactly the same
.npz format as recorded ones, so replay_demo.py validates them unchanged.

Source demos that cannot be segmented (a regrasp gives no single grasp event)
are reported and skipped.

Usage:
    python generate_demos.py -n 50
    python generate_demos.py -n 200 -o demos/generated --seed 1
    python generate_demos.py -n 20 --keep-failures     # for inspecting misses
    python generate_demos.py --source "demos/teleop_00[0-5].npz" -n 10
"""
import argparse
import collections
import glob
import pathlib
import time

import mujoco
import numpy as np

from demo_io import load_demo, next_demo_path
from mimicgen import generate_episode, usable_sources
from pick_place_task import SCENE_PATH, sample_eraser_pose

DEFAULT_OUT = pathlib.Path(__file__).parent / "demos" / "generated"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=50,
                    help="episodes to attempt")
    ap.add_argument("--source", default="demos/teleop_*.npz",
                    help="glob for the source demos (quote it)")
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    ap.add_argument("--scene", default=SCENE_PATH)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--keep-failures", action="store_true",
                    help="also write episodes that missed, flagged success=False")
    ap.add_argument("--prefix", default="gen")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.source))
    if not paths:
        ap.error(f"no source demos matched {args.source!r}")
    usable, rejected = usable_sources([load_demo(p) for p in paths])
    for demo, reason in rejected:
        print(f"skipping {demo.path.name}: {reason}")
    if not usable:
        ap.error("no usable source demos")
    print(f"{len(usable)} usable source demo(s), generating {args.num} episodes\n")

    model = mujoco.MjModel.from_xml_path(args.scene)
    rng = np.random.default_rng(args.seed)
    kept = 0
    per_source = collections.Counter()
    failures = collections.Counter()
    started = time.perf_counter()

    for i in range(args.num):
        demo, seg = usable[i % len(usable)]          # round-robin over sources
        target = sample_eraser_pose(rng)
        recorder, info = generate_episode(model, demo, seg, target)

        ok = info["success"]
        per_source[info["source_demo"]] += int(ok)
        kept += int(ok)
        if not ok:
            # Where it went wrong, roughly: never picked up vs. dropped short.
            failures["eraser never left the table"
                     if info["final_eraser_pos"][2] < 0.02 else
                     "eraser placed outside the tray"] += 1

        if ok or args.keep_failures:
            path = recorder.save(
                next_demo_path(args.out, prefix=args.prefix),
                scene_path=args.scene, success=ok, source="mimicgen",
                extra_metadata=info)
            label = path.name
        else:
            label = "(discarded)"

        x, y, yaw = target
        print(f"[{i + 1:4d}/{args.num}] {'OK  ' if ok else 'FAIL'} "
              f"x={x:.3f} y={y:+.3f} yaw={np.degrees(yaw):+6.1f}° "
              f"from {info['source_demo']:16s} "
              f"dyaw={info['yaw_delta_deg']:+6.1f}° {label}")

    elapsed = time.perf_counter() - started
    print(f"\n{kept}/{args.num} succeeded ({100 * kept / args.num:.0f}%), "
          f"{elapsed:.0f}s ({elapsed / args.num:.1f}s per episode)")
    if failures:
        print("failure modes:")
        for reason, count in failures.most_common():
            print(f"  {count:4d}  {reason}")
    if per_source:
        print("successes per source demo:")
        for name, count in sorted(per_source.items()):
            attempts = sum(1 for i in range(args.num)
                           if usable[i % len(usable)][0].path.name == name)
            print(f"  {name:16s} {count:3d}/{attempts}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
