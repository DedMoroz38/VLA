"""Cut the stationary tail off recorded demonstrations before export.

Every demo in demos/ and demos/generated/ keeps running for a while after the
eraser has already settled in the tray: teleop demos carry the operator's
reaction time between success and pressing P (seconds); generated demos carry
the entire unclipped tail of their source demo replayed through a slower
tracker (often 10+ seconds -- see mimicgen.py's `Segmentation.phase_b`, which
runs to `seg.n`, not to the settle point).

This trims each episode to shortly after `object_in_container()` -- the same
success test the recorder itself used -- first becomes durably true, keeping
a short settle buffer (`--settle-s`, default 0.3s) so the policy still sees
"hold at the goal," not three-to-thirteen seconds of it.

Raw demos are never modified. Trimmed copies go to mirrored directories
(demos/trimmed/, demos/generated_trimmed/ by default) with a `trim` block
added to the metadata.

Usage:
    python trim_demo.py "demos/*.npz" "demos/generated/*.npz"
    python trim_demo.py "demos/*.npz" --settle-s 0.5
"""
import argparse
import glob
import json
import pathlib

import mujoco
import numpy as np

from demo_io import CONTROL_SUBSTEPS, STATE_SPEC, DemoRecorder, load_demo
from pick_place_task import object_in_container
from replay_demo import load_model_for

DEFAULT_OUT_TELEOP = "demos/trimmed"
DEFAULT_OUT_GENERATED = "demos/generated_trimmed"


def success_flags(demo, model):
    """object_in_container() evaluated at every recorded frame."""
    data = mujoco.MjData(model)
    object_body = demo.meta["object_body"]
    flags = np.zeros(len(demo), dtype=bool)
    for k in range(len(demo)):
        mujoco.mj_setState(model, data, demo.mj_state[k], STATE_SPEC)
        mujoco.mj_forward(model, data)
        flags[k] = object_in_container(model, data, object_body)
    return flags


def find_cutoff(flags, fps, settle_s):
    """Last index to keep: settle_s after the success test stops flipping False.

    Scanning for the *last* False index (rather than the first True) guards
    against a transient dip mid-approach being mistaken for the settle point;
    only a dip after the true settle point could fool this, and a demo that
    ends successfully shouldn't have one.
    """
    n = len(flags)
    fail_idx = np.where(~flags)[0]
    last_fail = int(fail_idx[-1]) if len(fail_idx) else -1
    settle_frames = round(settle_s * fps)
    return min(last_fail + 1 + settle_frames, n - 1)


MAX_EXTEND_S = 3.0


def extend_until_settled(demo, model, flags, fps, settle_s):
    """Episodes stopped the instant the object landed end before the success test has held for
    settle_s. Continue them in the sim, holding the last action (the arm keeps still, as if the
    operator had waited before pressing P), until it has, at most MAX_EXTEND_S. Returns
    (extra frame arrays or None, extended success flags)."""
    settle_frames = round(settle_s * fps)
    fail_idx = np.where(~flags)[0]
    held = len(flags) - 1 - (int(fail_idx[-1]) if len(fail_idx) else -1)
    if held >= settle_frames + 1:
        return None, flags
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, demo.mj_state[-1], STATE_SPEC)
    hold = np.asarray(demo.action[-1], float)
    data.ctrl[:len(hold)] = hold
    for _ in range(CONTROL_SUBSTEPS):                 # the last recorded action, applied as recorded
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    rec = DemoRecorder(model, object_body=demo.meta["object_body"])
    extra_flags = []
    run = 0 if not flags[-1] else held
    for _ in range(round(MAX_EXTEND_S * fps)):
        ok = object_in_container(model, data, demo.meta["object_body"])
        rec.capture(data, hold)
        extra_flags.append(ok)
        run = run + 1 if ok else 0
        if run >= settle_frames + 1:
            break
        data.ctrl[:len(hold)] = hold
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    extra = rec.arrays()
    # keep the time axis continuous with the recording
    extra["time"] = demo.time[-1] + (np.arange(len(extra["time"])) + 1) / fps
    return extra, np.concatenate([flags, extra_flags])


def trim_demo(path, out_dir, settle_s, scene_override=None, extend=True):
    """Trim one demo file. Returns a stats dict, or None if it was skipped."""
    demo = load_demo(path)
    out_path = pathlib.Path(out_dir) / pathlib.Path(path).name

    if not demo.meta["success"]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, metadata_json=json.dumps(demo.meta, indent=2),
            **{k: getattr(demo, k) for k in demo.keys()})
        print(f"  {pathlib.Path(path).name}: recorded success=False, "
              f"copied through unchanged (no trim)")
        return None

    model = load_model_for(demo, scene_override)
    flags = success_flags(demo, model)
    if not flags.any():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, metadata_json=json.dumps(demo.meta, indent=2),
            **{k: getattr(demo, k) for k in demo.keys()})
        print(f"  {pathlib.Path(path).name}: WARNING success=True recorded "
              f"but object_in_container() never true on replay; copied "
              f"through unchanged, check this demo")
        return None

    n = len(demo)
    fps = demo.meta["fps"]
    extra = None
    if extend:
        extra, flags = extend_until_settled(demo, model, flags, fps, settle_s)
    full = {k: getattr(demo, k) for k in demo.keys()}
    if extra is not None:
        full = {k: np.concatenate([full[k], extra[k]]) for k in full}
    cutoff = find_cutoff(flags, fps, settle_s)
    n_ext = 0 if extra is None else len(extra["time"])

    arrays = {k: v[:cutoff + 1] for k, v in full.items()}
    metadata = dict(demo.meta)
    metadata["num_frames"] = cutoff + 1
    metadata["duration_s"] = float(arrays["time"][-1] - arrays["time"][0])
    metadata["trim"] = {
        "original_num_frames": n,
        "cutoff_frame": cutoff,
        "tail_removed_frames": max(n - 1 - cutoff, 0),
        "tail_removed_s": max(n - 1 - cutoff, 0) / fps,
        "settle_s": settle_s,
        # frames simulated after the recording (last action held) so the success test settles
        "extended_frames": max(cutoff + 1 - n, 0),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, metadata_json=json.dumps(metadata, indent=2),
                        **arrays)

    stats = {
        "name": pathlib.Path(path).name,
        "original_frames": n,
        "trimmed_frames": cutoff + 1,
        "tail_removed_s": metadata["trim"]["tail_removed_s"],
    }
    ext = metadata["trim"]["extended_frames"]
    print(f"  {stats['name']}: {n} -> {cutoff + 1} frames "
          + (f"(extended {ext} frames = {ext / fps:.2f}s, last action held, to let the object settle)"
             if ext else f"(removed {stats['tail_removed_s']:.2f}s)"))
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="+", help=".npz demo files (quote globs)")
    ap.add_argument("--settle-s", type=float, default=0.3,
                    help="seconds kept after the success test settles (default 0.3)")
    ap.add_argument("--out-teleop", default=DEFAULT_OUT_TELEOP,
                    help=f"output dir for non-generated demos (default {DEFAULT_OUT_TELEOP})")
    ap.add_argument("--out-generated", default=DEFAULT_OUT_GENERATED,
                    help=f"output dir for demos/generated/* (default {DEFAULT_OUT_GENERATED})")
    ap.add_argument("--scene", default=None, help="override the recorded scene path")
    ap.add_argument("--no-extend", action="store_true",
                    help="don't extend episodes that stop before the success test has settled")
    args = ap.parse_args()

    paths = []
    for pattern in args.demos:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    missing = [p for p in paths if not pathlib.Path(p).exists()]
    if missing:
        ap.error(f"no such demo file(s): {', '.join(missing)}")

    groups = {"teleop": [], "generated": []}
    for path in paths:
        group = "generated" if "generated" in pathlib.Path(path).parts else "teleop"
        groups[group].append(path)

    all_stats = []
    for group, group_paths in groups.items():
        if not group_paths:
            continue
        out_dir = args.out_generated if group == "generated" else args.out_teleop
        print(f"\n{group} ({len(group_paths)} files) -> {out_dir}/")
        for path in group_paths:
            stats = trim_demo(path, out_dir, args.settle_s, args.scene, extend=not args.no_extend)
            if stats:
                stats["group"] = group
                all_stats.append(stats)

    print(f"\n{'summary':-^70}")
    for group in ("teleop", "generated"):
        rows = [s for s in all_stats if s["group"] == group]
        if not rows:
            continue
        removed = [r["tail_removed_s"] for r in rows]
        saved_frames = sum(r["original_frames"] - r["trimmed_frames"] for r in rows)
        print(f"{group}: {len(rows)} trimmed, tail removed "
              f"mean={np.mean(removed):.2f}s min={np.min(removed):.2f}s "
              f"max={np.max(removed):.2f}s, {saved_frames} frames saved total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
