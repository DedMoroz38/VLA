"""Render camera frames from trimmed demos and build a LeRobotDataset.

This is "step 5" from README.md -- unwritten there, because camera images are
deliberately not stored in the .npz demos (see demo_io.py): mj_state is a full
mjSTATE_INTEGRATION state, so any camera view can be re-rendered exactly on
demand. This script does that re-rendering, once per frame per camera, and
writes the result straight into a LeRobotDataset (parquet + mp4), ready for
lerobot-train.

Run trim_demo.py first -- this script does not trim tails itself, and
exporting the raw demos as-is would carry their long stationary tails
(seconds for teleop, 10+ seconds for generated -- see plans/train-pick-place-policy.md)
straight into the training set.

Usage:
    python export_lerobot.py                                   # demos/trimmed + demos/generated_trimmed
    python export_lerobot.py "demos/trimmed/*.npz"              # just the human demos
    python export_lerobot.py --root lerobot_dataset/smoke_test "demos/trimmed/teleop_000.npz"
    python export_lerobot.py "demos/scripted_trimmed/*.npz" --width 320 --height 240

Joint units: by default observation.state and action are written in the REAL
arm's units, so a trained policy can drive the real SO-101 directly and real
LeRobot recordings can be mixed in: 5 arm joints in LeRobot degrees plus the
gripper in 0..100, named "shoulder_pan.pos" ... "gripper.pos" like
lerobot-record's datasets. The conversion (sim radians -> real) is
joint_units.py with calib/joint_map.json, which is copied into the dataset
root for the record. --joint-units sim keeps the old behaviour (sim radians,
named after the MuJoCo joints).

Sharded (each shard renders every Nth episode into its own dataset, so N jobs
run at once; merge_dataset.sh stitches the results back into one):
    python export_lerobot.py "demos/scripted_trimmed/*.npz" --shard 0/8 --root ds_shard0
"""
import argparse
import glob
import pathlib
import shutil
import time

import mujoco
import numpy as np

import joint_units

from demo_io import STATE_SPEC, load_demo
from wrist_lens import orient_wrist
from replay_demo import load_model_for

DEFAULT_DEMOS = ("demos/trimmed/*.npz", "demos/generated_trimmed/*.npz")
DEFAULT_REPO_ID = "local/so101-pick-place"
DEFAULT_ROOT = "lerobot_dataset/so101-pick-place"
CAMERAS = ("ext_cam", "wrist_cam")


def build_features(joint_names, width, height, wrist_rotated=False):
    return {
        "observation.state": {
            "dtype": "float32", "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
        "action": {
            "dtype": "float32", "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
        **{
            f"observation.images.{cam}": {
                # the real-look wrist view is rotated 90 deg (wrist_lens.orient_wrist): portrait
                "dtype": "video", "shape": ((width, height, 3) if cam == "wrist_cam" and wrist_rotated
                                            else (height, width, 3)),
                "names": ["height", "width", "channels"],
            }
            for cam in CAMERAS
        },
    }


def wrist_lens_for(model):
    """The real wrist lens (wrist_lens.py) if this model's wrist_cam carries the calibrated
    intrinsics (so101_real.xml), else None (older scenes keep a plain pinhole wrist view)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cid < 0 or model.cam_sensorsize[cid][0] <= 0:
        return None
    from wrist_lens import WristLens
    return WristLens()


def export_episode(dataset, path, width, height, scene_override=None, jmap=None):
    """jmap: a joint_units.JointMap to write real-arm units, or None for sim radians."""
    demo = load_demo(path)
    model = load_model_for(demo, scene_override)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    lens = wrist_lens_for(model)

    try:
        for k in range(len(demo)):
            mujoco.mj_setState(model, data, demo.mj_state[k], STATE_SPEC)
            mujoco.mj_forward(model, data)

            state, action = demo.joint_pos[k], demo.action[k]
            if jmap is not None:
                state, action = jmap.sim_to_real(state), jmap.sim_to_real(action)
            frame = {
                "observation.state": np.asarray(state, np.float32),
                "action": np.asarray(action, np.float32),
                "task": demo.meta["task"],
            }
            for cam in CAMERAS:
                renderer.update_scene(data, camera=cam)
                img = renderer.render().copy()
                if cam == "wrist_cam" and lens is not None:
                    img = lens.distort(img)            # the real wrist camera's lens distortion
                    img = orient_wrist(img)            # rotated 90 deg anticlockwise, as used for real
                frame[f"observation.images.{cam}"] = img
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        renderer.close()
    return len(demo)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="*", default=list(DEFAULT_DEMOS),
                    help=f".npz demo files (quote globs); default: {DEFAULT_DEMOS}")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="local dataset directory")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--scene", default=None, help="override the recorded scene path")
    ap.add_argument("--joint-units", choices=("real", "sim"), default="real",
                    help="real (default): LeRobot degrees + gripper 0..100, via "
                         "calib/joint_map.json; sim: MuJoCo radians")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete --root first if it already exists")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="export only every Nth episode, starting at I (e.g. 0/8)")
    ap.add_argument("--vcodec", default="auto",
                    help="video codec ('auto' picks a hardware encoder if "
                         "available, else libsvtav1 -- much slower)")
    ap.add_argument("--preset", default=None,
                    help="encoder speed preset, e.g. 'veryfast' for libx264 "
                         "(codec-specific; file size doesn't matter much here)")
    args = ap.parse_args()

    paths = []
    for pattern in args.demos:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    missing = [p for p in paths if not pathlib.Path(p).exists()]
    if missing:
        ap.error(f"no such demo file(s): {', '.join(missing)}")
    if not paths:
        ap.error("no demo files matched")

    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        if not 0 <= index < count:
            ap.error("--shard I/N needs 0 <= I < N")
        paths = paths[index::count]
        if not paths:
            ap.error(f"shard {args.shard} has no episodes")

    root = pathlib.Path(args.root)
    if root.exists():
        if not args.overwrite:
            ap.error(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    first = load_demo(paths[0])
    jmap = joint_units.JointMap.load() if args.joint_units == "real" else None
    joint_names = joint_units.REAL_NAMES if jmap is not None else first.meta["joint_names"]
    if jmap is not None:
        print(f"joint units: real (LeRobot), offsets {jmap.meta['offsets_deg']}, "
              f"gripper span {jmap.span:.1f} deg  [{joint_units.MAP_PATH.name}, {jmap.meta['updated']}]")
    else:
        print("joint units: sim (MuJoCo radians)")
    fps = round(first.meta["fps"])
    # real-look scenes (calibrated wrist lens) store the wrist view rotated 90 deg (portrait)
    wrist_rotated = wrist_lens_for(load_model_for(first, args.scene)) is not None
    if wrist_rotated:
        print("wrist_cam: real lens distortion + rotated 90 deg anticlockwise "
              f"-> {args.height}x{args.width} portrait (wrist_lens.orient_wrist)")

    # Imported here so --help doesn't pay for importing lerobot/torch.
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        fps=fps,
        features=build_features(joint_names, args.width, args.height, wrist_rotated),
        robot_type="so101",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec, preset=args.preset),
    )

    print(f"{len(paths)} episodes -> {root}/  ({args.width}x{args.height} @ {fps} fps)")
    start = time.time()
    total_frames = 0
    for i, path in enumerate(paths):
        n = export_episode(dataset, path, args.width, args.height, args.scene, jmap)
        total_frames += n
        elapsed = time.time() - start
        print(f"  [{i + 1}/{len(paths)}] {pathlib.Path(path).name}: {n} frames "
              f"({elapsed:.0f}s elapsed, {total_frames} frames total)")

    dataset.finalize()
    if jmap is not None:
        shutil.copy(joint_units.MAP_PATH, root / "joint_map.json")
    print(f"\ndone: {len(paths)} episodes, {total_frames} frames, "
          f"{time.time() - start:.0f}s -> {root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
