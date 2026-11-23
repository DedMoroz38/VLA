"""Evaluate a trained LeRobot policy closed-loop in the sim, on unseen eraser poses.

Phase 4 of plans/train-pick-place-policy.md. There is no lerobot-eval env for
this task, so this is the control loop from pick_place_teleop.py with the
policy standing in for the keyboard: each control step render ext_cam and
wrist_cam, read the arm's joint positions, ask the policy for an action, hold
it for CONTROL_SUBSTEPS physics steps, mj_forward.

"Unseen" means the eraser start pose is not taken from any training episode.
By default the poses come from the fixed test set eval/test_poses.json (100
poses, written by make_eval_poses.py), so every policy is measured on the same
poses; --seed draws a fresh random set instead. The start poses
of the training demos are read back from their first frame, and every test
episode reports its distance to the nearest one, so the summary can show
whether success depends on how far a pose is from the training data.

Success is object_in_container() -- the same test the recorder, the trimmer
and MimicGen all use -- held for SUCCESS_HOLD_FRAMES consecutive frames, which
also ends the episode early.

Usage:
    python eval_policy.py outputs/train/act_pick_place/checkpoints/last/pretrained_model
    python eval_policy.py CKPT --out eval/act_040000.json        # the fixed 100-pose set
    python eval_policy.py CKPT -n 50                             # its first 50 (= v1's test set)
    python eval_policy.py CKPT -n 20 --seed 7                    # fresh random poses instead
    python eval_policy.py CKPT --shard 0/4 --out eval/x_0.json   # every 4th pose; run the
                                                                 # shards in parallel, then
    python merge_eval.py eval/x_0.json eval/x_1.json ...         # one combined report
    python eval_policy.py CKPT -n 5 --video-dir eval/videos      # also write mp4s
    python eval_policy.py CKPT --poses eval/cube_poses.json --split test       # cube: the 20 held-out
    python eval_policy.py CKPT --poses eval/cube_poses.json --split sim_test   # cube: 100 more (sim only)
    python eval_policy.py CKPT --temporal-ensemble 0.01          # re-plan every step
    python eval_policy.py CKPT --n-action-steps 25               # re-plan every 0.75 s
"""
import argparse
import glob
import json
import pathlib
import time

import mujoco
import numpy as np

import joint_units
from demo_io import CONTROL_SUBSTEPS, load_demo, task_for
from pick_place_task import (
    SCENE_PATH, get_eraser_pose, object_in_container, reset_to_ready,
    sample_eraser_pose, set_eraser_pose, task_object)

CAMERAS = ("ext_cam", "wrist_cam")
DEFAULT_TRAIN_DEMOS = ("demos/trimmed/*.npz", "demos/generated_trimmed/*.npz")
DEFAULT_TEST_POSES = "eval/test_poses.json"
# The longest trimmed training episode is 1108 frames (33 s); 1500 (45 s)
# leaves room for a slow but eventually successful attempt.
DEFAULT_MAX_STEPS = 1500
SUCCESS_HOLD_FRAMES = 10
# Eraser z above which it counts as picked up (it rests at z=0.015).
LIFTED_Z = 0.04


# Rotational symmetry of each pick object about z: the eraser's square cross-section
# repeats every 180 deg, the cube every 90 deg.
YAW_PERIOD = {"eraser": np.pi, "cube": np.pi / 2}


def yaw_dist(a, b, period=np.pi):
    """Angular distance between two object yaws, modulo the object's symmetry."""
    d = (a - b) % period
    return min(d, period - d)


def quat_yaw(quat):
    w, _, _, z = quat
    return float(2.0 * np.arctan2(z, w))


def training_poses(patterns):
    """(x, y, yaw) of the eraser at frame 0 of every training demo."""
    poses = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            demo = load_demo(path)
            x, y, _ = demo.obj_pos[0]
            poses.append((float(x), float(y), quat_yaw(demo.obj_quat[0])))
    return np.array(poses)


def nearest_training_pose(pose, train, period=np.pi):
    """Distance from `pose` to the closest training start pose: (xy metres,
    yaw degrees of that same closest-in-xy demo, min yaw degrees overall)."""
    if len(train) == 0:
        return None
    xy = np.linalg.norm(train[:, :2] - np.asarray(pose[:2]), axis=1)
    i = int(np.argmin(xy))
    yaws = [yaw_dist(pose[2], t, period) for t in train[:, 2]]
    return {"xy_m": float(xy[i]),
            "yaw_deg_at_nearest_xy": float(np.degrees(yaws[i])),
            "min_yaw_deg": float(np.degrees(min(yaws)))}


def load_policy(checkpoint, device, n_action_steps=None, temporal_ensemble=None):
    """Load a policy plus its pre/post-processors.

    n_action_steps / temporal_ensemble override the checkpoint's inference
    settings, which need no retraining. They have to be set before the policy
    is built, since ACTPolicy decides in __init__ whether to create its
    temporal ensembler. Temporal ensembling queries the policy every step, so
    it forces n_action_steps=1.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    if temporal_ensemble is not None:
        if not hasattr(cfg, "temporal_ensemble_coeff"):
            raise ValueError(f"{cfg.type} has no temporal ensembling")
        cfg.temporal_ensemble_coeff = temporal_ensemble
        cfg.n_action_steps = 1
    elif n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}})
    return policy, preprocessor, postprocessor


from wrist_lens import orient_wrist  # noqa: E402


def wrist_lens_for(model):
    """The real wrist lens (wrist_lens.py) if this model's wrist_cam carries the calibrated
    intrinsics (so101_real.xml), else None (older scenes keep a plain pinhole wrist view)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cid < 0 or model.cam_sensorsize[cid][0] <= 0:
        return None
    from wrist_lens import WristLens
    return WristLens()


def checkpoint_joint_units(preprocessor):
    """'real' or 'sim', from the observation.state normalisation stats stored with
    the checkpoint: sim radians stay within about +-3, real LeRobot degrees don't
    (shoulder_lift / elbow_flex means are tens of degrees)."""
    for step in getattr(preprocessor, "steps", []):
        stats = getattr(step, "stats", None) or {}
        st = stats.get("observation.state")
        if st:
            vals = [np.abs(np.asarray(st[k], float)).max() for k in ("mean", "min", "max") if k in st]
            if vals:
                return "real" if max(vals) > 6 else "sim"
    return None


def run_episode(model, data, renderer, policy, preprocessor, postprocessor,
                pose, device, max_steps, video=None, jmap=None):
    """jmap: a joint_units.JointMap if the policy works in real-arm units, else None."""
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    obj = task_object(model)                           # 'eraser' or 'cube'
    task = task_for(obj)
    reset_to_ready(model, data)
    set_eraser_pose(model, data, *pose, object_body=obj)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    lens = wrist_lens_for(model)
    obj_bid = model.body(obj).id
    lo, hi = model.actuator_ctrlrange[:6].T
    max_z = float(data.xpos[obj_bid][2])
    held = 0
    steps = 0
    t0 = time.perf_counter()

    for steps in range(1, max_steps + 1):
        state = data.qpos[:6] if jmap is None else jmap.sim_to_real(data.qpos[:6])
        obs = {"observation.state": np.asarray(state, np.float32)}
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            img = renderer.render().copy()
            if cam == "wrist_cam" and lens is not None:
                img = lens.distort(img)                # the real wrist camera's lens distortion
                img = orient_wrist(img)                # rotated 90 deg anticlockwise, as in training
            obs[f"observation.images.{cam}"] = img
        if video is not None:
            # Both cameras side by side: external view | wrist view (scaled to the same height;
            # the real-look wrist view is portrait).
            ext, wr = (obs[f"observation.images.{cam}"] for cam in CAMERAS)
            if wr.shape[0] != ext.shape[0]:
                import cv2
                wr = cv2.resize(wr, (round(wr.shape[1] * ext.shape[0] / wr.shape[0]) // 2 * 2, ext.shape[0]))
            video.append(np.concatenate([ext, wr], axis=1))

        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                obs, torch.device(device), task, "so101")
            action = postprocessor(policy.select_action(preprocessor(batch)))
        action = action.squeeze(0).float().cpu().numpy()
        if jmap is not None:
            action = jmap.real_to_sim(action)

        data.ctrl[:6] = np.clip(action, lo, hi)
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        max_z = max(max_z, float(data.xpos[obj_bid][2]))
        held = held + 1 if object_in_container(model, data, obj) else 0
        if held >= SUCCESS_HOLD_FRAMES:
            break

    success = held >= SUCCESS_HOLD_FRAMES
    if success:
        outcome = "success"
    elif max_z < LIFTED_Z:
        outcome = "never picked up"
    else:
        outcome = "picked up, not placed in container"
    return {
        "success": success,
        "outcome": outcome,
        "steps": steps,
        "sim_s": steps * CONTROL_SUBSTEPS * model.opt.timestep,
        "wall_s": time.perf_counter() - t0,
        "max_eraser_z": max_z,
        "final_eraser_pose": list(get_eraser_pose(model, data, obj)),
        "final_eraser_z": float(data.xpos[obj_bid][2]),
    }


def write_video(path, frames, fps):
    # PyAV rather than imageio: lerobot already depends on it for its own video.
    import fractions
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(
            "libx264", rate=fractions.Fraction(fps).limit_denominator(1000))
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(results):
    """Print the report for a list of per-episode results. Returns (k, n, lo, hi).

    Shared with merge_eval.py, which feeds it the results of several shards.
    """
    n = len(results)
    k = sum(r["success"] for r in results)
    lo, hi = wilson_interval(k, n)
    print(f"\nsuccess: {k}/{n} = {100 * k / n:.0f}%  "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%)")
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    for name, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {name}")
    ok_times = [r["sim_s"] for r in results if r["success"]]
    if ok_times:
        print(f"time to success: mean {np.mean(ok_times):.1f}s, "
              f"min {min(ok_times):.1f}s, max {max(ok_times):.1f}s")

    def split(title, groups):
        print(f"{title}:")
        for label, keep in groups:
            sub = [r for r in results if keep(r)]
            if sub:
                ks = sum(r["success"] for r in sub)
                print(f"  {label:26s} {ks:3d}/{len(sub):3d} = {100 * ks / len(sub):3.0f}%")

    start = lambda r: r["eraser_start"]
    # yaw split at the middle of the range: +-90 deg poses (eraser) at 45, +-45 (cube) at 22.5
    yaw_cut = 45 if max(abs(start(r)["yaw_deg"]) for r in results) > 46 else 22.5
    split("by region", [
        ("y < 0.06 (near the tray)", lambda r: start(r)["y"] < 0.06),
        ("y >= 0.06", lambda r: start(r)["y"] >= 0.06),
        ("x < 0.19 (near the base)", lambda r: start(r)["x"] < 0.19),
        ("x >= 0.19", lambda r: start(r)["x"] >= 0.19),
        (f"|yaw| < {yaw_cut:g} deg", lambda r: abs(start(r)["yaw_deg"]) < yaw_cut),
        (f"|yaw| >= {yaw_cut:g} deg", lambda r: abs(start(r)["yaw_deg"]) >= yaw_cut),
    ])

    near = [r["nearest_training"] for r in results if r.get("nearest_training")]
    if near and len(near) == n:
        d = np.array([x["xy_m"] for x in near])
        median = float(np.median(d))
        print(f"distance to nearest training pose: median {median * 1000:.0f} mm, "
              f"max {d.max() * 1000:.0f} mm")
        split("by distance to training data", [
            (f"closer than {median * 1000:.0f} mm", lambda r: r["nearest_training"]["xy_m"] < median),
            (f"{median * 1000:.0f} mm or further", lambda r: r["nearest_training"]["xy_m"] >= median),
        ])
    return k, n, lo, hi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a pretrained_model/ directory")
    ap.add_argument("-n", "--episodes", type=int, default=None,
                    help="evaluate only the first N test poses (default: all)")
    ap.add_argument("--poses", default=DEFAULT_TEST_POSES,
                    help="fixed test-pose file from make_eval_poses.py or make_cube_poses.py")
    ap.add_argument("--split", default=None,
                    help="only the poses with this split in the pose file (cube_poses.json: "
                         "'test' = the 20 held-out poses, 'sim_test' = 100 more for sim-only eval)")
    ap.add_argument("--seed", type=int, default=None,
                    help="draw fresh random test poses from this seed instead "
                         "of using --poses")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--scene", default=None,
                    help="default: the pose file's scene, else the eraser scene")
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--train-demos", nargs="*", default=None,
                    help="training demos, for the distance-to-training report")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="actions executed per policy query (default: the "
                         "checkpoint's own, 100 for the ACT run)")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    metavar="COEFF",
                    help="ACT temporal ensembling with this coefficient (the "
                         "ACT paper uses 0.01); implies --n-action-steps 1")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="evaluate only every Nth pose, starting at I (e.g. 0/4), "
                         "so N jobs can split the test set; combine with merge_eval.py")
    ap.add_argument("--train-poses", action="store_true",
                    help="evaluate on training start poses instead of unseen "
                         "ones (spread evenly over the training set) -- the "
                         "in-distribution baseline to compare against")
    ap.add_argument("--video-dir", default=None,
                    help="write an ext_cam mp4 for the first --videos episodes")
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--out", default=None, help="write per-episode results as JSON")
    ap.add_argument("--joint-units", choices=("auto", "real", "sim"), default="auto",
                    help="units the policy's state/action use: real (LeRobot degrees, "
                         "converted with calib/joint_map.json) or sim (radians); "
                         "auto reads it from the checkpoint's normalisation stats")
    args = ap.parse_args()
    pose_file = {}
    if not args.train_poses and args.seed is None:
        pose_file = json.loads(pathlib.Path(args.poses).read_text())
    if args.scene is None:
        args.scene = (str(pathlib.Path(__file__).parent / pose_file["scene"])
                      if "scene" in pose_file else SCENE_PATH)
    if args.train_demos is None:
        args.train_demos = pose_file.get("train_demos", list(DEFAULT_TRAIN_DEMOS))

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")

    train = training_poses(args.train_demos)
    print(f"{len(train)} training start poses loaded")
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, device, args.n_action_steps, args.temporal_ensemble)
    units = args.joint_units
    if units == "auto":
        units = checkpoint_joint_units(preprocessor)
        if units is None:
            ap.error("could not tell the checkpoint's joint units; pass --joint-units real|sim")
    jmap = joint_units.JointMap.load() if units == "real" else None
    print(f"joint units: {units}" + (f" (offsets {jmap.meta['offsets_deg']})" if jmap else ""))
    print(f"policy: {policy.config.type} from {args.checkpoint} on {device}, "
          f"n_action_steps={policy.config.n_action_steps}, temporal_ensemble="
          f"{getattr(policy.config, 'temporal_ensemble_coeff', None)}")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    obj = task_object(model)
    period = YAW_PERIOD[obj]
    print(f"scene {pathlib.Path(args.scene).name}: object '{obj}', task \"{task_for(obj)}\"")
    renderer = mujoco.Renderer(model, args.height, args.width)
    fps = 1.0 / (CONTROL_SUBSTEPS * model.opt.timestep)
    pose_source = args.poses
    if args.train_poses:
        if len(train) == 0:
            ap.error("--train-poses needs --train-demos to match something")
        count = min(args.episodes or 50, len(train))
        picks = np.linspace(0, len(train) - 1, count)
        poses = [tuple(train[int(round(j))]) for j in picks]
        pose_source = "training"
    elif args.seed is not None:
        rng = np.random.default_rng(args.seed)
        poses = [sample_eraser_pose(rng, object_body=obj) for _ in range(args.episodes or 50)]
        pose_source = f"random seed {args.seed}"
    else:
        test_set = pose_file["poses"]
        if args.split is not None:
            test_set = [p for p in test_set if p.get("split") == args.split]
            if not test_set:
                ap.error(f"no poses with split '{args.split}' in {args.poses}")
            pose_source += f" [{args.split}]"
        test_set = test_set[:args.episodes]
        poses = [(p["x"], p["y"], p["yaw_rad"]) for p in test_set]
    pose_ids = ([p.get("id", j) for j, p in enumerate(test_set)]
                if not args.train_poses and args.seed is None else list(range(len(poses))))
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        if not 0 <= index < count:
            ap.error("--shard I/N needs 0 <= I < N")
        poses, pose_ids = poses[index::count], pose_ids[index::count]
    print(f"{len(poses)} test poses from {pose_source}"
          + (f" (shard {args.shard})" if args.shard else ""))

    results = []
    try:
        for i, pose in enumerate(poses):
            frames = [] if args.video_dir and i < args.videos else None
            r = run_episode(model, data, renderer, policy, preprocessor,
                            postprocessor, pose, device, args.max_steps, frames, jmap)
            r["episode"] = i
            r["pose_id"] = pose_ids[i]
            r["eraser_start"] = {"x": pose[0], "y": pose[1],
                                 "yaw_deg": float(np.degrees(pose[2]))}
            r["nearest_training"] = nearest_training_pose(pose, train, period)
            results.append(r)
            if frames:
                tag = "ok" if r["success"] else "fail"
                write_video(pathlib.Path(args.video_dir) / f"pose{pose_ids[i]:03d}_{tag}.mp4",
                            frames, fps)

            near = r["nearest_training"]
            near_s = (f"  nearest train {near['xy_m'] * 1000:4.0f} mm"
                      if near else "")
            print(f"[{i + 1:3d}/{len(poses)}] "
                  f"{'OK  ' if r['success'] else 'FAIL'} "
                  f"x={pose[0]:.3f} y={pose[1]:+.3f} yaw={np.degrees(pose[2]):+6.1f}°"
                  f"  {r['sim_s']:5.1f}s sim  {r['outcome']}{near_s}", flush=True)
    finally:
        renderer.close()

    k, n, lo, hi = summarize(results)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "poses": pose_source,
            "shard": args.shard,
            "episodes": n, "successes": k, "success_rate": k / n if n else 0.0,
            "ci95": [lo, hi], "max_steps": args.max_steps,
            "n_action_steps": policy.config.n_action_steps,
            "temporal_ensemble_coeff": getattr(
                policy.config, "temporal_ensemble_coeff", None),
            "results": results}, indent=2))
        print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
