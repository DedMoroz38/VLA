# Plan: train a pick-and-place policy on the SO-101 sim demonstrations

Goal: go from the `.npz` demos already in `demos/` and `demos/generated/` to a
trained policy (ACT baseline, then SmolVLA) loadable and evaluable in this sim,
training against the local `../robot/lerobot` checkout.

## 0. What exists today

- 11 human teleop demos (`demos/teleop_*.npz`, all `success=True`, `teleop_009`
  excluded from augmentation for being a regrasp) and 165 MimicGen-augmented
  demos (`demos/generated/gen_*.npz`), per `README.md` / `demo_collection_notes.md`.
- Format: `demo_io.py` — one frame per 30 ms control step, `action` = 6 joint
  position targets, plus `joint_pos`, `ee_pos/quat`, `obj_pos/quat`,
  `mj_state` (full `mjSTATE_INTEGRATION`), and a metadata JSON blob
  (`task`, `success`, `fps`, `cameras: ["ext_cam", "wrist_cam"]`, …).
- Camera frames are **not** stored — by design (`README.md` "Camera images are
  deliberately not stored"), re-rendered from `mj_state` on demand. This is
  still true; nothing renders or exports to LeRobot's dataset format yet. The
  README calls this out as "step 5" and it's unwritten.
- `replay_demo.py` already has the two building blocks export needs: a
  `Replayer` that restores `mj_state[k]` and calls `mj_forward` per frame, and
  (in `view()`) a `mujoco.Renderer` loop over `cameras=("wrist_cam","ext_cam")`.
- `pick_place_task.py` defines `object_in_container(model, data)` — the
  authoritative success test (inside the tray XY/Z tolerance **and**
  `speed < AT_REST_SPEED (0.01)`), already imported by `replay_demo.py`.

## 1. The problem: long stationary tails

Measured directly from the files (`ee_pos` frame-to-frame delta < 1e-4 as the
"stopped moving" proxy):

| set | tail length |
| --- | --- |
| `demos/teleop_*.npz` (11 files) | 1.9 s – 3.7 s (63–123 frames), mean ≈ 2.9 s |
| `demos/generated/*.npz` (sampled) | 11.8 s – 13.5 s (392–451 frames), mean ≈ 12.7 s |

Root cause for teleop: `pick_place_teleop.py` stops recording on a keypress
(`P`), so every episode carries the operator's reaction time after the eraser
settles. For generated demos it's worse: `mimicgen.py`'s `phase_b` runs to
`seg.n` (`Segmentation.phase_b = slice(self.lift, self.n)`, `mimicgen.py:90`)
— i.e. it replays the *entire* source tail — and the closed-loop tracker
(`track()`/`_solve()`) apparently takes several times longer per waypoint than
the original teleop pacing, roughly quadrupling the dead time.

For a generated episode like `gen_000` (1048 frames), **40% of the episode is
the arm sitting motionless after the task is already solved.** Training on
that as-is means a big chunk of imitation loss is spent on a trivial
"do nothing" target, and the action chunks a policy like ACT/SmolVLA samples
during training will frequently span pre- and post-success frames glued
together with nothing happening in between.

Trim before export. Don't edit `demos/` in place — keep the raw recordings as
source of truth, write trimmed copies.

## Phase 0 — `trim_demo.py` (new script)

For each `.npz`:

1. Load it (`demo_io.load_demo`), build the model (`replay_demo.load_model_for`).
2. Replay in **state mode** (`mujoco.mj_setState(model, data, mj_state[k], STATE_SPEC)`
   + `mj_forward`, exactly `Replayer.step()`'s `"state"` branch), and at each
   `k` evaluate `object_in_container(model, data)` — the same function the
   recorder itself used to decide when a take was done, so the trim boundary
   is defined identically to "success," not a separately-tuned heuristic.
3. Find `last_fail` = the last frame index where `object_in_container` is
   `False` (scanning from the end guards against a transient dip mid-approach
   registering as "settled"; only a dip *after* the true settle point matters,
   and there shouldn't be one in a demo that ends successfully).
4. `cutoff = min(last_fail + 1 + settle_frames, n - 1)`, where
   `settle_frames = round(settle_s * fps)`, default `settle_s = 0.3` (~10
   frames at 33.3 Hz). Keep a short post-success buffer — don't cut the instant
   success flips true — so the policy still sees "hold position at the goal,"
   just not three-to-thirteen seconds of it.
5. Slice every per-frame array to `[: cutoff + 1]`, recompute
   `num_frames`/`duration_s` in the metadata, add a
   `"trim": {"original_num_frames", "cutoff_frame", "tail_removed_s", "settle_s"}`
   block, and write to a mirrored path (`demos/trimmed/teleop_000.npz`,
   `demos/generated_trimmed/gen_000.npz`).
6. Episodes with `meta["success"] is False` (only reachable via
   `generate_demos.py --keep-failures`): skip the trim (no valid success
   region to anchor on), copy through unchanged, and warn — these shouldn't
   be in the training set regardless.

CLI: `python trim_demo.py "demos/*.npz" "demos/generated/*.npz" --settle-s 0.3
--out-teleop demos/trimmed --out-generated demos/generated_trimmed`. Print a
before/after summary table (mean/min/max tail removed, total frames saved) —
expect teleop tails to drop to ~0.3 s and generated tails similarly, i.e.
removing on the order of 45,000+ frames (~22 minutes of wall-clock sim time)
from the generated set alone.

**Validation:** re-run the `replay_demo.py` **action-mode** check against the
trimmed files up to their new length (states up to `cutoff` must still
reproduce bit-exactly from `action` alone — trimming only removes trailing
frames, it doesn't touch the kept ones, so this should be a no-op check, but
run it anyway before trusting the export).

**Follow-up, not blocking:** the generated-demo tail is 4x longer than its
source for a reason that isn't understood yet (tracker convergence speed in
`generate_episode`/`track()`). Worth a one-line note in `mimicgen.py` or a
follow-up task to cap `phase_b` at its own settle point — that would fix this
at the source for future `generate_demos.py` runs instead of needing Phase 0
to do more work than it should.

## Phase 1 — `export_lerobot.py` (new script): render + build a `LeRobotDataset`

This is the unwritten "step 5." Pattern lifted directly from
`../robot/lerobot/examples/port_datasets/port_droid.py` (the closest existing
example of porting an external trajectory format into LeRobot's dataset):

```python
from lerobot.datasets import LeRobotDataset

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,),
        "names": {"axes": JOINT_NAMES}},          # == demo.meta["joint_names"]
    "action": {"dtype": "float32", "shape": (6,),
        "names": {"axes": JOINT_NAMES}},
    "observation.images.ext_cam": {"dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channels"]},
    "observation.images.wrist_cam": {"dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channels"]},
}

dataset = LeRobotDataset.create(
    repo_id="local/so101-pick-place",       # folder label; hub push is optional, later
    root="lerobot_dataset/so101-pick-place",
    fps=33,                                  # == round(demo.meta["fps"]); control_dt is exactly 0.03s
    features=FEATURES,
    robot_type="so101",
    use_videos=True,
)

for path in trimmed_demo_paths:              # demos/trimmed/*.npz + demos/generated_trimmed/*.npz
    demo = load_demo(path)
    model = load_model_for(demo)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    mujoco.mj_setState(model, data, demo.mj_state[0], STATE_SPEC)
    for k in range(len(demo)):
        mujoco.mj_setState(model, data, demo.mj_state[k], STATE_SPEC)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ext_cam");   ext = renderer.render().copy()
        renderer.update_scene(data, camera="wrist_cam"); wrist = renderer.render().copy()
        dataset.add_frame({
            "observation.state": demo.joint_pos[k].astype("float32"),
            "action": demo.action[k].astype("float32"),
            "observation.images.ext_cam": ext,
            "observation.images.wrist_cam": wrist,
            "task": demo.meta["task"],          # "Pick up the eraser and place it in the container"
        })
    dataset.save_episode()
```

Notes / things to confirm while implementing, not now:

- `fps=33` vs the true `control_dt=0.03` (exactly 33.333… Hz) — `LeRobotDataset.create`
  takes `fps: int`, and `tolerance_s` defaults to `1e-4`. Over a ~700-frame
  episode the rounding error accumulates to tens of ms; if dataset creation or
  `lerobot-train` complains about timestamp drift, pass a wider `tolerance_s`
  rather than fighting the fps field.
- Video encoding needs `ffmpeg` (per `AGENTS.md`'s note on video decoding) —
  check it's on `PATH` before running this at scale, both locally and later on
  Iridis.
- **Do this export locally, not on Iridis.** MuJoCo offscreen rendering here
  already works around a broken GLFW on this Mac via Tkinter (`README.md`
  "Notes"); Iridis compute nodes have no display and no guaranteed
  EGL/OSMesa context, and no internet access to fetch anything mid-job
  (`docs/Iridis-IDL.pdf`, "Basics"). Render + build the dataset here, then
  `rsync` the finished `lerobot_dataset/so101-pick-place/` (parquet + mp4,
  no MuJoCo dependency to read) to Iridis for the training step only.
- A held-out validation *split* of the dataset isn't necessary for this task:
  Phase 4's eval re-samples fresh random eraser poses through
  `sample_eraser_pose()` and runs the live sim, which already tests
  generalization better than holding out a slice of the same recorded
  trajectories would.

## Phase 2 — Dataset QA (before spending any GPU time)

- `huggingface.co/spaces/lerobot/visualize_dataset` (or the local equivalent)
  against the exported dataset — per `AGENT_GUIDE.md` §4.7, **always** do this
  before training. Look for: blank/black frames (wrong camera name or
  un-forwarded state), a wrist view that doesn't track the gripper, episodes
  that still end mid-task (a Phase 0 bug), and the task string being right.
- Spot-check frame counts: `sum(episode lengths) == sum(trimmed .npz lengths)`.
- Spot-check a handful of `ext_cam` frames by eye against
  `replay_demo.py ... --view` on the same source file — should look identical.

## Phase 3 — Train

Per `AGENT_GUIDE.md` §5.5 (the "golden rule") and §6: validate the pipeline
end-to-end with the cheap model before spending real GPU-hours on the
project's actual target.

**3a. ACT baseline (sanity check the data, not the final model).**

```bash
lerobot-train \
  --dataset.repo_id=local/so101-pick-place --dataset.root=lerobot_dataset/so101-pick-place \
  --policy.type=act --policy.device=cuda \
  --output_dir=outputs/train/act_pick_place --job_name=act_pick_place \
  --batch_size=8 --wandb.enable=true
```

If this doesn't comfortably clear the >70% success bar `AGENT_GUIDE.md` §8.3
sets as the "clean 50-episode" expectation, the problem is almost certainly
upstream (Phase 0 trimming, Phase 1 rendering/alignment) — fix that before
touching SmolVLA, per §5.8 ("policy ignores X → data/camera issue, not a
model issue").

**3b. SmolVLA (the actual target — `README.md` line 3, "demonstration pipeline
for the SmolVLA project").**

```bash
lerobot-train \
  --dataset.repo_id=local/so101-pick-place --dataset.root=lerobot_dataset/so101-pick-place \
  --policy.type=smolvla --policy.device=cuda \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --output_dir=outputs/train/smolvla_pick_place --job_name=smolvla_pick_place \
  --batch_size=4 --wandb.enable=true \
  --steps=<N> --policy.scheduler_decay_steps=<N> --save_freq=<N/10>
```

Unfreezing the vision encoder is called out in §7.6 as "usually improves
performance substantially" — do it from the start rather than as a later
iteration, since this is a single specialized task, exactly the case the
guide flags.

**Picking `--steps`:** apply §7.2's formula once Phase 0's trim summary gives
the real post-trim frame totals (teleop tails drop to ~0.3 s with ~10 frames
restored to roughly 600–750 frames/episode; generated episodes land in a
similar ~600–900 frame range once their much larger tail is cut). Ballpark
with 176 episodes × ~700 frames ≈ 123k total frames: at `batch_size=4`,
~31k steps/epoch, so 5–10 epochs ≈ 150k–310k steps — recompute from the
script's actual printed totals before committing to a number, and scale
`scheduler_decay_steps`/`save_freq` to match per §7.5.

**Where to run:** Iridis X (session already has access confirmed to the open
`l4` / `scavenger_l4` partitions under the `student` account;
`ecsstudents_l4` needs the registration form in `README.md`'s link first —
use open partitions meanwhile). Conda env `my-pytorch-env` is already set up
per `docs/conda-setup.sh`. `rsync` the exported `lerobot_dataset/` dir plus
this repo's scripts over, `pip install -e ../robot/lerobot[smolvla]` in that
env, then submit via `sbatch` (reuse `launch.sh`'s structure from the Iridis
tutorial repo: `module load conda/python3`, `conda activate
my-pytorch-env`, then the `lerobot-train` command above). WandB needs
`--mode offline` on compute nodes (no internet) per `docs/Iridis-IDL.pdf`
"Basics."

## Phase 4 — Evaluate in sim

No `lerobot-eval` path exists for this task (`src/lerobot/envs/` has no SO-101
pick-and-place env registered, and building one is out of scope here). Write
a small `eval_policy.py` instead:

- Load the trained policy: `PreTrainedPolicy.from_pretrained(checkpoint_dir)`
  (or the policy-specific class) via the factory in
  `../robot/lerobot/src/lerobot/policies/factory.py`.
- Loop: `sample_eraser_pose()` (`pick_place_task.py`) → reset sim → each
  control step render `ext_cam`/`wrist_cam` + read `joint_pos` → policy
  forward → `data.ctrl = action` → step `CONTROL_SUBSTEPS` → `mj_forward`.
  This mirrors `pick_place_teleop.py`'s control loop with the policy standing
  in for the keyboard.
- Success per episode = `object_in_container(model, data)` at episode end,
  same test used everywhere else in this repo.
- Run ≥ 50 episodes (`AGENT_GUIDE.md` §8.2 bar for a stable estimate) across
  the reachable region (`ERASER_SAMPLE_X/Y/YAW` in `pick_place_task.py`), not
  just the poses the generated set happened to cover.
- Compare against: the scripted policy's own success rate
  (`verify_pick_place.py` already does a PASS/FAIL version of this for the
  hand-coded policy) and the ~81% MimicGen generation success rate
  (`README.md`) as rough baselines — the trained policy existing to match or
  approach either would be the signal the whole pipeline worked.

## Checklist

- [ ] `trim_demo.py` — write, run on `demos/*.npz` + `demos/generated/*.npz`,
      sanity-check with `replay_demo.py` on the trimmed outputs.
- [ ] `export_lerobot.py` — write, run locally, producing
      `lerobot_dataset/so101-pick-place/`.
- [ ] Dataset QA pass (Phase 2).
- [ ] ACT baseline run (local or Iridis) — confirms the pipeline before the
      real model.
- [ ] `rsync` dataset + repo to Iridis X; SmolVLA training job via `sbatch`.
- [ ] `eval_policy.py` — write, run ≥ 50 episodes, report success rate against
      the scripted-policy and MimicGen-generation baselines.
