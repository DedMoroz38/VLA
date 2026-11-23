# SO-101 pick-and-place simulation

MuJoCo simulation and demonstration pipeline for the SmolVLA project. Trains
against `../robot/lerobot`.

Environment: conda env `mujoco-so101` (mujoco 3.16).

## Scene

`mujoco_menagerie/robotstudio_so101/pick_place_scene.xml` — SO-101 arm on a
white table, a blue eraser to pick up, an open tray to drop it in, a wrist
camera and a fixed external camera ~45° above the front-left of the workspace.

Real-look scenes, matched to the lab rig (white robot, grey table, both
cameras matched to the lab's side and wrist cameras at 640x480; how, and how to
refine it: `calib/README.md`):

- `pick_place_scene_real.xml`: a 40 mm cube (`../models/cube40mm.stl`) is the
  only object on the table. The eraser/tray task scripts don't support it yet.
- `pick_place_scene_real_eraser.xml`: the same look with the original eraser
  and tray. The recorded demos use it, so pass it as `--scene` to
  `export_lerobot.py`, `eval_policy.py` or `replay_demo.py` to re-render them
  with the real look.

## Files

| file | what it is |
| --- | --- |
| `ik_solver.py` | Damped-least-squares velocity IK; position task first, gripper pitch resolved in its null space |
| `pick_place_task.py` | Scene path, success test, and the scripted reference policy (`PolicyRunner`) |
| `demo_io.py` | Demonstration file format — `DemoRecorder`, `load_demo` |
| `pick_place_teleop.py` | Keyboard teleoperation **with recording** |
| `record_scripted_demo.py` | Headless demo recording via the scripted policy |
| `replay_demo.py` | Replay a demo and verify it was recorded correctly |
| `mimicgen.py` | MimicGen-style augmentation: segment a demo, retarget it to a new eraser pose |
| `generate_demos.py` | Generate many episodes from the recorded ones |
| `verify_pick_place.py` | Headless PASS/FAIL check that the scripted policy still solves the task |
| `view_so101.py` | Plain scene viewer |
| `calib/` | Sim-to-real camera matching: capture, compare, wrist-camera fit |
| `joint_units.py` | Sim radians <-> real-arm LeRobot units (`calib/joint_map.json`), used by export, eval and `calib/live_overlay.py` |
| `demos/` | Recorded episodes (`.npz`) |
| `demos/generated/` | Automatically generated episodes |

## Step 2 — record and replay a demonstration

Record one by hand:

```bash
python pick_place_teleop.py
```

`W/S` `A/D` `R/F` jog the gripper in X/Y/Z, `T/G` pitch, `Q/E` roll, `C/V`
close/open the gripper. Press **P** to start recording, solve the task, then
press **P** again to stop and save into `demos/`. **X** discards a take in
progress. The status line shows frame count and whether the task is currently
solved.

To collect a set covering different eraser positions and rotations:

```bash
python pick_place_teleop.py --randomize             # random start pose
python pick_place_teleop.py --randomize --seed 7    # repeatable sequence
python pick_place_teleop.py --eraser 0.22 0.05 45   # explicit: x y yaw_degrees
```

**N** draws a new random pose; **Space** resets the arm but keeps the current
one, so a botched take can be retried on the same configuration. The start pose
is written to the episode's metadata as `eraser_start`.

Poses are sampled from `ERASER_SAMPLE_X/Y/YAW` in `pick_place_task.py` — the
region where the gripper can actually reach a straight-down grasp, measured by
sweeping the IK over the tabletop (see `demo_collection_notes.md`). Outside it
`wrist_flex` saturates at its ±95° limit and the descent stalls centimetres
short. `--eraser` warns if you hand it a pose outside that region or on top of
the tray, but still runs.

Record one without a human, for testing or as a fallback source trajectory:

```bash
python record_scripted_demo.py            # -> demos/scripted_000.npz
```

Then verify it replays:

```bash
python replay_demo.py "demos/*.npz"                    # headless check
python replay_demo.py demos/scripted_000.npz --view    # watch it
python replay_demo.py demos/scripted_000.npz --mode state --view
```

### What is recorded

One frame per control step (30 ms, 33.3 Hz), as a compressed `.npz`:

| array | contents |
| --- | --- |
| `time` | sim time |
| `action` | the 6 commanded joint position targets — the action a policy must predict |
| `joint_pos`, `joint_vel` | arm joint states |
| `ee_pos`, `ee_quat` | gripper site pose, world frame |
| `obj_pos`, `obj_quat` | eraser pose, world frame |
| `qpos`, `qvel` | full generalised state |
| `mj_state` | complete `mjSTATE_INTEGRATION` state |

Plus a JSON metadata blob: task string, success flag, source (`teleop` /
`scripted`), scene path, control rate, joint and camera names.

Frame `k` is the state the recorded `action[k]` was issued *from*, so replay is
a plain loop: restore `mj_state[0]`, then set `ctrl = action[k]` and step
`control_substeps` times.

Camera images are deliberately **not** stored. `mj_state` is a full integration
state, so any view — either camera, any resolution, or a camera that didn't
exist at record time — can be re-rendered exactly from the file. That is how
the LeRobot export in step 5 will get its frames, and it keeps episodes at
~0.7 MB instead of hundreds.

### Replay modes

- `--mode action` (default) restores only the *first* state and re-drives the
  sim from the recorded actions alone. This is the real check: it proves the
  actions reproduce the demonstration, which is what a trained policy will have
  to do. Currently bit-exact (divergence ~1e-16, floating-point noise only).
- `--mode state` sets each recorded state directly. Verifies the stored states
  are self-consistent with the stored poses; use it with `--view` to just watch
  what was recorded.

## Step 3 — generate demonstrations automatically

```bash
python generate_demos.py -n 200                     # -> demos/generated/
python generate_demos.py -n 20 --keep-failures      # keep the misses too
python replay_demo.py "demos/generated/*.npz"       # they verify like any demo
```

Each attempt samples a new eraser pose, picks a source demo round-robin,
retargets it, and executes the result. Only successes are kept unless
`--keep-failures`. Generated episodes use the same `.npz` format as recorded
ones, with `source="mimicgen"` and the source demo recorded in the metadata.

**Around 80% of attempts succeed**, so ~250 attempts yields ~200 usable
episodes at roughly 1.2 s each.

### How the retargeting works

A demo is split automatically at its gripper events into two object-centric
phases:

| phase | frames | referenced to |
| --- | --- | --- |
| A | reach, grasp, lift | the **eraser** — rotated and translated to the new pose |
| B | carry, release, retreat | the **tray** — unchanged, it hasn't moved |

The cut sits at the top of the lift, not at the grasp, so the phases meet in
free space at hover height; joining them at grasp height would drag the eraser
across the table. Straight-line bridges connect the arm's actual start pose to
phase A, and phase A's moved endpoint to phase B's fixed start.

Three things this needed that are easy to get wrong, each found by the success
rate refusing to budge:

- **Lead-in.** Rotating phase A about the eraser moves its *first* waypoint
  too, so the arm starts the episode displaced from the trajectory it is
  tracking. Without a bridge it chases the reference all the way down and swats
  the eraser off the table. 15% success.
- **Converged IK per waypoint.** Taking one velocity-IK step per waypoint is a
  proportional chase of a moving reference and lags it by 22–29 mm — against a
  30 mm-wide eraser. Solving each waypoint to convergence on a scratch `MjData`
  took it to under 1 mm. 15% → 60%.
- **Pan-corrected roll.** With the gripper straight down the jaw axis sits at
  world azimuth `wrist_roll - shoulder_pan + const`. An eraser at a new position
  is reached at a different pan, so replaying the source's roll puts the jaws in
  the right place rotated wrong, and they close beside the block. Re-deriving
  roll from the commanded pan each step fixed it. 60% → 81%.

### Scope

This works because the grasps are top-down: with the gripper pointing along
world −Z, `wrist_roll` *is* rotation about world Z, so an eraser yaw change is
absorbed exactly by a roll offset and the transform stays planar. The SO-101's
five arm joints cannot track arbitrary 6-DOF poses, so tilted approaches would
need both real SE(3) retargeting and a 6-DOF arm.

Yaw deltas are wrapped into ±90°, since the eraser's square cross-section makes
yaw and yaw+180° identical. The wrap also keeps the roll offset inside
`wrist_roll`'s limits — without it a −80° source and a +80° target would ask for
160° of extra roll and saturate the joint.

Source demos that cannot be segmented are reported and skipped. A regrasp gives
four gripper transitions instead of two, so there is no single grasp event to
anchor phase A — `teleop_009` is excluded for this reason.

## Cube baseline: demos on fixed poses (2026-11-21)

The real-look cube scene (`pick_place_scene_real.xml`: 40 mm black cube, the real
transparent container with its green tape, calibrated cameras and lens) with a fixed
pose set, so sim and real runs are measured on the same start poses.

1. Poses: `eval/cube_poses.json` is now v4. v3 (`make_cube_poses_v3.py`) added 50 train poses
   much further out (x 0.24-0.33, y 0-0.20, reachable with a tilted gripper; one at 38.5 cm was
   then dropped as unreachable by hand). v4 (`make_cube_poses_v4.py`) added 10 far test poses
   (ids 500-509), so `--split test` = 20 near + 10 far, and marks pose 251's demo as dropped
   (teleop_071, arm caught on its own shoulder). Final data: 148 demos in `demos/cube/`.
   Earlier: v2 (`make_cube_poses_v2.py`): the 20 train poses
   already demonstrated are kept, and the other 80 train, the 20 test and the 100 sim_test poses
   are further from the robot (x 0.16-0.26, y 0.08-0.18, IK-checked reachable), as the
   near-base poses were hard to teleoperate. v1 is kept as `eval/cube_poses_v1.json`; its
   description follows (don't re-run `make_cube_poses.py` -- it would overwrite v2 with v1):
   `python make_cube_poses.py` -> `eval/cube_poses.json` + `eval/cube_poses.png`.
   120 Latin-hypercube poses (x 0.12-0.22, y 0.08-0.18 = at least 8 cm to the robot's left, yaw -45..45 deg; the cube
   repeats every 90 deg): 100 `train`, 20 `test` (a well-spread random subset), plus
   100 `sim_test` for larger sim-only evals.
2. Record, driven by the real leader arm (lerobot env, leader plugged in):

       /opt/anaconda3/envs/lerobot/bin/python pick_place_teleop.py --poses eval/cube_poses.json --leader

   After every reset, bring the leader to the start pose shown in the status line
   (every joint within 15 deg) and the sim arm starts following; then P to record, put
   the cube in the container, P to save. A successful demo moves on to the next pose;
   a failed one stays (Space retries). Restarting resumes at the first pose without a
   successful demo. N / B skip forward / back. Demos go to `demos/cube/`, each with its
   `pose_id` and the task "Pick and place the black cube into the container".
   Without `--leader` the same works with the keyboard.
3. Trim and export:

       python trim_demo.py "demos/cube/teleop_*.npz" --out-teleop demos/cube/trimmed
       python export_lerobot.py "demos/cube/trimmed/*.npz" --repo-id local/so101-cube --root lerobot_dataset/so101-cube

4. Evaluate (the scene and training demos come from the pose file):

       python eval_policy.py CKPT --poses eval/cube_poses.json --split test        # the 20 held out
       python eval_policy.py CKPT --poses eval/cube_poses.json --split sim_test    # 100 more, sim only

   On the real robot, put the cube on a test pose with
   `calib/live_overlay.py --cube X Y YAW_DEG` (values from the pose file).

5. Run it on the real arm (`run_policy_real.py`: follower + both cameras; same test poses,
   the cube's place drawn on the live side view):

       python run_policy_real.py CKPT --split test --shadow   # dry run: arm holds 'ready'; the
                                                               # policy's plan from real (red) vs sim (green) images
       python run_policy_real.py CKPT --split test            # live; mark each outcome with s / 1 / 2 / 3 (x = abort)

   Commands are clipped to `--max-step` (10 deg per step) and the sim's joint limits.
   Results, videos and per-step joints go to `eval/real/<run>/` and resume where they stopped.

Real-camera settings at deployment, so the policy sees what it was trained on: cameras
named `ext_cam` and `wrist_cam`; the wrist image rotated 90 deg anticlockwise
(`wrist_lens.orient_wrist`; in LeRobot `OpenCVCameraConfig(width=480, height=640,
rotation=Cv2Rotation.ROTATE_270)`); ext_cam 640x480; the control loop at 33 fps (the
sim's 30 ms step). The episode starts from the cube scene's 'ready' keyframe (the arm's
rest pose with small margins). `calib/live_overlay.py --cam wrist` still shows the raw,
unrotated wrist image (it is a calibration tool).

## Joint units (sim <-> real)

`export_lerobot.py` writes `observation.state` and `action` in the **real
arm's units** by default, so a policy trained on sim data can drive the real
SO-101 without conversion, and real `lerobot-record` data can be mixed in:

- 5 arm joints in LeRobot degrees (0 = the calibration pose) and the gripper in
  0..100 (0 = closed), named `shoulder_pan.pos` ... `gripper.pos`
- arm: `sim_rad = radians(real_deg + offset)`; gripper:
  `sim_rad = -0.1745 + radians(pct/100 * span + offset)`, where span is the
  follower's calibrated gripper range in degrees
- offsets and span live in `calib/joint_map.json`, which is copied into each
  exported dataset. `calib/live_overlay.py` keeps it up to date while you tune;
  after re-calibrating the arms, re-tune and re-export.

`--joint-units sim` gives the old datasets (MuJoCo radians). `eval_policy.py`
detects a checkpoint's units from its stored normalisation stats, so old and
new checkpoints both evaluate correctly; override with `--joint-units`.

## Notes

- Rendering costs ~16 ms per camera versus ~0.05 ms for a control step of
  physics, so the teleop window refreshes every other control step
  (`--render-every`). The default keeps the sim at ~1.0x real time with a 16 fps
  display; `--render-every 1` is smoother to watch but runs at ~0.8x.
- MuJoCo's `mj_step` integrates `qpos` but leaves `xpos`/`site_xpos`/`cvel`
  describing the *pre-integration* state, so both the control loop and the
  replayer call `mj_forward` after stepping. Without it, recorded
  end-effector and object poses sit one physics step behind the `qpos`
  recorded beside them.
- Tkinter + offscreen rendering is used instead of `mujoco.viewer` because
  GLFW/NSWindow is broken on this machine's macOS.
