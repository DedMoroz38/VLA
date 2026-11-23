# Sim-to-real camera matching

Goal: the sim cameras see what the real rig's cameras see. Same view, same
robot and table colours, same image size. The background (floor, skybox) is
deliberately left as it was.

Result, in `mujoco_menagerie/robotstudio_so101/`:
- `pick_place_scene_real.xml`: the real look, with a 40 mm cube as the only
  object on the table.
- `pick_place_scene_real_eraser.xml`: the same look with the original eraser
  and tray. It uses the same physics as `pick_place_scene.xml`: recorded demos
  replay bit-exactly in it, so any dataset can be re-rendered with the real look:

```bash
python export_lerobot.py "demos/scripted_trimmed/*.npz" \
    --scene mujoco_menagerie/robotstudio_so101/pick_place_scene_real_eraser.xml   # 640x480 is the default
python eval_policy.py ... --scene mujoco_menagerie/robotstudio_so101/pick_place_scene_real_eraser.xml
```

Both real cameras should be run at **640x480**, matching the sim.

## The real cameras

| sim name | real camera | OpenCV index here | 640x480 mode | sim fovy |
| --- | --- | --- | --- | --- |
| `ext_cam` | Logitech C920 (side) | 0 | 4:3 centre crop of the 1080p frame | 43.3° |
| `wrist_cam` | icspring USB module (wrist) | 1 | 4:3 centre crop of the 1080p frame | 87.6° (fitted) |

Both cameras' 640x480 mode is a centre crop of the 16:9 frame, so they keep
the 1080p vertical field of view. For the C920 that is 43.3°, from its
published 70.4° x 43.3° at 1080p. This was checked from the positions of
static landmarks (carpet tiles, table edge) in both modes.

## What was changed, and how

**Robot colour.** In `so101_real.xml`, the printed parts are off-white
(`0.70 0.69 0.645`) instead of yellow, and the servos stay black. Under the
scene lighting the lit plastic renders at about RGB (161,159,148), against the
real camera's about (160,158,148).

**Table.** Matte neutral grey (`0.58 0.595 0.58`), which renders at about
(141,145,141) against the real (143,146,143). The table is now 1.2 m wide and
its back edge is at x = −0.071, about 4 cm behind the robot base (fitted, see
below). Its top is still at z=0 and covers the whole task area.

**Lighting.** The lab is lit by diffuse ceiling panels, so the sim light is
mostly ambient, with one soft overhead light. Shadows are faint, like the real
ones.

**Side camera (`ext_cam`).** It sits in front of the robot, facing it across the
table: pos (0.594, 0.005, 0.493), pitched 45° down, fovy 43.3. It was fitted
with the real arm as a calibration object (`capture_poses.py`, then
`fit_side_cam.py`):
1. The follower arm moved through 8 poses spread over the workspace, plus
   rest. At each, a side-cam frame and the measured joint angles were saved
   (`poses/`).
2. The per-pixel median of the 9 frames is the empty scene. The real arm in
   each frame is whatever differs from it.
3. An optimiser adjusted the camera pose, the table's back-edge position, and
   per-joint offsets between LeRobot and MuJoCo angles. It maximised the
   overlap between the real and sim arm silhouettes (mean IoU 0.64), while
   keeping the sim table edge on the real one (0.35 px rms).

The IoU is capped by the real masks: white jaws on the white table are partly
missed, and cables aren't in the model. `compare_all_poses.png` shows real /
sim / blend for every pose. The arm lines up in all of them.

**LeRobot ↔ MuJoCo joint offsets.** The same fit found that LeRobot's
calibrated angles and MuJoCo's differ:
`sim_rad = radians(lerobot_deg + offset)`, with offsets in
`side_cam_fit.json`. The two large ones are shoulder_lift ≈ +9° and
wrist_flex ≈ −20°. They are fitted jointly with the camera, so treat them as
±5°. Two independent signs support them: the real rest reading
(shoulder_lift −105°) is outside the sim's ±100° range without the offset,
and the shoulder_lift offset came out at +11° without the table-edge term
too. This matters for any real↔sim transfer of joint angles: actions from
the real arm need the offsets added before they mean the same pose in the
sim.

**Wrist camera (`wrist_cam`).** The jaws are rigidly attached to the camera, so
their outline in the image does not depend on the arm pose. `fit_wrist_cam.py`
segments the cream jaws in a real frame and optimises the camera's pose
(relative to the `camera_mount` body) and fovy, maximising mask overlap with a
sim segmentation render. Result: IoU 0.906 (`wrist_fit_overlay.png`). The fitted
lens is about 17 mm from the official menagerie position and much wider
(87.6° vs 61°).

## Files

| file | what it is |
| --- | --- |
| `capture_real.py` | Grab a 640x480 frame from each real camera (run with `/opt/anaconda3/bin/python`, which has OpenCV) |
| `live_camera.py` | Live view of a real camera at 640x480; S saves a frame |
| `capture_poses.py` | Move the real follower through `poses/plan.json` and save side-cam frames + measured joints (lerobot env) |
| `fit_side_cam.py` | side camera fit; writes `side_cam_fit.json` and `side_fit_overlay.png` |
| `compare_views.py` | real / sim / blend images for both cameras, plus colour readouts; `--pose P3` / `--all-poses` for the captured poses |
| `fit_wrist_cam.py` | wrist camera fit; writes `wrist_cam_fit.json` |
| `poses/` | captured frames, `joints.json`, `table_edge.json`, `background_median.png`, and the motion plan |
| `real/` | reference frames: `*_pair_1080` / `*_pair_480` were taken back to back to measure each mode's crop |
| `before_*.png`, `compare_*.png`, `demo_before_after.png` | before/after renders |

## Known gaps and how to close them

- **If the side camera or the arm is moved**, the fit has to be redone:
  `capture_poses.py --port <port>` (about 3 min of arm motion), then
  `fit_side_cam.py` (about 5–10 min).
- **Wrist lens distortion.** An ~88° lens has visible barrel distortion, and
  MuJoCo renders an ideal pinhole. A checkerboard calibration would measure it,
  and frames could then be undistorted (real) or distorted (sim) to match.
- **Jaw shading in the wrist view.** The real jaws look darker and warmer than
  the sim ones, because the room lights them from behind while the sim
  headlight lights them from the camera.
- **Cube colour and mass are placeholders** (red, 30 g). Set `cube_mat` and
  the geom `mass` to the real cube's.
- **Camera index order.** macOS can renumber the cameras when devices are
  plugged in or out, so check which index is which before recording.

## Touch test (joint conversion accuracy)

`touch_test.py` checks `joint_map.json` against the real arm with a ruler,
independently of any camera. It solves in the sim for poses that hold the fixed
jaw tip 15 mm straight above 6 table points, converts them with
`joint_units.py`, and moves the real follower there. You mark a dot under each
tip and measure its height, then each dot's position from the base's front line
(a ruler laid across the front of the base) and centreline. The script prints
the per-point error and fits corrections for the shoulder_pan, shoulder_lift,
elbow_flex and wrist_flex offsets. It can apply them to `live_overlay.json` and
`joint_map.json`.
wrist_roll isn't observable in these poses. Pass: errors within about 5 mm.

```bash
/opt/anaconda3/envs/lerobot/bin/python calib/touch_test.py --dry-run   # plan + sim checks, no arm
/opt/anaconda3/envs/lerobot/bin/python calib/touch_test.py             # follower connected
```
Every path is checked in the sim for table contact, and every target against
the real calibrated limits (minus 3°), before the arm moves. Moves are slow,
and on a tracking fault or Ctrl-C the arm holds its position with torque on.

## Wrist camera lens (checkerboard calibration, 2026-11-21)

`calibrate_intrinsics.py` (board → capture → calibrate) measured the real wrist
camera at 640x480, from 19 views (RMS 0.21 px; corner coverage 36%, edges
included). `intrinsics/wrist.json` has fx 445.8, fy 436.7 (fovy 57.6°, not the
87.6° of the old jaw-outline fit), an optical centre 23 px right of the image
centre, and distortion of about 12 px at the left and right edges.

- `so101_real.xml` `wrist_cam` carries these intrinsics (`resolution`,
  `focalpixel`, `principalpixel`, `sensorsize`). MuJoCo's `principalpixel` is
  (W/2 - cx, H/2 - cy), the opposite sign of OpenCV's offset; this was
  verified by projecting test markers.
- Distortion can't be modelled in MuJoCo. `sim/wrist_lens.py` (`WristLens.distort`)
  remaps the pinhole render, at any 4:3 size. `export_lerobot.py`,
  `eval_policy.py` and `live_overlay.py --cam wrist` apply it whenever the
  scene's `wrist_cam` has the calibrated intrinsics.
- Pose refitted with the lens fixed (`fit_wrist_pose.py`; real jaw mask from
  `real/wrist_640x480.png`). The lens lands 3.5 mm from where the Seeed
  bracket's CAD puts it (pos 1.2, 71.0, -3.8 mm in camera_mount), and the jaw
  outline matches a newer, independent frame too (`intrinsics/posefit/pose_fit_overlay.png`).
  Backups: `so101_real.xml.before_intrinsics`, `.before_wrist_pose`.
