# Progress

## 2026-11-18

**Summary:** Set up project instructions and a daily progress log. Sim-to-real camera matching done: new real-look scene `pick_place_scene_real.xml` (white robot, grey table, lab lighting, side and wrist cams matched to the real C920 and icspring cameras at 640x480). Physics unchanged, demos replay bit-exactly. Side cam then re-fitted using the real arm as a calibration object (9 poses): camera 7.5 cm higher than the photo estimate, plus measured LeRobot↔MuJoCo joint offsets.

- Created `CLAUDE.md` with the rule to log progress after each prompt, grouped by day.
- Created `PROGRESS.md` (this file).
- Sim-to-real camera matching, step 1: captured one 1920x1080 frame from each real camera (`sim/calib/real/cam0.png` = side cam, `cam1.png` = wrist cam) and rendered the current sim views (`sim/calib/before_ext_cam.png`, `before_wrist_cam.png`). Main differences: the real side cam faces the robot from across the table, not from the front-left; the robot is white, not yellow; the real wrist cam looks along both jaws from above them, with a wider FOV. Decided with user: both cameras at 640x480; sim arm pose doesn't matter, only where the base sits.
- Sim-to-real camera matching, step 2 (done):
  - Measured both cameras' 640x480 mode: a 4:3 centre crop of the 1080p frame. Side cam = C920, so fovy 43.3°.
  - Wrist cam fitted from the jaw outline (`sim/calib/fit_wrist_cam.py`): IoU 0.906, fovy 87.6°, about 17 mm from the official position.
  - Side cam placed from the table edge, base position and servo/PCB scale: (0.562, -0.012, 0.418), pitched 43° down. Checked by overlaying a matching sim arm pose.
  - Robot recoloured white, table matte grey, lighting mostly ambient with soft shadows. Table and robot pixel values within about 2% of the real side-cam frame.
  - New files: `so101_real.xml`, `pick_place_scene_real.xml` (originals untouched), `sim/calib/` (capture, compare and fit scripts, plus README with known gaps). `sim/README.md` updated. Installed scipy in `mujoco-so101`.
  - Verified that 7 demos replay bit-exactly in the new scene.
  - Next: tape-measure the side cam position, checkerboard-calibrate the wrist lens distortion, re-export the dataset with `--scene ...pick_place_scene_real.xml`.
- Launched the real-look scene in the teleop viewer (`pick_place_teleop.py --scene ...pick_place_scene_real.xml`) so the user could see it. No file changes.
- Added `sim/calib/live_camera.py` (live view of a real camera at 640x480, S saves a frame) and launched it for the side cam (index 0).
- Side-cam calibration with the real arm (follower on `/dev/tty.usbmodem5B790809571`, side cam = OpenCV index 0):
  - Fixed the `lerobot` conda env: its editable install pointed at the old `VLA project/robot/lerobot` path; updated the `.pth` (backup kept as `.pth.bak`).
  - Wrote `sim/calib/capture_poses.py`. It moved the arm through 8 sim-checked poses (`sim/calib/poses/plan.json`) plus rest, saving side-cam frames and measured joints to `sim/calib/poses/`. No aborts, tracking within 2.5°, arm returned to rest with torque released.
  - Wrote `sim/calib/fit_side_cam.py` (silhouette IoU over all 9 frames + the table edge measured in `poses/table_edge.json`). First fit without the edge: IoU 0.649, camera (0.590, 0.007, 0.514), pitch 46.6°, joint offsets lift +10.9°, wrist_flex -12.2° (saved in `side_cam_fit_fixedfovy.json`).
  - Combined fit (silhouettes + table edge) finished: camera (0.594, 0.005, 0.493), pitch 45°, roll ~0, mean IoU 0.643, table edge 0.35 px rms; table back edge at x=-0.071. Joint offsets (LeRobot deg → MuJoCo): shoulder_lift +9.4°, wrist_flex -19.9° (±5°), others ≈0. Saved in `sim/calib/side_cam_fit.json`.
  - Applied to `pick_place_scene_real.xml` (ext_cam pose, table moved back). `compare_views.py --all-poses` → `sim/calib/compare_all_poses.png` shows the arm aligned in all 9 poses. Demos still replay bit-exactly. Updated `sim/calib/README.md`. Stopped the free-FOV cross-check early (battery at 2%).
- Relaunched the teleop sim with the calibrated `pick_place_scene_real.xml` (closed the earlier window, which had the pre-fit camera loaded).
- Replaced the eraser and tray in `pick_place_scene_real.xml` with the 40 mm cube (`models/cube40mm.stl`, scaled from mm). Body `cube` with a free joint, at the eraser's old start (0.18, 0.14). Placeholder red colour and 30 g mass. Settles flat; keyframe updated.
  - Kept the previous version as `pick_place_scene_real_eraser.xml`, since the recorded demos and the task scripts (teleop, scripted policy, success test) are built around the `eraser` body and the tray. Those scripts don't support the cube scene yet.
  - Updated `sim/README.md` and `sim/calib/README.md`. `compare_views.py --all-poses` now works with or without a tray.
- Opened the cube scene in `view_so101.py` (teleop does not support the cube yet).
- Black cube in the real-look scene, with teleop support:
  - Cube is now black (`cube_mat` 0.08 grey). Fixed the cube's origin: the STL's origin is a corner, so the geom is shifted -20 mm to centre it on the body (before, the reported pose was off by 2 cm and resets dropped the cube).
  - `pick_place_task.py`: added `task_object()` (eraser or cube), `has_container()` and `object_rest_z()` (rest height from the scene XML). Tray checks are now optional. Eraser behaviour unchanged.
  - `pick_place_teleop.py`: uses the scene's object everywhere (pose, reset, recorder, status). Without a tray it shows "no tray" and saves demos without a success test.
  - Checked: `verify_pick_place.py` PASS, demos replay OK, cube rests at z=0.0199.
  - Found: the recorded demos store the scene path from before the project moved into `coding/`, so `replay_demo.py` needs `--scene` for them.
  - Launched teleop with the cube scene.

## 2026-11-19

**Summary:** Real-arm calibration of the sim. Both arms recalibrated by the user. A live real-vs-sim overlay tool was built and used to re-tune the side camera, table position, light and joint offsets. A new sim↔real joint conversion is now used by dataset export and eval. A ruler-based touch test verified the joints to about 0.5 mm after a shoulder_lift fix. The real kit's Seeed wrist-camera mount was found and added to the robot model; it lines up with the real one in the side camera to about 1–2 mm. Cameras are now opened by name.

**Final state at the end of the day**
- Joint offsets (`sim/calib/joint_map.json`, LeRobot deg → MuJoCo): shoulder_pan -0.72, shoulder_lift +0.63, elbow_flex -6.84, wrist_flex -2.44, wrist_roll +3.07, gripper 0. They match the 15:36–15:38 recalibration of both arms.
- `pick_place_scene_real.xml` (cube scene): side camera at (0.4647, 0.0027, 0.595), about 59° down, fovy 43.55. Table back edge at x = -0.021 (base overhangs about 1 cm; agrees with the measured 87 mm from table edge to stand). Shadow light at (-0.50, -0.44, 1.5), diffuse 0.17. Backup: `pick_place_scene_real.xml.before_live_overlay`.
- `so101_real.xml` (robot, used by both real-look scenes): Seeed wrist-camera mount, board and lens added as visuals only. Physics unchanged. Backup: `so101_real.xml.before_seeed_mount`. The `wrist_cam` viewpoint is still the Nov 18 jaw fit.
- Not yet updated: `pick_place_scene_real_eraser.xml` still has the Nov 18 camera, table and light.

**Hardware**
- Arms identified from their motor registers (read-only): leader `/dev/tty.usbmodem5B790804261`, follower `/dev/tty.usbmodem5B790809571`. The user then recalibrated both arms (`lerobot-calibrate`; official calibration video linked).
- Issues seen and handled:
  - The follower dropped off USB once (cable or hub).
  - Its motors once didn't answer because the 12 V supply was off.
  - One dropped status packet from the elbow servo crashed start-up. All 12 motors pinged healthy afterwards. Added `configure_with_retry()` (3 tries) to `live_overlay.py` and `touch_test.py`.

**Live overlay tool: `sim/calib/live_overlay.py`** (`lerobot` env)
- The leader drives the follower, and the follower's measured joints pose the sim. The sim camera view is drawn over the live real camera.
- View modes (`m`): outline, blend, real, sim, separate (sim in its own window).
- Keys:
  - camera: level moves `w s a d`, vertical `t b`, along the image's up direction `r f`, tilt/pan/roll `i k j l u o`, fovy `z x`
  - table back edge: `c v`
  - joint offsets: `1`–`6` select, `-`/`=` adjust
  - Tab switches the movement keys to the shadow light: position, beam, brightness, ambient, softness. A live real/sim grey-level readout keeps colours matched.
  - undo: Backspace; resets: `0` session, `F` original fit, `J` offsets only
- Values autosave to `live_overlay.json`. `X` writes them into the scene XML (one-time backup).
- `--cam wrist` overlays the wrist camera instead. It moves only the `wrist_cam` element, so the side view is unchanged (verified). `--cube X Y` places the sim cube where the real one is.
- Other options: `--no-leader` (follower limp, moved by hand) and `--snapshot` (one frame, no torque).

**Cameras by name: `sim/calib/cameras.py`**
- OpenCV camera indices on macOS change with what's plugged in: index 1 was once the FaceTime camera, and with the C920 unplugged, index 0 became the wrist camera. Cameras are now opened by name through PyAV ('C920' = side, 'icspring' = wrist). `python calib/cameras.py` lists them.
- Used by `live_overlay.py` and `capture_poses.py`.

**Sim↔real joint conversion: `sim/joint_units.py`**
- `JointMap` converts in both directions. Arm: `sim_rad = radians(real_deg + offset)`. Gripper: real 0–100 ↔ angle, using the follower's calibrated range (129.9°). Data lives in `sim/calib/joint_map.json`, inside the repo, so it also works on Iridis. `live_overlay.py` refreshes it on every save.
- `export_lerobot.py` writes real LeRobot units by default (`--joint-units real`; names `shoulder_pan.pos` … `gripper.pos`, like `lerobot-record`) and copies `joint_map.json` into the dataset. `--joint-units sim` keeps the old radians.
- `eval_policy.py` detects a checkpoint's units from its normalisation stats and converts, so old and new checkpoints both evaluate correctly.
- Verified: exact round trip; a 1-episode export matches the conversion (about 4e-6).
- Finding: demos hold wrist_flex at the sim limit, which equals the real limit (101.4° vs 101.3°). About 0.3% of scripted frames use shoulder_pan up to 110.7°, beyond the real ±105°.

**Joint accuracy: touch test, `sim/calib/touch_test.py`**
- How it works: the arm hovers the fixed jaw tip above table points, computed by sim IK and converted with the joint map. The user marks and measures each dot against the base's front line and centreline, plus the tip height. The script fits offset corrections.
- Safety: every path is sim-checked for table contact and joint limits. Moves are slow. On a lag fault or Ctrl-C the arm holds with torque on.
- `--reanalyze` corrects paper errors after the fact (flipped signs, edge gap, tilt). The fit uses a ±3° prior.
- Run 1 (6 points, 15 mm hover; `touch_test_20261119_200335.json`): after fixing one sign error, RMS 7.1 mm. Every point landed about 8 mm short; pan was fine.
- Overlay re-tune by the user: shoulder_lift and wrist_flex. Re-scored on run 1's measurements: RMS 4.1 mm, bias gone.
- Run 2 (5 centreline points, 5 mm hover, with heights; `touch_test_20261119_centreline.json`): the tip sank with reach. A single shoulder_lift correction of +1.2° explains it (RMS 3.7 → 0.5 mm) and was applied.
- Side camera after the fix: the user checked it in the overlay and found it fine.
  - Also prepared, not run: an automatic camera-only refit (`refit_camera.py`, tested on converted Nov 18 frames: IoU 0.608 → 0.652), and a capture plan converted to today's calibration through encoder ticks (`poses_20261119/plan.json`).

**Wrist-camera mount**
- The kit (Amazon B0GS4WQ1TM) is Seeed's SO-ARM101 Pro assembled kit with a 32×32 mm USB camera.
- None of TheRobotStudio's three 32×32 wrist mounts matches the real thin bent-strip bracket.
- Found Seeed's `SO-ARM101_CAMERA_MOUNT` (from `xiehuangbao888/soarm_soft_gripper`, already converted to the Menagerie gripper frame in `Fatsharkovo/so101-benchmark` PR #7). Files and notes are in `sim/calib/wrist_mount/`.
- Added to `so101_real.xml` as massless visuals; the old mount is hidden but kept. Mass, inertia and CoM are identical, and demos replay bit-exactly.
- Checked against the real arm in the side camera, without moving it: board and bracket match within about 1–2 px. Before, the old board hung visibly lower and to the left.
- The real lens is about 42 mm from the current `wrist_cam` viewpoint. The Nov 18 jaw fit probably made up for the wrong distance with a wide FOV (87.6°).

**Other**
- Confirmed the black STL cube scene (`pick_place_scene_real.xml`, from Nov 18) works with today's camera, table and light, and gave the commands to open it (`view_so101.py`, `pick_place_teleop.py --scene …`).

**Next**
- Move `wrist_cam` to the Seeed bracket's lens position and refit only its rotation and fovy against the real wrist view, with the cube in view.
- Copy the camera, table and light into `pick_place_scene_real_eraser.xml` before re-exporting the dataset.
- Re-export the dataset (now in real joint units).

## 2026-11-21

**Summary:** Rewrote the 2026-11-19 progress entry as a clear end-of-day summary. Added a copy of the real lab floor to the cube scene (photo + side camera, metric, colour- and brightness-matched; sim floor within about 6 grey levels of the real view). Calibrated the wrist camera lens, fixed the gripper passing through the cube, and added the real container (transparent, 55 mm, green tape outline on the table; placed, resized and colour-matched live against the real camera) to the cube scene with a working success test. Set up the cube baseline (fixed poses, leader-arm teleop, cube-aware task/eval), recorded 148 good demos (incl. 49 far poses), exported them to a LeRobot dataset (72,966 frames), and prepared + locally validated SmolVLA fine-tuning for Iridis (blocked on VPN).

- Restructured the 2026-11-19 section of `PROGRESS.md` into an accurate summary: final state, hardware, overlay tool, cameras, joint conversion, touch test, wrist mount, next steps. It replaces the earlier chronological, prompt-by-prompt bullets, which had an out-of-date summary line. 2026-11-18 is untouched. The previous version is in a scratchpad backup.
- Real lab floor in the cube scene (user away; agreed beforehand: hybrid photo + camera texture, table top 72 cm above the floor, cube scene only, use `photos/floor.jpeg` for the floor hidden behind the arm):
  - New `sim/calib/floor/` (`build_floor.py`, `floorlib.py`, README). The side camera's floor strip is traced to the floor plane. The phone photo is registered to it (SIFT + RANSAC homography, 79 inliers, 1.2 mm median), colour-mapped to the C920, and the C920's own pixels fill the strip's right end. Unseen floor is filled with 1 m blocks aligned to the measured 0.25 m tile grid, and the brightness is matched to the real frame in the sim render.
  - The 0.5 m tiles come out 0.5 m, which cross-checks the camera calibration and the 72 cm height.
  - `pick_place_scene_real.xml` (backup `.before_floor`): textured visual box at z=-0.72 (3.4 x 4.4 m, 2 mm/px texture `assets/floor_real.png`); floor plane moved -0.75 → -0.72 with the mean floor colour; table legs shortened; header comment updated. Reloads; cube still settles at z=0.0199.
  - Result: the sim side view's floor matches the real one to about 6 grey levels mean abs difference (`calib/floor/floor_compare.png`). The wrist view is unaffected (checked: no Seeed bracket geometry is in view, only its shadow).
- Fixed the wrist view cutting off the jaws (user report from the wrist overlay). The floor patch had raised the model's `stat.extent` from 1.87 to 5.57 m. MuJoCo scales near clip and shadow area with it, so the near clip went 19 → 56 mm, hiding the jaws, and shadows got about 3x coarser. `build_floor.py` now writes `<statistic center="0.472 0 -0.01" extent="1.8722"/>` (the size without the floor), and the scene was rebuilt. Near clip is back to 18.7 mm: at the user's current wrist_cam, the jaw pixels in a test view went from 17k (clipped) to 48k (complete). Floor match unchanged (about 6 grey levels).
- New `sim/calib/calibrate_intrinsics.py`: checkerboard calibration of a real camera's intrinsics and lens distortion at 640x480 (MuJoCo's camera is an ideal pinhole, and the wide wrist lens likely bends lines).
  - Steps: `board` writes a printable A4 PDF (`calib/intrinsics/checkerboard_A4.pdf`, 10x7 squares of 25 mm). `capture --cam wrist|side` shows a live view with the corners and a coverage grid, saving on SPACE or automatically when the board is held still in a new pose. `calibrate` runs OpenCV calibrateCamera (or `--model fisheye`), drops outlier views, and writes `calib/intrinsics/<cam>.json` (fx, fy, cx, cy, distortion, fovy, RMS) plus an undistort preview. Cameras are opened by name through `cameras.py`.
  - Validated on 22 synthetic views with a known wide, barrel-distorted, off-centre lens: fx/fy within 0.5 px, centre within 1.7 px, k1/k2 within 0.006, RMS 0.14 px. Capture not yet run on the real cameras. OpenCV's OpenCL is disabled before import (`OPENCV_OPENCL_RUNTIME=disabled`) to silence cache errors on this Mac, and only numbered frames are read.
- `calibrate_intrinsics.py capture` crashed on the first save: the coverage-grid colour was a numpy integer, which `cv2.rectangle` rejects. Cast to int and tested the drawing path on the saved view (54 corners found). The first real wrist view (`calib/intrinsics/wrist/frames/000.png`) was kept, and capture resumes from it.
- Wrist camera lens calibrated and applied to the sim:
  - Checkerboard calibration (19 real views, RMS 0.21 px): fx 445.8, fy 436.7 → fovy 57.6° (the old jaw-outline fit's 87.6° had traded a wrong distance for a wide FOV). Optical centre +23 px right. Distortion about 12 px at the left/right edges, 17 px in the corners. `calib/intrinsics/wrist.json`.
  - Fixed a capture crash first (numpy int colour in `cv2.rectangle`).
  - `so101_real.xml` `wrist_cam`: calibrated intrinsics instead of fovy (backup `.before_intrinsics`). Found by projection test that MuJoCo's `principalpixel` = (W/2 - cx, H/2 - cy), the opposite sign of OpenCV; the render then matches the pinhole to within 0.2 px.
  - New `sim/wrist_lens.py` (`WristLens`): applies the real lens distortion to pinhole renders at any 4:3 size. Used by `export_lerobot.py`, `eval_policy.py` and `live_overlay.py --cam wrist` whenever the scene's wrist_cam has calibrated intrinsics; old scenes are unchanged.
  - New `calib/fit_wrist_pose.py`: refits the wrist_cam pose with the lens fixed, using jaw-mask IoU and a small built-in Nelder–Mead. Fixed two bugs on the way: camera world poses need `mj_camlight` after `mj_kinematics`. Result: IoU 0.77, lens 3.5 mm from the Seeed bracket CAD estimate. It also matches a newer frame it wasn't fitted on. Written to `so101_real.xml` (backup `.before_wrist_pose`) and `live_overlay.json`.
  - `live_overlay.py` wrist mode: renders with the lens; `z`/`x` (fovy) disabled because the lens is measured; `X` now changes only pos/quat and keeps the intrinsics; the `F` reset reads fovy from the intrinsics.
  - Tested: lens detection per scene (real-look yes, old no), overlay render, `X` write on a temp copy, and a 1-episode export with the eraser scene.
- XML check: the user's wrist_cam tweak from the overlay (`live_overlay.json`, 14:17; never written with `X`) differed from `so101_real.xml` by about 5 mm (pos -0.00416, 0.07346, 0.00134; same rotation). Written into `so101_real.xml`, keeping the calibrated lens attributes.
- Fixed the gripper going through the cube (cube scene):
  - Diagnosis by simulated tests. Collisions were on, but the finger collision shapes are thin 2 mm plates and contacts are soft (5 ms step; gripper force-limited at 2.94 N·m). Squeezing sank the jaws about 4.8 mm into the cube, past the plates' thickness, which can tunnel through; pressing from above pushed the cube about 7.4 mm into the table. Just stiffening the cube made squeezing worse (25 mm, through).
  - Fix: explicit contact pairs in `pick_place_scene_real.xml` (backup `.before_cube_contacts`): cube vs every finger/wrist collision shape with margin 1.5 mm, solref 0.01, solimp 0.95/0.99, condim 4, friction 1.5; cube vs table_top stiffer; default cube–gripper/table contacts excluded. Margins 0–4 mm were compared, and 1.5 mm leaves the jaws about 0.35 mm from the cube surface while holding.
  - To reference them, 4 unnamed collision shapes in `so101_real.xml` got names (`gripper_wrist_box`, `fixed_jaw_mesh`, `moving_jaw_mesh0/1`; backup `.before_cube_contacts`). Names only, no physics change.
  - Result: squeeze overlap 4.8 → 0.67 mm max, press overlap 2.0 → 0.6 mm, cube into table 7.4 → 1.4 mm; the cube still grasps and lifts, and its rest height is unchanged. The eraser scene is unaffected: 3 demos replay bit-exactly and `verify_pick_place.py` passes.
- Cube physics checked by the user in the teleop sim (both camera views): works correctly. Advised taping the container only if the sim gets the same tape, since the policy trains on sim renders.
- Added the real placement container to the cube scene:
  - New `sim/container/build_container.py`: 220 x 130 mm outer, 17 mm corner radius, a green tape band (18 mm, placeholder colour) round the top of the walls. Height 60 mm, 2 mm walls/floor and the colours are guesses until measured; re-run with `--height` etc. Writes the visual meshes `assets/container_shell.obj` and `container_tape.obj` and patches `pick_place_scene_real.xml` between `container:begin/end` markers (backup `.before_container`). Collision uses boxes and cylinders (floor cross + corner discs, 4 walls, 4 segments per rounded corner). The cube–container contacts are stiff like cube–table.
  - Placed at (0.20, -0.15), long side along x, beside the robot opposite the cube's start area: fully in the side camera frame (u 92–250, v 197–418), and 17 mm clear of the cube's sampling region. Check render: `sim/container/check/side_cam.png`.
  - `pick_place_task.py`: new `container_xy(model)` / `container_tolerances(model)`. The success test and the scripted policy's drop point now come from the scene (`<numeric name="container_tol">`), with the old tray's constants as fallback (identical for the eraser scenes). The `pick_place_teleop.py` docstring was updated.
  - Tested: cube drops (flat, tilted, near the end and a corner, 15 cm drop) all settle on the container floor at z=22 mm with `object_in_container` True; a drop on the rim falls outside (False); the arm reaches the drop point at hover and release height (1.4 mm) without touching the walls. `verify_pick_place.py` passes and the eraser scenes' container values are unchanged.
- Container updated to the real one: 55 mm tall and transparent (tint 0.90/0.93/0.95, alpha 0.25, slightly glossy; `--alpha` to tune). The 18 mm green tape is now stuck to the tabletop round the container's base (flat band, 0.4 mm thick, inner edge at the outer wall; `--tape-gap` if the real tape sits further out), not on the walls. Rebuilt with `build_container.py` (new defaults). Still fully in the side camera frame (u 94–250, v 200–418). Drops still settle inside with the success test True (z range now 0–55 mm); the arm reaches the drop point (1.4 mm) without touching the walls. Note: MuJoCo casts a full shadow from the transparent shell (real clear plastic casts a fainter one).
- Tape outline now has square corners: four straight strips against the walls, so inner and outer edges are rectangles and the table shows in the small gap at each rounded container corner (as real tape laid in strips). `build_container.py` (`rounded_rect` handles a sharp-corner outline); rebuilt; check renders `sim/container/check/side_cam.png`, `top_view.png`.
- `live_overlay.py`: container placement aid. In the side view it now draws the sim container's base footprint (orange) and the tape band's outer and inner edges (red), projected from the scene so they stay exact under the transparent shell, in outline and blend modes. It also prints the container centre and the tape's outer corners in robot-frame mm (centre (200, -150); corners x 72..328, y -233..-67). New `--no-arm` runs it without the arms (sim arm in the 'ready' pose). Tested offline (sim render as the 'real' frame: lines land on the sim tape; the panel works without joints). Not run live yet: the C920 wasn't connected, so a background watcher launches `live_overlay.py --no-arm` once it appears.
- Moved the sim container at the user's request: half its width (65 mm) further to the robot's right (away from the robot) and one tape width (18 mm) forward, from (0.200, -0.150) to (0.218, -0.215); now the `build_container.py` default. It's still fully in the side camera frame (tape's nearest corner 18 px from the left edge). Drops still settle inside with the success test True; the arm reaches the drop point (1.2–1.8 mm) without touching the walls. Tape outer corners now at x 90..346 mm, y -298..-132 mm.
- `build_container.py`: container size is now two variables at the top, `LENGTH` and `WIDTH` (also `--length`/`--width`). `LENGTH_FROM` (front/back/centre) and `WIDTH_FROM` (outer/inner/centre) choose which side moves; the opposite side stays on the current placed footprint (`PLACED_CENTRE` (0.218, -0.215), 220x130). The tape and collision are built from the outline, so they follow. It prints the base and tape-edge coordinates, and the view check now covers the tape too. Tested: length 240 from the front (back edge and tape stay at x 108/90 mm), width 150 from outer (inner side stays at -150 mm; tape reaches the image edge, reported CUT OFF), width 150 inner + length 200 back. The default rebuild is byte-identical to the current scene and meshes.
- Live container resizing in `live_overlay.py`: Tab now cycles CAMERA → LIGHT → CONTAINER. In container mode, w/s/a/d pick the edge as seen in the side image (top = back end, bottom = front end, left = outer side, right = inner side), and x/z move it out/in by 1 mm (scaled by [ ]). The opposite edge stays fixed and the tape follows. Each step rebuilds via `build_container.build()` (meshes, collision, success test, scene XML), reloads the sim (~0.3 s) and saves the size into `build_container.py`'s `LENGTH`/`WIDTH`/`PLACED_*` (new `save_size()`, `default_args()`, `build()`). Backspace undoes resizes. The panel shows the size and edges in mm. Tested offline on backups (front +10 mm / outer -5 mm: correct edges moved, script variables updated, outlines follow), then restored the original 220x130 container.
- XML check after the user's live container resize: the only change in `pick_place_scene_real.xml` is the container block, now 203 x 131 mm at centre (0.2095, -0.2245) (base x 108..311, y -290..-159 mm), with `build_container.py`'s variables updated to match. Camera, table, light and contacts are unchanged. `so101_real.xml` (14:25) and the eraser scene are untouched. `live_overlay.json` and `joint_map.json` were re-saved on quit (18:29) with identical values. Nothing to undo. Resized container checked: fully in the side view (tape 11 px from the left edge), drops settle inside (success True), and the arm reaches the drop point (1.2–1.9 mm).
- Matched the container colours to the real side camera. New `sim/container/match_color.py`: averages 15 C920 frames (saved as `container/check/real_side.png`), renders ext_cam, and compares medians over the eroded sim tape pixels and container-inside pixels, then scales the material RGB until the render matches. Real tape RGB (49, 92, 67) vs sim (25, 154, 62) → tape colour 0.10/0.62/0.25 → 0.196/0.368/0.270. Container inside real (147, 150, 143) vs sim (168, 173, 173) → tint 0.90/0.93/0.95 → 0.714/0.727/0.673 (alpha 0.25 kept). Both now render to the real values exactly. `build_container.py` gained `TAPE_COLOR`/`CONTAINER_COLOR`/`CONTAINER_ALPHA` variables and `save_colors()`; scene rebuilt (size/placement unchanged: 203x131 at (0.2095, -0.2245)); XML comments no longer say placeholder. Comparison: `container/check/color_match.png`.
- Readiness review for the cube baseline demos: the scene was ready, but the pipeline still assumed the eraser (task text, eval object, no fixed pose list, eraser yaw range). Recommended a stratified 120-pose set with a spread-out 20-pose test split plus extra sim-only test poses, and leader-arm teleop in the sim.
- Cube baseline pipeline (user: "do all of it including leader arm teleop"; task text agreed as "Pick and place the black cube into the container"):
  - `demo_io.py`: `TASKS` / `task_for(object)`. Each episode's task now follows the scene object; eraser demos keep the old text.
  - `pick_place_task.py`: `SAMPLE_YAW` per object (cube ±45°, its 90° symmetry); `sample_eraser_pose(..., object_body=)`.
  - `eval_policy.py`: works on the cube scene. Object from the scene (place/check/track, task text to the policy), yaw distance modulo the object's symmetry, `--split` to filter a pose file, scene and train demos defaulting from the pose file, yaw split at the middle of the range in the report.
  - New `make_cube_poses.py` → `eval/cube_poses.json` + `eval/cube_poses.png`. 120 Latin-hypercube poses over x 0.14–0.26, y −0.04–0.16, yaw ±45° (all ≥10 mm clear of the container tape), 20 `test` chosen as the most spread of 5000 random splits (closest pair 0.227 vs 0.106 typical), 100 `train` in random order, plus 100 independent `sim_test`. Seed 20261121.
  - `pick_place_teleop.py`:
    - `--poses FILE [--split train]` steps through the poses. A successful save marks the pose done and advances; a failure stays. Resumes from saved demos; N/B go next/previous.
    - Demos go to `demos/cube/` and store `pose_id`/`pose_split`/`pose_file`/`control`.
    - `--leader [PORT]` drives the sim arm from the real leader arm via `joint_units` (lerobot env). After each reset it waits until every arm joint is within 10° of the start pose (the status shows the offsets), then follows at ≤2.5 rad/s. P is refused until then, and the keyboard motion keys are off.
    - `--scene` defaults from the pose file.
  - Tested:
    - teleop with a fake leader: no engage when 30° off, P refused; engages within tolerance; follow step capped at 4.3°/tick and reaches the target; a forced success saves the right metadata and task, advances to the next pose, and the resume scan finds it; B and the status line work;
    - eval `run_episode` on a cube test pose with a dummy policy: the policy receives the cube task, and the result is correct;
    - `eval_policy.py --help`; `verify_pick_place.py` (eraser) PASS.
  - `sim/README.md`: new "Cube baseline" section (poses → record → trim → export → eval → real placement).
- Teleop display, from the user's first recording session:
  - `pick_place_teleop.py` now opens full screen: a large operator view from behind the robot (base in the middle, container on the right; free camera az 0°, el −50°, 0.8 m; arrows orbit, -/= zoom), plus the two recorded cameras (wrist_cam, ext_cam) stacked on the right. `--windowed` gives the old layout.
  - Display only: demos store the sim state, and dataset images are still rendered from wrist_cam/ext_cam at export.
  - Performance: one shared renderer (a second GL context cost ~10 ms per switch), and the small views refresh alternately. The full frame went 94 → 39 ms, inside the 60 ms budget (render every 2nd 30 ms step), so the sim stays real-time.
  - Fixed the recording message that said 'eraser' (now the scene object).
- Start pose → the arm's real rest pose (user request; agreed it's right: the real robot starts from rest, so sim demos, sim eval and real runs would share one start). A watcher reads the leader's rest pose once the user closes the running teleop; it then gets written as the cube scene's 'ready' keyframe, with clearance margins. Pending.
- Start pose = real rest, written (`pick_place_scene_real.xml` 'ready' keyframe, backup `.before_rest_start`).
  - Leader at rest read: pan -1.1, lift -103.4, elbow 92.3, wrist_flex 65.2, roll 6.0, gripper 0.4 (steady).
  - Exact rest isn't usable in the sim. Shoulder_lift is 3° past the sim joint limit, so it is set 2° inside. And at rest the jaws sit over the cube area (site at x 195 mm, z 48 mm), overlapping some poses by 12 mm, which would be true on the real table too.
  - A search over small raises found elbow and wrist_flex each 6° less folded. Result: ≥24 mm jaw clearance to the cube at all 220 poses and 24 mm to the table. Start in leader units: -1.1/-98.6/86.3/59.2/6.0/0.4; every joint within 10° of rest, so a leader at rest engages immediately (tested with a fake leader). It holds with no drift.
  - This start is used by teleop, eval (`reset_to_ready`) and everything else that uses the cube scene's keyframe. For real runs, start the follower from this pose too (rest with the elbow/wrist slightly unfolded).
- The user's first full-screen leader session ran at 0.76x real time. Demos recorded that way would play back about 30% faster than the hand motion, because each sim step reads the leader once. Cause: the leader's USB read (about 10 ms) ran on every 30 ms step, on top of the 39 ms render every 2nd step. Fix: `Leader` in `pick_place_teleop.py` now reads the bus in a background thread (newest reading kept under a lock; glitches keep the last one), so the sim loop never waits on USB. Not yet measured live: the user restarts and checks the status line shows about 1.00x (fallback: `--render-every 3`).
- Recording session feedback:
  - Moved the user's 2 cube recordings (both failed, pose 0: 146 and 1068 frames) out of `demos/cube/` to `demos/cube_removed/` (recoverable).
  - Operator view changed to the container side, 45° down, looking across at the robot (free camera az 90°, el −45°, 0.7 m, lookat (0.13, −0.06, 0.03)).
- Wrist image rotated 90° anticlockwise everywhere (user's choice from 4 options; jaws on the right):
  - New `wrist_lens.orient_wrist()` (`WRIST_ROTATION`), applied after the lens distortion in `export_lerobot.py` (wrist feature now 640x480 HxW portrait, only for real-look scenes with the calibrated lens, so eraser datasets are unchanged), `eval_policy.py` (policy input; the eval video scales the portrait wrist view to the side view's height) and the teleop display.
  - The teleop now shows the wrist view exactly as the dataset gets it (lens distortion + rotation; distorted at tile size, 43 ms/frame). `wrist_lens.py` imports OpenCV lazily, so the keyboard teleop still runs in the mujoco-so101 env (rotation only there).
  - Real deployment must match: LeRobot `OpenCVCameraConfig(width=480, height=640, rotation=Cv2Rotation.ROTATE_270)` (checked: LeRobot maps ROTATE_270 to cv2.ROTATE_90_COUNTERCLOCKWISE, and width/height are after rotation). README updated with all real-camera settings (names, rotation, 33 fps, start pose).
  - Tested: export of a removed demo (wrist (3, 640, 480), ext (3, 480, 640), cube task, real units), the exported wrist frame looks right, eval `run_episode` with video (480x1000), fullscreen layout in both envs, `verify_pick_place.py` PASS.
- Cube start area moved at least 8 cm to the robot's left, away from the resting gripper (user: nothing too close to the gripper, at least ~8 cm left of the robot). An IK sweep of straight-down grasps (cube height) over x 0.12–0.28, y 0.06–0.24 found reach within 3 mm up to about 28 cm from the base. The largest fully reachable box with y ≥ 0.08 is x 0.12–0.22, y 0.08–0.18 (10x10 cm, same area as before). New `CUBE_SAMPLE_X/Y` and `SAMPLE_REGION` in `pick_place_task.py` (eraser region unchanged); `make_cube_poses.py` uses them. `eval/cube_poses.json` + png regenerated (same seed and split structure; no demos depended on the old poses). Checked over all 220: top-down reach max 2.1 mm; resting arm ≥42 mm from every cube (was 24); cube centres ≥75 mm from the resting gripper; all in the side camera frame (u 371–538, v 203–338). README updated.
- Recording fixes (user: scene doesn't reset after stopping; failed trials pile up; how to start from 0):
  - Moved all 4 cube demos (2 successful, 2 failed; poses 0–2, two of them recorded before the poses were regenerated) to `demos/cube_removed/20261121_193725/`. `demos/cube/` is empty, so recording starts at pose #1.
  - `pick_place_teleop.py`: stopping a recording (P) now always resets. A success advances to the next undone pose; a failure resets to the same pose. Failed takes are saved to `demos/cube/failed/`, so `demos/cube/` holds only good demos.
  - Resume now counts a pose as done only if the saved demo's start matches that pose's x/y within 2 mm, so a regenerated pose file can't mark the wrong poses done.
  - Tested with a fake leader (fail → `failed/`, reset to the same pose; success → next pose; restart resumes at #2).
- Checked the first 5 cube demos (`demos/cube/teleop_000–004`, all successful, poses 0–4; plus 1 failed take for pose 5):
  - Metadata: task, pose id, start position matching the pose file (0.0 mm), leader control, steady 30 ms steps, 15–20 s each; cube lifted to 116–145 mm.
  - All replay exactly (`replay_demo.py`: 5x REPLAY OK, 0 error).
  - Trimmed in scratch (tails of 0–4.1 s removed).
  - Command-to-actual position gap ≤6° except one 15° moment in 004 (shoulder_lift while carrying the cube over the container wall). The 2.5 rad/s speed cap was hit on only 1.7–3% of steps.
  - 004 was stopped only 9 frames (0.27 s) after the success test settled (it flickered while the gripper still touched the cube). Advised waiting about 1 s after release.
- 'Arm doesn't react after stopping': not the save (14 ms). After the reset the sim waits for the leader at the start pose, and the leader's real rest was 8–18° off it (wrist_flex about +17°, pan varies ±8°). That exceeded the 10° engage tolerance, so the sim waited until the operator moved. Fix:
  - Start keyframe set to the median first frame of the 6 takes (pan 0, lift -1.745 = sim limit, elbow 1.4329, wrist_flex 1.2854, roll 0.0955, gripper closed). Clear of the table by 24 mm and of all 220 cube poses by ≥22 mm; holds with no drift.
  - Engage tolerance 10° → 15°. Recorded demos are unaffected: they store full state.
  - So eval now starts where the demos actually start.
- Export check of the first 5 (trimmed, in scratch): 5 episodes, 2451 frames, 33 fps; ext_cam 480x640, wrist_cam 640x480 (rotated, lens-distorted). Review video `sim/demos/cube_review_first5.mp4` (side view + wrist, per-episode labels). Frames look right: grasp, carry, release into the container. Episodes 0–3 end right after the release (trim at settle + 0.3 s). Episode 4's success test flickered, so it kept the return to rest: an inconsistent ending, minor.
- Cube poses v2 (user: keep the current 20, move the remaining 80 away from the robot, towards the side camera; near-base poses were hard to grab by hand). New `sim/make_cube_poses_v2.py` → `eval/cube_poses.json` (v1 backed up as `eval/cube_poses_v1.json/.png`):
  - Kept the 20 demonstrated train poses (same ids and coordinates, first in order).
  - The new 80 train + 20 test come from a scrambled Halton sequence over x 0.16–0.26, y 0.08–0.18, yaw ±45°, each IK-checked (straight-down grasp within 3 mm; 8 far-left candidates rejected). The test set is the most spread of 5000 splits (0.231 vs 0.117 typical); ids 200+/300+.
  - 100 new sim_test in the same region (ids 1000+).
  - Checked: the teleop resume counts the 20 done and continues at #21 (id 200); new poses ≥26 mm from the resting arm and all in the side view (u 373–541, v 235–378).
  - Test moved to the new region too, since that's where most training data will be. README updated. The user's running teleop session must be restarted to load v2.
- All 100 cube demos checked (`demos/cube/teleop_000–099`; 27 failed takes in `failed/`):
  - Data: 100/100 successful; exactly one per train pose (100 of 100, no duplicates or missing); each starts exactly on its pose; cube task text; leader control; 12.6–29.7 s (mean 18.2); steady 30 ms steps.
  - Replay: all 100 exact (`replay_demo.py`).
  - Tracking gap: median 5°. Over 10° in 8 demos. Biggest: 071 (53° elbow for 1.5 s at the start: the fixed jaw caught on the shoulder while unfolding from rest), 078/081 (~20°, 0.36 s, same cause), 004 (cube against the container wall while carrying).
  - The jaw slides along the shoulder at the start in 82/100 demos (mostly harmless). Speed cap hit on a median 3% of steps (max 7%).
- Trimming found 18–19 demos stopped before the success test settled. Cause: the at-rest check (angular speed < 0.01 rad/s, a quirk of the original test) vs a resting cube's invisible micro-rocking on the container's overlapping floor pieces (0.005–0.018 rad/s).
  - Tried and reverted, to keep the physics identical to the recording (exact replay): torsional friction (condim 4) and a single-box container floor. Extending old-physics states under new physics made the cube rock up to 0.04 rad/s. The scene is back to the recorded physics (backup `.before_torsional` restored and rebuilt; the only diff is the new numeric).
  - Fix: `<numeric name="container_rest" data="0.005 0.05">` (written by `build_container.py`). `object_in_container` uses linear < 5 mm/s and angular < 0.05 rad/s when the scene has it; the eraser scenes keep the original test (`verify_pick_place.py` PASS).
  - `trim_demo.py` gained an `extend_until_settled` safety net (hold the last action up to 3 s if an episode stops before settling; `--no-extend` to disable). It is not needed now: 0 frames simulated.
  - Result: `demos/cube/trimmed/`: 100 episodes, 51,239 frames (25.6 min); tails of 0.75–4.86 s removed; every episode ends with ≥10 settled success frames.
- 50 extra train poses much further from the robot (user: many poses felt too close and were hard to grab). New `sim/make_cube_poses_v3.py` appends them to `eval/cube_poses.json` (v2 backed up as `eval/cube_poses_v2.json/.png`):
  - Region x 0.24–0.33, y 0.00–0.20, ids 400–449, after the 100 done; scrambled Halton over x/y/yaw.
  - Reach check with free gripper pitch (as when grabbing by hand). A straight-down grasp only reaches ~27 cm; tilted (~42°) the arm reaches beyond 36 cm at cube height within 3 mm, so the side camera (bottom edge ~37 cm) is the limit.
  - Each pose has the whole cube in the side view with a 15 px margin, ≥20 mm from the resting arm, and is clear of the container tape. No candidates were rejected.
  - Teleop continues at #101 (id 400).
  - The test set (20) and sim_test (100) are unchanged (x 0.16–0.26), so the far region has no test poses yet (offered).
  - `draw_map` legend now shows real counts.
- Dropped pose #128 (id 427, x 0.330 y 0.198, 38.5 cm from the base; user: too far, unreachable) from `eval/cube_poses.json`: split 'dropped' with a reason (backup `eval/cube_poses_v3_before_drop.json`). 127 demos done; the far demos already reached 36.0 cm and all 22 remaining poses are ≤35.0 cm. Train is now 149; the teleop continues at #128/149 (id 428) after a restart.
- All 149 cube demos checked:
  - 150 files, all successful, every start on its pose, cube task text, leader control; 29 failed takes in `failed/`.
  - All 149 train poses covered; pose 449 (the last) was recorded twice, 21 s apart, because the teleop stays on the final pose once everything is done. The later take (`teleop_149.npz`) moved to `demos/cube_removed/duplicate_pose449/`.
  - All 150 replay exactly.
  - Far set (49): mean 16.0 s (11.4–20.4), tracking gap median 5°, no gap >10° lasting >0.5 s, speed cap median 3.1%. The first 100 are unchanged from the earlier check (071 still the only bad one: 53° for 1.5 s).
  - Re-trimmed: `demos/cube/trimmed/` has 149 episodes, 73,466 frames (36.7 min). Every episode ends with ≥10 settled success frames; only `teleop_111` needed the extension (10 frames = 0.3 s, last action held, same physics as recorded).
  - Open: whether to drop or re-record 071; whether to add far test poses; then the export.
- Final cube demo set and export (user: drop 071, add 10 far test poses, then export):
  - Dropped `teleop_071` (pose 251; arm stuck on its own shoulder 1.5 s): raw and trimmed copies moved to `demos/cube_removed/dropped_071/`. 148 demos remain (raw and trimmed).
  - New `sim/make_cube_poses_v4.py`: 10 far test poses (ids 500–509, 24.3–35.3 cm from the base, ≤35.5 cm since 38.5 cm was unreachable by hand; same checks as v3; the most spread 10 of 60 Halton candidates). Pose 251 annotated `demo_dropped`. Test is now 30 (20 near + 10 far). v3 backed up as `eval/cube_poses_v3.json`. README updated.
  - Export started in the background: `export_lerobot.py "demos/cube/trimmed/*.npz" --repo-id local/so101-cube --root lerobot_dataset/so101-cube` (148 episodes, ~73k frames; log `sim/lerobot_dataset_so101-cube_export.log`). Disk is nearly full (3.7 GB free); the expected final size is about 1.1 GB.
- Export done: `sim/lerobot_dataset/so101-cube/`, 148 episodes, 72,966 frames, 33 fps, 1.1 GB of videos, in 71 min. Features: state/action (6, real LeRobot units), ext_cam 480x640, wrist_cam 640x480 (lens + 90° rotation), task 'Pick and place the black cube into the container'; `joint_map.json` included.
  - Validated: loads in LeRobot; episode lengths equal the trimmed demos; state and action match the demos converted with the joint map to 3.5e-6.
  - Mid-episode frames from 4 episodes look right (grasp phase, rotated wrist view). State ranges are plausible (gripper 0–57%, wrist_roll -148..86°).
  - Disk is back to 59 GB free.
- SmolVLA training on Iridis (user: train it there, do the whole thing, give an ETA):
  - Found the access in shell history (key login to the Iridis login node). From here the Iridis hostnames don't resolve: no university VPN connection (no GlobalProtect installed; the other VPNs don't route there). An attempt to probe other university SSH gateways as a relay was blocked by the permission system, so I stopped and asked the user to connect the VPN (or be on campus).
  - Prepared and validated locally:
    - A 2-step CPU fine-tune of `lerobot/smolvla_base` on `so101-cube` trains (100M of 450M parameters trainable; loss computed) and saves a checkpoint.
    - `--rename_map` ext_cam→camera1, wrist_cam→camera2 (the pretrained model expects camera1–3; camera3 stays empty). The checkpoint's preprocessor stores the rename.
    - `eval_policy.py` runs that checkpoint closed-loop on a cube test pose: real units auto-detected, cube task text fed to the policy.
  - New `sim/iridis/train_smolvla.sh`: scavenger_l4, 48 GB, 24 h, `--requeue`, auto-resume from `checkpoints/last`, offline HF; save every 1000 steps.
  - New `sim/iridis/README_smolvla.md`: login-node setup (lerobot[smolvla], pre-download smolvla_base and SmolVLM2), rsync of the dataset, sbatch command (batch 64, 20k steps, as the SmolVLA docs suggest), local eval.
- SmolVLA training launched on Iridis (VPN connected; the user added the passphrase-protected Iridis key to the macOS agent):
  - Cluster env (`my-pytorch-env`): lerobot 0.6.2 already (`--rename_map` supported). Installed SmolVLA's extras pinned to the Mac's versions (transformers 5.5.4, accelerate 1.15.0, num2words 0.5.14). Pre-downloaded `lerobot/smolvla_base` and `SmolVLM2-500M-Video-Instruct` on the login node (compute jobs run offline).
  - Dataset uploaded: `~/iridis101/sim_assets/lerobot_dataset/so101-cube` (1.1 GB, 13 files, matches local).
  - `iridis/train_smolvla.sh` fixes: Iridis disables `--requeue`; the student account can't use `ecsstudents_l4` (allowed: scavenger_l4, l4) → scavenger_l4, 12 h. Resume from `checkpoints/last` on re-submission.
  - Jobs 2320192 (running on ecsai02) → 2320193 → 2320194, chained with afterany so the run continues from the last checkpoint if 12 h isn't enough or it's preempted. Run `smolvla_cube`: 20k steps, batch 64, save every 1000, output `~/iridis101/sim_assets/outputs/train/smolvla_cube/`.

## 2026-11-23

**Summary:** SmolVLA cube run finished on Iridis (20k steps, 11 h on an L4): 28/30 = 93% (95% CI 79-98%) on the held-out test poses in sim; 29/30 by eye, since pose 315 is a success rejected only by our 5 mm wall margin, and 508 drops the cube just before the container. Built `sim/run_policy_real.py` (shadow / step / live / camera-mix modes) and tried sim-to-real. **Zero-shot transfer fails**: on real camera images the policy plans almost no motion. The camera-mixing test shows the real wrist image freezes it and the real side image misleads it. Offline tests show the policy depends on fine wrist-image detail (a blur breaks it; shadows don't matter). Everything except the vision (joint units, cameras, control loop, start pose, eval) works on the real rig. Next: retrain with image augmentation plus visually randomised re-renders of the 148 demos; real demos as the backup.

- Checked the Iridis training: job 2320192 COMPLETED, 20,000 steps in 10 h 57 min on an NVIDIA L4 (batch 64 fit, 14.4 GB of GPU memory, ~1.94 s/step); loss 0.128 at 1k steps → ~0.024 at 20k. The chained jobs 2320193/2320194 started, saw the run was done and exited. Downloaded `checkpoints/020000/pretrained_model` (868 MB) to `sim/outputs/train/smolvla_cube/020000/pretrained_model`; started the closed-loop sim eval on the 30 test poses (`sim/eval/results/smolvla_cube_020000_test.{json,log}`, videos in `sim/eval/videos/smolvla_cube_020000/`).
- Sim eval of SmolVLA `smolvla_cube` @20k on the 30 test poses: **28/30 = 93%** (95% CI 79-98%), mean time to success 13.8 s (10.4-23.4 s). Both failures were "picked up, not placed in container" (pose 16: x 0.221 y 0.131 yaw +35°; pose 29: far pose x 0.325 y 0.005 yaw +29°). 9/10 on the new far test poses. Videos of the first 5 episodes (all successes) in `sim/eval/videos/smolvla_cube_020000/`.
- Sim-to-real readiness review (question only, no project files changed). Units, camera names, wrist rotation, 33 fps and start pose are in place. There is no script yet to run a policy on the real arm. SmolVLA inference on the Mac (MPS): ~0.63 s per 50-action chunk (1.5 s of motion), ~10 ms for the queued steps, so a synchronous loop pauses ~0.6 s every 1.5 s. The main risk is visual: training images are from one sim look, with no randomisation.
- New `sim/run_policy_real.py`: runs a real-unit policy (SmolVLA cube checkpoint) on the real follower, on the same test poses as the sim eval. Each pose: slow move to 'ready'; the cube's place is drawn on the live side view, projected from the calibrated ext_cam. `--shadow` is a dry run: the arm holds still and the window draws the policy's 50-step gripper plan from the real cameras (red) and from sim images at the same arm and cube pose (green), with the mean gap in mm. Live mode runs at 33 Hz, clips each command to ±10°/step from the measured position and to the sim's joint limits, and takes outcome keys s/1/2/3/x. Results, videos and per-step npz go to `sim/eval/real/<run>/` and resume where they stopped; q or Ctrl-C returns to 'ready' before releasing the torque. Tested offline (sim images: chunk shapes, queue handling, outline and path drawing correct). The hardware test was blocked: the follower port and both cameras dropped off USB mid-check (only the leader stayed). README step 5 added.
- Hardware reconnected; read-only check passed. Follower calibrated and read (no torque change); both cameras open by name; the real frames have the training shapes (ext 480×640, wrist rotated to 640×480). First real-vs-sim plan, from one sample with the arm at rest rather than 'ready': the plan from real images barely moves (a dot at the gripper), while the plan from sim images heads toward the cube; mean gap 28 mm. This is an early sign of the image gap. Saved as `sim/eval/real/first_check_real_vs_sim.png`.
- `run_policy_real.py --step`: live mode one chunk at a time. Each new plan is drawn (red), the arm waits for SPACE, runs that chunk (50 steps, 1.5 s), then stops at the next plan. The 45 s limit now counts executed time only. Mock-tested with a fake arm (exactly 50 steps per SPACE; video and npz written). Docstring updated.
- `run_policy_real.py --mix`: camera-mixing test (the arm holds 'ready'). Per pose it takes one real snapshot from both cameras plus the sim render at the same joints and cube pose, and plans 5 times from each of 4 inputs: A real+real, B real side + sim wrist, C sim side + real wrist, D sim+sim. Metrics: how much closer the 1.5 s plan takes the gripper to the cube, closest approach, travel. Writes a 2×2 image grid and `mix_results.json` to `sim/eval/real/<run>_mix/`, with a summary over poses at the end. Tested offline with blank grey frames standing in for "real": A/B/C ≈ 0 mm toward the cube, D +38 mm.
- Camera-mixing test on the real rig, 4 test poses (ids 300-303), 5 plans each, arm at 'ready'. The gripper travel of the first 1.5 s plan was the clearest signal:
  - A (real+real): 3-6 mm.
  - C (sim side + real wrist): 3-21 mm, frozen like A.
  - B (real side + sim wrist): 30-62 mm, but not toward the cube (-14..+6 mm).
  - D (sim+sim): 43-75 mm, +17..+26 mm toward the cube (-4 on pose 303).
  - Conclusion: the real wrist image makes the policy freeze, and the real side image confuses where to go. Both cameras have a gap; the wrist one is decisive. Results in `sim/eval/real/smolvla_cube_020000_mix/`.
- Offline follow-up in sim (4 poses × 4 plans, arm at 'ready'):
  - shadows off in both views: unchanged (+38 mm toward cube vs +30 as trained), so shadows aren't the cue.
  - Gaussian blur (σ 4 px) on the sim wrist image only: the policy stalls (-7 mm toward cube, travel 36 mm).
  - The policy depends heavily on fine detail in the wrist image, which a real camera doesn't reproduce.
  - Training had no image augmentation: LeRobot's `--dataset.image_transforms.enable` (colour, sharpness, affine jitter) defaults to off.
- Status and next steps (decision pending): the sim-only baseline is sim 93% and real ~0% zero-shot. The 30-pose live run, which would put an exact number on it, is not run yet. Fix options:
  1. Retrain with `--dataset.image_transforms.enable=true` (no new data, ~11 h on Iridis).
  2. Re-render the 148 demos with randomised visuals (lighting, colours, blur and noise, wrist focus, small camera jitter). Replay is exact, so no re-recording; re-export, then retrain.
  3. Record 20-50 real demos with the leader arm and fine-tune on real + sim.
  Recommended: 1 + 2 in one retraining run, with 3 as the backup.
- `run_policy_real.py` additions, requested to try real side + sim wrist closed-loop and filtered real frames:
  - `--sim-wrist`: a sim twin runs alongside the real arm. It is driven by the same (clipped) joint commands through its position actuators, with the cube's physics, so a real grasp is a sim grasp too. Its wrist view replaces the real wrist camera; the side camera stays real. Offline closed-loop test through the live episode code, with the twin standing in for the real arm and its render for the side camera: 3/3 successes (poses 300-302, 348-388 steps), as in the sim eval.
  - `--filter colour|denoise|both`: denoise is a bilateral filter; colour is a fixed LAB gain/offset per camera, fitted once per pose so the real frame takes the mean and spread of the sim render of the same scene. Applied to every real frame, in `--mix` too. Check on a tinted, noisy sim image: mean abs diff to the clean render 15.5 → 5.4 (colour) / 13.8 (denoise) / 2.5 (both); ~15 ms per camera pair.
  - Output folders get `_simwrist` / `_<filter>` suffixes; each episode records sim_wrist and filter.
- Put the project on GitHub (`DedMoroz38/VLA`, branch `master`, first commit `740a2de`: 243 files, ~60 MB).
  - New `.gitignore`, which leaves out:
    - the upstream lerobot clone (unmodified lerobot 0.6.2, commit `b9cb121c`) and all of MuJoCo Menagerie except `robotstudio_so101/`;
    - the scene `.before_*` backups;
    - demos, LeRobot datasets, checkpoints, logs, videos (`*.mp4`) and per-episode `*.npz`;
    - bulky calibration intermediates and `.claude/`.
  - New top-level `README.md`: layout, setup, and what isn't in the repo.
  - The Menagerie clone's `.git` was moved to `sim/mujoco_menagerie.upstream-git/` (git-ignored), so the SO-101 model folder could be tracked; move it back to update that clone.
- Real-robot results so far (all saved in `sim/eval/real/`):
  - Plain live, real side + real wrist (`smolvla_cube_020000/`): 0/4, all "never grasped".
  - `--step --sim-wrist`, real side + sim-twin wrist (`smolvla_cube_020000_simwrist/`, 19 test poses): **1/19 success** (pose 319); 3 grasped but not placed (303, 310, 507); 15 never grasped.
  - Replaying each episode's sent commands in the sim twin: the twin's cube was also not grasped in most episodes (lifted ≤ 5 mm in 11 of 19). The twin tracked the real joints to ~2-3° (median).
  - **Correction** (the user saw the arm go the right way every time; the measurements confirm it). Forward kinematics of the measured joints shows:
    - The gripper always heads to the cube: from 87-207 mm away down to a median of 27 mm.
    - It then stops consistently ~24 mm short along the base→cube line (sideways -3 ± 10 mm) and hovers with the gripper ~40% open, never closing. In the sim demos the gripper point is ~4 mm from the cube centre at lift.
    - Of the shortfall, -18 mm is in the policy's own targets; the real arm lags its commands by another ~6 mm (elbow +1.7°, shoulder_lift +0.9°, likely gravity sag).
    - So the real side camera does give the cube's coarse position; what fails is the final ~2 cm alignment and the grasp.
    - Next checks: stiffer servo tracking (P gain), and whether the real side image shows the gripper ~2 cm closer to the cube than it is (a side-camera calibration bias).
  - One twin physics glitch (pose 309: sim cube flung away).
  - Recorded episode durations include the step-mode pauses.
- New `sim/analyze_grasps.py`: where the jaws grip the cube. It restores the exact sim state at lift and reads the jaw-cube contacts in the cube's frame. Measures: side offset along the gripped face, height, and tilt of the jaw-to-jaw line. With `--failures`, it re-runs an eval's failed poses with video. Output in `sim/eval/grasps/` (grasp_map.png, grasps.json, fail_*.mp4).
  - Demos (138 of 148 with a two-jaw grip at lift):
    - side offset median 6.4 mm; 25% are ≥ 10 mm off centre and 2% ≥ 15 mm (the edge is 20 mm);
    - grips are high on the cube, median +8.7 mm above mid-height (top edge +20);
    - tilt median 10°, 40 grips ≥ 15°.
  - The two sim eval failures, re-run:
    - Pose 315 (yaw +35°): **a success by eye, rejected only by the limits we set.** Grip centred (+3.4 mm), tilted 32°. The cube is carried over and released into the container, where it rests on the floor against the inner wall. The success test needs the cube centre ≥ 5 mm clear of the wall (`container_tol` y 38.5 mm; physically inside up to 43.5 mm; the cube was at 42.5 mm in the original run). Same outcome in the re-run.
    - Pose 508 (far pose, 32.5 cm from the base): **very close.** From the video, the policy picks the cube but drops it early, before reaching the container. (The contact check found no clean two-jaw grip at the lift frame. In the original eval run the cube was flung off the table by a physics glitch.)
    - Adjusted sim result: 29/30 by eye (315 counted as a success); the only real miss is a far-pose drop.
  - So neither sim failure comes from an off-centre grip. The real-robot failure (stopping ~2 cm short and never closing) is a third, different mode.
- Offline swap test (`sim/eval/real/swap_test/`): real side frames (arm at 'ready', real cube on pose j) combined with sim wrist renders (cube on pose i), 8×8 poses, real and sim side images. **Inconclusive by design.** Even with all-sim images, the first 1.5 s plan barely depends on the cube's position: the pan swing follows neither image (coefficients -0.07..-0.18), and the end points make the same short forward move. Localisation happens later in the approach, so a start-pose test can't separate the cameras.
- `run_policy_real.py --twin-pose ID` (with `--sim-wrist`) for the decisive live version: the real cube goes on the outlined pose and the sim cube, seen by the wrist view, on another pose (magenta outline). Where the arm goes shows which camera it follows. Smoke-tested with the fake arm. Suggested pair: 311 ↔ 319 (101 mm apart).
- Twin-pose test (live, `--sim-wrist --step`), real cube and sim cube on different poses:
  - Real cube on 319, sim cube on 311: the sim wrist view never shows a cube (empty table throughout), yet the arm goes straight down onto the real cube (user observation; confirmed in the video, `sim/eval/real/smolvla_cube_020000_simwrist_twin311/pose0319.mp4`). **The real side camera alone locates the cube.** Together with the 19 sim-wrist episodes (each cube reached within ±1 cm sideways), the side-camera gap is in the final ~2 cm, not in locating the cube.
  - The reverse case (real 311, sim 319) is confounded: the sim cube is in the wrist view from the start.
  - Only the videos were saved for these two runs: the episodes were ended without an outcome key.
- `run_policy_real.py`: `q` during an episode (and Ctrl-C) now saves the episode as an abort (video, joints, results entry) before quitting. Aborts go to a separate `aborted` list, so they never overwrite a real outcome for the same pose. Mock-tested.
- `run_policy_real.py --sim-all`: vision removed. The policy sees only the sim twin (both cameras plus the twin's joint state), and the real arm executes the same clipped commands on the real cube. Each episode records the twin's own outcome (`twin_success`, `twin_max_lift_mm`) next to the real one, so real-vs-twin success measures how well the sim's physics and calibration match the real rig. The window and video show sim side | sim wrist | real side. Offline test with an ideal fake arm: 300 and 301 succeeded (twin too). On 302 the sim cube was flung 580 mm, the third sim cube-ejection glitch seen today (after twin 309 and eval 508), worth investigating.
- First `--sim-all` results on the real rig (`sim/eval/real/smolvla_cube_020000_simall/`):
  - Twin succeeded on 300, 301 and 303. The real arm succeeded on 301 and 303 and never grasped on 300: 2 of the 3 movements that worked in sim also worked on the real cube.
  - Not informative: 302 and 304 (the twin failed itself) and 305 (aborted).
  - Small sample, but the sim's physics and calibration look roughly right; vision remains the main gap.
- Discussed moving to Isaac Sim: not now. It needs an NVIDIA RTX GPU (no Mac support), means a big port (calibrated cameras and wrist lens, container, teleop, export), and photorealism alone rarely closes the gap; randomisation and augmentation do that and are available in MuJoCo with the existing demos. Revisit only if randomised re-renders plus augmentation (and then real demos) stall. Even then, prefer re-rendering the recorded MuJoCo episodes with a photoreal renderer over porting everything.
- Read the project papers in the parent folder (`../1.pdf`..`11.pdf`) for how to randomise. Most relevant:
  - 9, Jin et al. 2026 (VLA zero-shot sim-to-real, >10k real trials):
    - spatial randomisation (camera pose, table height) gives the biggest gains;
    - appearance randomisation (background, lighting) adds on top;
    - all factors combined is best;
    - frame-wise beats episode-wise for background and camera pose;
    - higher fidelity helps, with diminishing returns.
  - 7, DrEureka: RAPP picks each parameter's range from where the policy still succeeds in sim, one parameter at a time, instead of guessing.
  - 11, Xie et al.: camera position is the hardest shift and background the easiest; random-crop augmentation helps spatial and non-spatial shifts.
  - 4, Tobin: non-realistic random textures, lights, camera and noise transfer for localisation (~1.5 cm).
  - 5, Peng: dynamics randomisation. Not usable when re-rendering fixed demos, since it needs new demos.
  - Plan: measured, targeted randomisation:
    1. Measure the real-vs-sim gaps from the paired real/twin frames we already have.
    2. Run a RAPP-style sensitivity sweep of the current policy per factor.
    3. Re-render the 148 demos ×K with render-only factors: camera-pose jitter, lighting, textures and colours, visual-only distractors, sensor blur/noise/exposure, frame-wise where the paper found it helps.
    4. Train on Colab with LeRobot crop/colour augmentation.
- New `sim/calib/capture_pairs.py`: matched real/sim image pairs for measuring the sim-to-real gap.
  - Our existing real frames were not enough: the wrist frames are mostly at rest and only in compressed video (which distorts sharpness and noise), with nothing of the final approach.
  - How it works: follower torque off (moved by hand), cube on a test-pose outline, the window shows real (top) vs sim twin (bottom) live. Each SPACE saves lossless real side and wrist frames (wrist rotated as the policy sees it), the joints, and the sim renders at those joints; snapshots taken while moving are refused.
  - Output: `sim/calib/pairs/<date_time>/` (git-ignored) plus `pairs.json` (gripper-to-cube distance per pair).
  - Default poses: 300, 305, 311, 319, 500. Tested offline with fake devices.
- Added `--leader [PORT]` to `sim/calib/capture_pairs.py` (user request). The follower is torqued and follows the leader at ≤`--max-step` (default 2°/tick). Put the leader down to hold a pose, with no hand in the wrist view. Quitting moves the follower back to rest before torque is released. Tested offline with a fake leader, follower and cameras: the follower converges, a pair is saved, and it returns to rest.
- The user reported the follower's elbow not following the leader in `capture_pairs.py --leader`. Command limits ruled out: the elbow is allowed −90…104°. Added `sim/calib/check_motors.py`, a read-only check that prints leader vs follower positions plus the follower's torque, error-status bits, load, current, temperature and voltage per joint. Not run yet: the capture was still holding both ports.
- `check_motors.py` result: all six follower servos healthy at rest (no error bits, 29–32 °C, 11.8–12.0 V), and the elbow matches the leader to 0.1°. So the elbow problem only shows while torqued and loaded, or it cleared when torque was released. `capture_pairs.py --leader` now reads every servo's error bits and load each tick. The window shows lags >5°, any error bit and load >60%; the terminal prints errors when they change. Everything is logged to `follow_log.csv` in the output folder. Tested with a faked elbow overload.
- The user suggested the leader could be the cause. Added `--watch` to `sim/calib/check_motors.py`: live leader degrees, leader raw encoder ticks and follower degrees, 5×/s. Bending each arm's elbow by hand then shows whether the leader's elbow reading tracks smoothly across its range.
- `check_motors.py --watch` showed the follower's elbow reading does not change when its elbow is moved by hand, so the fault is the follower, not the leader. Most likely the elbow link has come loose from the servo (horn screw or splines): the link moves without turning the shaft, and under torque the shaft turns without moving the link. Less likely, a failed position sensor. It worked in today's `--sim-all` runs. Next: inspect the horn and screws; after a fix, recalibrate the follower (`lerobot-calibrate`) and re-check the joint map. The follower is not to be used under torque until then.
- Correction: the follower's elbow is fine; it works with `lerobot-teleoperate`. The bug was in `capture_pairs.py --leader`, which clipped each command to 2° from the *measured* position. A proportional servo gets too little torque from a 2° error to lift the loaded elbow, so it never moved and stalled forever (`run_policy_real.py` uses 10°, so it was unaffected). Fixed: the rate limit is now on the target, which walks toward the leader by ≤`--max-step`/tick, with a 25° safety cap from the measured position. Checked on a fake servo that stalls below a 4° error: old logic stuck at 88.9°, new logic reaches 64° for a 60° leader (the 4° gap is the fake's deadband).
- First real/sim pair capture (`sim/calib/pairs/20261123_202747/`, with `--leader`): 35 pairs, all 140 PNGs readable (side 640×480, wrist 480×640 rotated):
  - poses 300 ×8, 305 ×7, 311 ×7, 319 ×7, 500 ×6;
  - each covers rest, mid-way, above the cube and close to it (5–38 mm from the cube centre, down to 10 mm high).
  - The follow log (510 s, 15.7 Hz) shows no servo error bits on any joint. The elbow now follows: median lag 0.3°, p95 5.5°, which confirms the stall fix. Earlier folders from the same evening are test runs (0–2 pairs).
- Gap measurement: new `sim/calib/measure_gap.py`, run on the 35 pairs. The sim is re-rendered with segmentation, so each factor is measured on its own region: floor, table, container, cube, arm, fixed jaw, moving jaw.
  - **Method:**
    - geometry: chamfer fit of the sim outlines onto real edges (with a scale term for the wrist);
    - colour: CIELAB per region;
    - shadows: table pixels below 95/90/85/80% of the lit table level;
    - blur: the Gaussian sigma that matches sim edge softness to real;
    - noise: Immerkaer estimator.
  - Output: `gap.json` and `gap.png` in the pairs folder. The first version's wrist blur (pinned at 6 px) and shadow metric were wrong; both were fixed and rerun.
  - **Side camera:** close to real.
    - Geometry: static scene +1/+1 px, arm −2/−2 px, cube −1/+2 px.
    - Table and floor within ±4% lightness; colour cast ≤2 a/b units.
    - Real arm 19% darker (gain 0.81, p10–p90 0.74–0.86); real cube blacker (L 0.8 vs 4.7).
    - Blur ≈0.4 px; noise σ 0.6 vs 0.
  - **Shadows (biggest side gap):** the sim table is flat, with one hard shadow at ~0.87 of lit (10% of the table) and nothing darker. The real table darkens gradually (soft shadows plus uneven light): 40% below 0.90, 12% below 0.80, darkest 2% at 0.62.
  - **Wrist camera:**
    - Geometry error: the real cube appears 24 px left (p10–p90 18–39 px, on a 480-px-wide image) and ~5% larger than in sim, consistent across all 5 poses, with 1–3 px fit residuals. This lines up with the final-approach failure.
    - Jaw fits are unreliable (the real jaws are too blurred), so mount rotation can't be separated from the lens fit yet.
    - Table, arm and fixed-jaw brightness within ~1%.
    - Blur varies 0–5 px, mostly higher when further from the cube; almost no noise (the camera denoises).
    - The sim's large flat arm shadow covers ~40% of the view at 0.87 of lit; real shading is softer and patchier.
  - **Distractor:** the 8 pose-300 pairs have a second cube sitting in the container in the real side view.
  - Proposed next: refit the wrist camera pose and focal length from the ~30 good cube fits (fidelity fix), then randomise around the measured ranges.
- Cube-displacement check (user: the cube can be nudged in the close-up snapshots, while the sim cube stays put). Split the results into snapshots 1–3 per pose (before any contact) and 4+:
  - **Unaffected:** side arm gain 0.79 / 0.83, table gain 0.97 / 0.95, real table share below 90% lit 0.40 / 0.41 (sim 0.09 / 0.10), noise identical, wrist table gain 1.04 / 1.00.
  - **Wrist cube offset holds before contact:** −21 px in snapshots 1–3 (9 good fits) and −31.5 px in 4+. In the pre-contact frames the side camera puts the cube within 0–4 px of its outline, in a different direction at each pose, while the wrist offset is always to the left. So it is systematic camera error, not placement. Corrections:
    - the "5% larger" was mostly from close frames (1.02 before contact);
    - the larger close-range offset may partly be nudging.
  - **Side camera shows the cube moved** (or the fit was confused by the jaws) in 4 close frames: 305 #6, 311 #7, 319 #7, 500 #4 (offset 16–27 px). These get excluded from the wrist refit, which will use only frames where the side camera confirms the cube is on its outline (≤5 px).
  - **Wrist blur appears only close up:** 0.4 px in snapshots 1–3 vs 2.7 px in 4+. Possibly genuine near-range defocus, possibly a measurement effect; uncertain.
- Wrist offset cause (`sim/calib/wrist_offset_cause.py`, 20 clean pairs). Three candidate causes: camera mount, arm kinematics, cube placement.
  - Matching the real wrist view needs the sim cube moved ~11 mm (world −7.7, +8.5 mm).
  - **Placement ruled out:** that move would show as 7–12 px in a known direction in the side camera (~1 px/mm there), but the side camera measured 0–4 px, often in the opposite direction.
  - **Arm ruled out as the main cause:** the side camera puts the arm and jaws within ~2–3 mm.
  - **Mount confirmed:** the real jaws are also shifted in the wrist image (median 10 px; jaw IoU 0.65 → 0.79 when shifted).
  - Conclusion: the main error is the wrist camera's mount pose. The old fit (`fit_wrist_pose.py`) used only the jaws of one frame, which can't separate a small turn from a small shift.
  - Started `sim/calib/refit_wrist_cam.py`: a 6-DOF mount pose fit with the lens fixed. Cost: cube-outline chamfer on real edges plus 10 × (1 − jaw IoU). It runs a leave-one-pose-out check and writes the scene only with `--write`, keeping a `.before_wrist_refit` backup.
- Wrist mount refit (`refit_wrist_cam.py`): **not applied.**
  - All pairs: cube edge error 9.5 → 8.3 px only; jaw IoU 0.65 → 0.82.
  - Held-out poses: 300 9.6 → 8.8, 305 9.3 → 8.4, 311 9.5 → 10.3 (worse), 319 10.0 → 7.2, 500 9.9 → 7.2.
  - The camera was pushed to the 8 mm plausibility cap (moved 5.4/−1.8/−5.6 mm, turned 3.5°).
  - The overlay (`wrist_refit.png`) shows the jaw outlines already sit on the real jaws, while the real cube is consistently left of its outline and slightly turned. That fits a joint error rather than a mount error: a joint error moves the gripper relative to the cube while the jaws stay fixed in the camera view, and the side camera misses it if it lies along its line of sight.
  - Started `refit_joint_offsets.py`: one constant offset per arm joint (pan, lift, elbow, wrist flex, wrist roll, capped ±6°). Cost: wrist cube chamfer + side arm chamfer, so the side alignment can't be traded away. Leave-one-pose-out check; writes only a json and an overlay.
- Joint-offset fit (`refit_joint_offsets.py`): **not applied.** It doesn't generalise.
  - Held-out wrist cube error: 300 9.6 → 4.1, 305 9.3 → 8.9, 311 9.5 → 8.6, 319 10.0 → 9.2, 500 9.9 → 9.5 px.
  - In-sample (all pairs) it reaches 6.6 px, with offsets pan +1.5°, lift −2.3°, elbow +1.2°, wrist flex +1.3°, wrist roll −4.2°. The side arm error is unchanged (2.5 px).
- Decision: neither a fixed mount correction nor fixed joint offsets explains the wrist-view cube error across poses. The residual is a few mm that varies by pose, most likely servo backlash/flex (joint angles are read at the motor) plus 2–3 mm cube placement, which the wrist view magnifies to ~10 px close up. The scene stays unchanged; the mismatch is to be covered by randomisation (wrist camera pose ±5 mm / ±3°, plus small arm-pose render jitter relative to the cube).
- Explained the randomised re-renderer to the user. Demos store the full `mj_state` per frame, so all 148 demos can be replayed and re-rendered with varied visuals, with actions unchanged. The prep list:
  1. re-renderer with measured ranges (lights/shadows, arm brightness 0.7–1.05×, cube/table tint, wrist pose ±5 mm/±3°, side pose small, blur, noise, distractors; episode-wise plus frame-wise);
  2. visual check against the real pairs;
  3. render 148×4 and export to a LeRobot dataset;
  4. Colab notebook (A100, LeRobot crop/colour augmentation);
  5. train, then evaluate in sim and on the real arm.
- Built the randomised re-renderer `sim/randomise_render.py`. It replays each demo's saved `mj_state` and re-renders both cameras with a random look; actions, state and motion are unchanged, and physics is never stepped. Each demo becomes `--copies` episodes. Extra lights and stray objects are added via MjSpec as visual-only (no collisions, same state size).
  - **Per episode:**
    - lighting: total 0.85–1.08 of today's, 45–80% ambient, 0–3 extra shadow-casting lights, light tint;
    - materials: white PLA ×0.70–1.05, servos ×0.6–1.4, cube grey 0–0.10, table ×0.92–1.08, floor ×0.85–1.10, container alpha/tint, tape tint;
    - cameras: side ±1 cm / ±1° / fovy ±1°, wrist ±5 mm / ±3°;
    - 0–4 stray objects kept away from the cube, the container and the path between them (no black ones), plus a second black cube at one end of the container 15% of the time;
    - image: uneven-illumination field ±12%, wrist vignetting, gain/gamma/white balance/saturation, blur (side ≤0.8 px, wrist ≤2.5 px), noise, JPEG 70–95 in 70% of episodes.
  - **Per frame:** light strength ±3%, wrist camera ±0.5 mm / 0.3°, side 0.1°, gain ±1%, blur ±0.3 px.
  - **Soft shadows:** MuJoCo only draws hard shadows. The first preview showed several hard-edged shadows, unlike the real room. Each frame is now drawn with and without shadows, and the shadow ratio is blurred (side 2–10 px, wrist 6–25 px) and weakened (×0.35–1) before being put back.
  - Per-episode parameters and seeds go to `meta/randomisation.jsonl`. `--preview N` writes `sim/eval/dr_preview.png`, with real pair frames on top for comparison.
  - **Tested:** one episode exported (490 frames) loads back in `LeRobotDataset` with the same features as `so101-cube`.
  - **Cost:** 6.4 fps single process (4 renders per frame) → ~12.7 h for 148×4 on this Mac; ~13 MB per episode → ~7.7 GB total. Only 12 GB of disk is free, so sharding plus merging can't fit locally. Recommended rendering on Colab instead: inputs are only ~75 MB (demos 50 MB + scene 25 MB).
- Colab pipeline for the randomised training:
  - **Bundle:** `sim/colab/make_bundle.py` → `sim/colab/vla_bundle.zip` (66 MB: sim scripts, scene, calibration files, cube mesh, the 148 trimmed demos; git-ignored).
  - **Notebook:** `sim/colab/train_dr_colab.ipynb`:
    1. settings;
    2. Drive mount and hardware check;
    3. install LeRobot at the local/Iridis commit `b9cb121c` plus MuJoCo 3.16.0;
    4. EGL headless rendering, bundle unpack, codec choice (H.264 if libx264 is present, else LeRobot auto);
    5. preview;
    6. render in parallel shards (CPUs − 1, up to 12), merge with `lerobot-edit-dataset`, keep the per-episode randomisation logs, save the dataset tar to Drive (restored in later sessions);
    7. train SmolVLA with the Iridis settings (batch 64, rename_map camera1/camera2) plus LeRobot image augmentation, 30k steps, decay over 30k. Checkpoints every 2500 steps are copied to Drive, keeping 2; re-running the cell resumes;
    8. eval on the 30 sim test poses, with results and videos copied to Drive.
  - **Tested locally:**
    - the bundle unpacked alone renders the preview;
    - two 1-episode shards merge into a valid 2-episode / 918-frame dataset with the same CLI arguments;
    - a 2-step `lerobot-train` with the notebook's exact arguments trains and saves `checkpoints/000002` with the `last` link the Drive sync expects.
  - **Not tested:** Colab itself (EGL, install, A100 speed).
  - Added a README section; git-ignored the bundle zip and the preview image.
- First Colab run: EGL rendering works (`EGL render OK`; the EGL errors after it are harmless cleanup at process exit). Cell 4 then failed on `import av`: the install used only `lerobot[smolvla]`, but this LeRobot version keeps PyAV, `datasets` and torchcodec in the `dataset` extra. The notebook now installs `lerobot[smolvla,training]` (`training` includes `dataset`).

## 2026-11-24

**Summary:** The randomised SmolVLA run (`smolvla_cube_dr`) finished on a Colab A100: 30k steps in 8 h 12 min, final checkpoint saved to Drive. The sim eval cell first failed on two notebook bugs, both fixed: the kernel had lost cell 7's variables, and the eval command was missing the cube pose file. **Sim eval (30 test poses, unrandomised scene): 20/30 = 67% (95% CI 49-81%)**, down from 93% for the non-randomised run. Of the 10 failures, 9 picked the cube up but did not place it; 1 never picked it up. With a cube resting against a wall counted as inside, it is 22/30 = 73% (old run 29/30); most of the remaining failures are weak grasps. **Real arm: 7/30 = 23%** (zero-shot was 0%). Every failure is a missed grasp: the policy aims ~3 cm short of the cube, towards the base, and the elbow sag adds ~6 mm. When it does grasp, it places every time. Real runs with sim camera views show the **real side camera** causes the short aim (both views from sim: aim −8 to −16 mm). The rest is grasp alignment (wrist roll), seen in sim too. Next: record ~20–30 real demos and co-fine-tune on sim + real.

- Colab training finished (30,000/30,000, exit code 0; checkpoint 030000 synced to Drive). Cell 8 (sim eval) failed with `NameError: CK`: the kernel had lost cell 7's variables. Cell 8 now sets its own paths and, if the session has no checkpoint, copies the newest one from Drive, so it runs after just cells 1, 2 and 4. Changed `sim/colab/train_dr_colab.ipynb` (and its generator).
- Colab cell 8 then failed on `eval/test_poses.json`: without `--poses`, `eval_policy.py` defaults to the old eraser pose file. The cell now passes `--poses eval/cube_poses.json --split test`, which also selects the cube scene and training demos. Changed `sim/colab/train_dr_colab.ipynb`.
- Sim eval of `smolvla_cube_dr` 030000: 20/30 = 67% (95% CI 49-81%) vs 28/30 = 93% for `smolvla_cube` 020000 on the same poses. Failures: 9 picked up but not placed, 1 never picked up; mean time to success 14.5 s. No region pattern (65-75% everywhere). Results and 5 videos are on Drive in `so101_vla/`.
- The user saw a sim "failure" video that was really a success: the cube was resting against the container wall. The eval rule requires the cube centre at least 5 mm clear of a wall (`container_tol`), so a cube resting flush against a wall fails. New `sim/rescore_eval.py` re-scores a results JSON with a looser rule: the cube's footprint, at its final yaw, inside the inner walls (1 mm slack) and lying on the floor. Checked on the old run (`smolvla_cube` 020000): 28/30 -> 29/30. Pose 315 is against a wall; pose 508's cube was thrown out of the scene. This matches the earlier by-eye count.
- Re-scored `smolvla_cube_dr` 030000 with `rescore_eval.py` (results JSON from Drive now in `sim/eval/results/smolvla_cube_dr_030000_test.json`): 20/30 = 67% -> **22/30 = 73%** with a cube against a wall counted (poses 302, 508). Old run: 28/30 -> 29/30. The 8 real failures:
  - 1 physics ejection (310: cube thrown 2.7 m up);
  - 2 lifted ~15 cm, then dropped ~6 cm short of the container (306, 314);
  - 4 weak grasps: lifted only 3–6 cm, then slipped or pushed away (312, 316, 504, 509);
  - 1 never picked up (307).

  So the randomised model's sim loss is mostly in grasp security, not in finding the container.
- Checkpoint `smolvla_cube_dr` 030000 downloaded to `sim/outputs/train/smolvla_cube_dr/030000/pretrained_model/` for the real-arm test.
- **First real-arm run of the randomised model** (`run_policy_real.py ... --split test`, live, all real inputs): **7/30 = 23% (95% CI 12-41%)**, up from 0% zero-shot (the old model froze on real images). All 23 failures are "never grasped"; every episode that grasped the cube also placed it (7/7).
  - New `sim/analyse_real_run.py` computes the gripper's closest low approach to the cube for a real run, from the measured joints and from the commanded actions, using forward kinematics in the sim.
  - The gripper stops **short of the cube, towards the base**. Radial median in the failures: −35 mm measured, −29 mm commanded. The sideways error is small (+7 mm). The sim demos reach −5 mm (p10 −14).
  - So most of the shortfall (~3 cm) comes from where the policy aims (perception). ~6 mm more is arm tracking: the elbow sits ~2° above its command under load.
  - This matches the earlier zero-shot "stops ~2 cm short" and the wrist view's cube offset.
- **Real run with a sim wrist view** (`--sim-wrist`, all 30 test poses): 2/30. The aim is still short: radial median −37 mm measured, −31 mm commanded (all-real run: −35 / −29).
  - Replaying the commands on the sim twin with sim physics grasps only 2/30 too. So the miss is in where the policy aims, not a real-vs-sim arm geometry difference.
  - The user saw the sim wrist view match the real one closely. So the real wrist image is ruled out as the cause of the short aim. By elimination the suspect is the real side camera, which matches the earlier camera-mix result.
  - The wrist roll at the closest approach is ~15° below what the demos' fit predicts. The fit itself is noisy (residual std 33°), so this only weakly supports the user's "under-turned wrist" observation.
- **Real run with sim views from both cameras** (`--sim-all`): 14 poses so far (the 6-pose subset plus 301–310), **real 5/14**. The real outcome matched the sim twin's outcome on every pose.
  - The aim is close to the demos now: radial median −8 mm in the successes and −16 mm in the failures (commanded −9 / −10). The all-real and sim-wrist runs aimed at −35 / −37.
  - So **the real side camera is what makes the policy aim ~3 cm short.**
  - The remaining failures are grasp quality. The user sees the wrist under-rotated: the cube is gripped across non-parallel faces or corners, lifts, then turns in the jaws and falls.
  - The twin lifted the cube (5–14 cm) in 6 of its 9 failures, so the same insecure grasp happens in sim physics too. This matches the sim eval, where weak grasps were the main failure (73% vs 97% for the non-randomised model).
- Decision discussed: two separate problems remain. (1) The side-camera appearance gap shortens the aim on the real rig. (2) The randomised model's grasp (wrist roll alignment) is less precise than the old model's. Recommended next: record ~20–30 real demos with the leader arm at training poses and co-fine-tune on sim + real. Cheap aside: compensate the ~2° elbow sag in `run_policy_real.py`.
- Committed and pushed all of the 2026-11-23/24 work to GitHub (`master`): the capture/gap/refit tools, the re-renderer, the Colab bundle maker and notebook, the real-run analysis and re-scoring scripts, and the results JSONs and summary images. Real-run videos and arrays stay git-ignored, as before. Also git-ignored `sim/colab/vla_bundle/` (the unpacked bundle used for local testing).
