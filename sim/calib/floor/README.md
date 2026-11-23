# Real lab floor in the sim (cube scene)

`build_floor.py` builds `mujoco_menagerie/robotstudio_so101/assets/floor_real.png`
and puts it in `pick_place_scene_real.xml` (backup `*.xml.before_floor`).

```bash
cd sim
/opt/anaconda3/envs/lerobot/bin/python calib/floor/build_floor.py              # rebuild + write the scene
/opt/anaconda3/envs/lerobot/bin/python calib/floor/build_floor.py --no-write   # rebuild + check only
```

Inputs: `side_1080p_now.png` (1080p frame from the calibrated side camera) and
`../../../photos/floor.jpeg` (phone photo of the same floor from above the robot).

How:
1. Side-camera pixels above the table's back edge are traced onto the floor plane
   (72 cm below the table top) with the tuned `ext_cam`. This gives a metric top-down
   image of the 0.6–1.3 m strip behind the robot that the camera sees.
2. The photo is registered to that strip (SIFT + RANSAC homography: 79 inliers,
   1.2 mm median reprojection), then warped onto the same 2 mm/px grid. It covers
   about 0–1.9 m behind the base, including the floor hidden behind the arm.
   Non-floor parts of the photo (table, robot, cables, a stool, a chair leg) are
   masked by hand-drawn polygons in `build_floor.py`.
3. A robust 3x3 + bias colour transform maps the photo's colours to the C920's
   (median residual 5 grey levels).
4. Texture: colour-corrected photo, plus the C920's own pixels at the strip's
   right end, which the photo misses. Partly seen 0.25 m tile cells are completed
   from their own colour; the rest of the 3.4 x 4.4 m patch is filled with real
   1 m blocks, aligned to the measured 0.25 m tile grid.
5. Lighting: the rendered sim floor is matched to the real frame per channel
   (final ratio about 1.00, mean abs difference about 6 grey levels).

Scene changes: textured visual-only box at z = -0.72; physical floor plane moved
from -0.75 to -0.72 with the mean floor colour; table legs shortened to reach it.
Only the cube scene has it; the eraser scene is unchanged.

Check: `floor_compare.png` (640x480 crop: real / sim / difference). The floor is
dark in the difference except thin tile-edge lines; the arm differs because the
sim shows its keyframe pose.
Floor outside the camera's view is plausible, not exact: copied blocks repeat,
and a few cells at the photo's far edge are smudged.

Note: MuJoCo sets the camera near and far clip and the shadow-map area from the
model's size (`stat.extent`). The 4 m floor patch tripled that size (1.87 → 5.57 m),
which pushed the wrist camera's near clip from 19 to 56 mm and cut off the jaws.
It also made shadows about 3x coarser. `build_floor.py` therefore pins the scene
size to its value without the floor (`<statistic center=... extent="1.8722"/>`),
which keeps the near clip at 19 mm and the shadows as they were.
