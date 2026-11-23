# Wrist camera mount (Seeed SO-ARM101 Pro kit)

The real arm (Seeed SO-ARM101 Pro assembled kit, Amazon B0GS4WQ1TM) carries its
32x32 mm USB wrist camera on Seeed's own bracket, not one of TheRobotStudio's
official mounts (`candidates/` holds those three, which don't match: see
`candidates_render.png`, `readme_plug_photos.jpg`).

Seeed's wiki links the part `SO-ARM101_CAMERA_MOUNT` inside
`xiehuangbao888/soarm_soft_gripper` (`soarm_soft_gripper.stp`, rev 9ea41e7).
`seeed/` holds the mesh of that part as converted and aligned to the SO-101
MuJoCo gripper frame by `Fatsharkovo/so101-benchmark` PR #7, plus its manifest
(source hashes, transform) and placement XML. The source repo has no explicit
licence file.

Applied to `mujoco_menagerie/robotstudio_so101/so101_real.xml` (2026-11-19,
backup `so101_real.xml.before_seeed_mount`): the Seeed mesh plus a 32x32 board
and lens on its camera plate (board/lens placement from PR #7's estimate:
1.6 mm PCB + 5 mm lens from the seating face). The official mount, board and
lens are kept but hidden (group 5), and the new geoms are massless visuals,
so mass, inertia and collisions are unchanged and demos replay bit-exactly.
The `wrist_cam` viewpoint was NOT changed.

Check (`overlay_after.png`, `board_zoom.png`): with the real arm at rest
(`side_now.png`, joints in `joints_now.json`), the sim board and bracket land
on the real ones in the side camera to within ~1-2 px (~1-2 mm). Before
(`overlay_before.png`) the sim board hung clearly lower and to the left.
