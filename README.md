# SO-101 pick-and-place: calibrated MuJoCo sim → VLA policies → real robot

A MuJoCo simulation of the LeRobot SO-101 arm, calibrated to a real rig (follower arm,
Logitech C920 side camera, wrist camera), used to collect teleoperated demonstrations,
train ACT / SmolVLA policies, and test sim-to-real transfer on the real arm.

Details, commands and results: [`sim/README.md`](sim/README.md); day-by-day log: [`PROGRESS.md`](PROGRESS.md).

## Layout

| Path | What |
|---|---|
| `sim/` | Simulation, teleop recording, export to LeRobot datasets, evaluation, real-robot runner |
| `sim/mujoco_menagerie/robotstudio_so101/` | The SO-101 model plus this project's real-look scenes (`pick_place_scene_real.xml`) and assets |
| `sim/calib/` | Camera and joint calibration against the real rig (`joint_map.json`, camera fits, intrinsics) |
| `sim/container/` | Builds the placement container and its tape into the scene |
| `sim/eval/` | Fixed start-pose sets (`cube_poses.json`) and evaluation results |
| `sim/iridis/` | Slurm scripts for training SmolVLA on the Iridis cluster |
| `models/`, `photos/` | Cube mesh; floor photo used for the sim floor texture |

## Setup

- Python 3.12 with `mujoco` (3.x), `opencv-python`, `av`, `numpy`, `pillow`
- `lerobot[smolvla]==0.6.2` (developed against upstream commit `b9cb121c`)

## Not in the repo (too large; regenerated or stored elsewhere)

- `sim/demos/`: raw teleoperated demonstrations (`.npz`)
- `sim/lerobot_dataset/`: exported LeRobot datasets
- `sim/outputs/`: trained checkpoints
- evaluation videos (`*.mp4`) and per-episode arrays (`*.npz`)
- the rest of MuJoCo Menagerie and the lerobot source tree
