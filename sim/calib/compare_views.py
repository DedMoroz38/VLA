"""Render the sim cameras next to real frames, for checking the sim-to-real match.

    python calib/compare_views.py                       # default frames
    python calib/compare_views.py --side real/x.png --wrist real/y.png
    python calib/compare_views.py --qpos 0 -1.6 1.5 1.0 0 -0.17
    python calib/compare_views.py --pose P3     # a captured pose: real frame + measured joints
    python calib/compare_views.py --all-poses   # grid of every captured pose

Writes calib/compare_<cam>.png: real | sim | 50/50 blend, all 640x480, and
prints mean RGB of matching regions (table, robot) in both.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
W, H = 640, 480


def load_640x480(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if (w, h) != (W, H):        # both cameras' 640x480 mode = centre 4:3 crop of 1080p
        cw = h * 4 // 3
        img = img.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h)).resize((W, H), Image.LANCZOS)
    return np.asarray(img)


def patch(a, x, y, r=6):
    return a[y - r:y + r, x - r:x + r].reshape(-1, 3).mean(0).round().astype(int)


ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def pose_qpos(name):
    """Measured LeRobot joint angles of a captured pose, mapped to sim radians
    with the per-joint offsets from side_cam_fit.json."""
    joints = json.loads((HERE / "poses/joints.json").read_text())[name]
    offs = json.loads((HERE / "side_cam_fit.json").read_text())["joint_offsets_deg"]
    return [float(np.radians(joints[j] + offs[j])) for j in ARM] + [-0.17]


def all_poses(scene):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    rows = []
    for name in json.loads((HERE / "poses/joints.json").read_text()):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:6] = pose_qpos(name)
        d.qpos[6] = 5                                   # task object off the table
        tray = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("container_")]
        m.geom_rgba[tray, 3] = 0                        # hide the tray, if the scene has one
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="ext_cam")
        sim = r.render()
        real = load_640x480(HERE / f"poses/{name}.png")
        blend = (real.astype(float) * 0.5 + sim.astype(float) * 0.5).astype(np.uint8)
        rows.append(np.hstack([real, sim, blend])[::2, ::2])
    Image.fromarray(np.vstack(rows)).save(HERE / "compare_all_poses.png")
    print("wrote", HERE / "compare_all_poses.png")
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--side", default=str(HERE / "real/side_now_480.png"))
    ap.add_argument("--wrist", default=str(HERE / "real/wrist_pair_480.png"))
    # Defaults: the pose the real arm was in for side_now_480.png (fitted by
    # overlay), and a gripper-down pose like the one in the wrist frame.
    ap.add_argument("--qpos", type=float, nargs=6, default=[0, -1.7, 1.69, 1.0, 1.57, -0.17],
                    help="arm joint angles for the side-cam render (rad)")
    ap.add_argument("--wrist-qpos", type=float, nargs=6, default=[0, -0.3, 0.6, 1.3, 0, -0.17],
                    help="arm joint angles for the wrist-cam render (rad)")
    ap.add_argument("--pose", help="name of a captured pose in calib/poses/ (uses its frame and measured joints)")
    ap.add_argument("--all-poses", action="store_true", help="real|sim|blend for every captured pose")
    args = ap.parse_args()
    if args.all_poses:
        return all_poses(args.scene)
    if args.pose:
        args.side = str(HERE / f"poses/{args.pose}.png")
        args.qpos = pose_qpos(args.pose)

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    r = mujoco.Renderer(m, H, W)

    for cam, real_path, q in [("ext_cam", args.side, args.qpos), ("wrist_cam", args.wrist, args.wrist_qpos)]:
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        sim = r.render()
        real = load_640x480(real_path)
        blend = (real.astype(float) * 0.5 + sim.astype(float) * 0.5).astype(np.uint8)
        Image.fromarray(np.hstack([real, sim, blend])).save(HERE / f"compare_{cam}.png")
        pts = {"ext_cam": [("table near", 320, 440), ("table mid", 150, 250), ("table far", 500, 175)],
               "wrist_cam": [("table", 320, 100), ("table side", 100, 250)]}[cam]
        for name, x, y in pts:
            print(f"{cam:9s} {name:11s} real {patch(real, x, y)}  sim {patch(sim, x, y)}")
    r.close()


if __name__ == "__main__":
    main()
