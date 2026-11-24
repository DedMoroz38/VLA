"""Where do the jaws grip the cube? Grasp placement in the demos and in policy rollouts.

    cd sim
    python analyze_grasps.py                                     # all trimmed cube demos
    python analyze_grasps.py --failures eval/results/smolvla_cube_020000_test.json \
        --ckpt outputs/train/smolvla_cube/020000/pretrained_model  # + re-run the failed eval poses

For every episode the moment of lift is found (the cube first 5 mm above the table) and the
exact sim state there is restored. The contacts between the cube and the two jaws (fixed jaw =
body 'gripper', moving jaw = 'moving_jaw_so101_v1') are read in the cube's own frame
(centre = 0, faces at +-20 mm):

    side    how far the grip centre is from the middle of the gripped faces, along the face,
            horizontally (0 = centred; +-20 mm = at the cube's edge)  <- "barely to one side"
    height  how high on the faces (0 = mid-height, +20 = top edge)
    tilt    angle between the jaw-to-jaw line and the faces' normal (0 = square grip,
            45 = across two corners)

Writes eval/grasps/grasp_map.png (every grip's centre on the face, demos vs rollouts) and
eval/grasps/grasps.json. With --failures, each failed pose from that eval is re-run (up to
--tries times, new noise each time) until it fails again; the failing run's video and its
grasp go to eval/grasps/.
"""
import argparse
import glob
import json
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from demo_io import CONTROL_SUBSTEPS, STATE_SPEC, load_demo  # noqa: E402

SCENE = HERE / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
HALF = 0.02
LIFT = 0.005
JAWS = {"fixed": "gripper", "moving": "moving_jaw_so101_v1"}


class Grasp:
    def __init__(self, model):
        self.m = model
        self.cube = model.body("cube").id
        self.cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == self.cube}
        self.jaw_of = {}
        for jaw, body in JAWS.items():
            b = model.body(body).id
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == b:
                    self.jaw_of[g] = jaw

    def at(self, d):
        """Grip measures at this state, or None if both jaws don't touch the cube."""
        pts = {"fixed": [], "moving": []}
        R, c = d.xmat[self.cube].reshape(3, 3), d.xpos[self.cube]
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            other = g2 if g1 in self.cube_geoms else g1 if g2 in self.cube_geoms else None
            if other is None or other not in self.jaw_of:
                continue
            pts[self.jaw_of[other]].append((np.asarray(con.pos) - c) @ R)      # cube frame
        if not pts["fixed"] or not pts["moving"]:
            return None
        f, mv = np.mean(pts["fixed"], 0), np.mean(pts["moving"], 0)
        axis = mv - f
        centre = (mv + f) / 2
        k = int(np.argmax(np.abs(axis[:2])))                 # which pair of side faces is gripped
        side_axis = 1 - k
        tilt = np.degrees(np.arctan2(abs(axis[side_axis]), abs(axis[k])))
        return {"side_mm": float(centre[side_axis] * 1000), "height_mm": float(centre[2] * 1000),
                "tilt_deg": float(tilt), "faces": "xy"[k],
                "n_contacts": {j: len(p) for j, p in pts.items()}}


def lift_index(obj_z, rest_z):
    up = np.nonzero(obj_z > rest_z + LIFT)[0]
    return int(up[0]) if len(up) else None


def analyse_states(model, states, obj_z, grasp):
    """Grip at the lift frame (or a few frames later if contacts flicker)."""
    i = lift_index(obj_z, HALF)
    if i is None:
        return None, None
    d = mujoco.MjData(model)
    for j in range(i, min(i + 10, len(states))):
        mujoco.mj_setState(model, d, states[j], STATE_SPEC)
        mujoco.mj_forward(model, d)
        g = grasp.at(d)
        if g is not None:
            return g, j
    return None, i


def demos(pattern):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    grasp = Grasp(model)
    out = []
    for f in sorted(glob.glob(str(HERE / pattern))):
        demo = load_demo(f)
        g, i = analyse_states(model, demo.mj_state, demo.obj_pos[:, 2], grasp)
        out.append({"file": pathlib.Path(f).name, "frame": i, **(g or {"side_mm": None})})
    return out


def rerun_failures(eval_json, ckpt, tries, outdir):
    """Re-run the failed poses of an eval with the policy until each fails again."""
    import torch
    import eval_policy as ev
    from pick_place_task import reset_to_ready, set_eraser_pose, object_in_container, task_object
    from wrist_lens import orient_wrist
    from demo_io import task_for
    import joint_units
    from lerobot.policies.utils import prepare_observation_for_inference

    res = json.loads(pathlib.Path(eval_json).read_text())
    fails = [e for e in res["results"] if not e["success"]]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    grasp = Grasp(model)
    renderer = mujoco.Renderer(model, 480, 640)
    lens = ev.wrist_lens_for(model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy, pre, post = ev.load_policy(ckpt, device)
    jmap = joint_units.JointMap.load() if ev.checkpoint_joint_units(pre) == "real" else None
    obj = task_object(model)
    task = task_for(obj)
    lo, hi = model.actuator_ctrlrange[:6].T
    bid = model.body(obj).id
    out = []
    for e in fails:
        st = e["eraser_start"]
        pose = (st["x"], st["y"], np.radians(st["yaw_deg"]))
        for t in range(tries):
            torch.manual_seed(1000 + t)
            reset_to_ready(model, data)
            set_eraser_pose(model, data, *pose, object_body=obj)
            policy.reset(); pre.reset(); post.reset()
            states, zs, frames, held = [], [], [], 0
            for step in range(ev.DEFAULT_MAX_STEPS):
                st = np.zeros(mujoco.mj_stateSize(model, STATE_SPEC))
                mujoco.mj_getState(model, data, st, STATE_SPEC)
                states.append(st)
                zs.append(float(data.xpos[bid][2]))
                obs = {"observation.state": np.asarray(data.qpos[:6] if jmap is None else
                                                       jmap.sim_to_real(data.qpos[:6]), np.float32)}
                for cam in ev.CAMERAS:
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render().copy()
                    if cam == "wrist_cam" and lens is not None:
                        img = orient_wrist(lens.distort(img))
                    obs[f"observation.images.{cam}"] = img
                if step % 2 == 0:
                    import cv2
                    wr = obs["observation.images.wrist_cam"]
                    wr = cv2.resize(wr, (round(wr.shape[1] * 480 / wr.shape[0]) // 2 * 2, 480))
                    frames.append(np.concatenate([obs["observation.images.ext_cam"], wr], 1))
                with torch.inference_mode():
                    a = post(policy.select_action(pre(prepare_observation_for_inference(
                        obs, torch.device(device), task, "so101"))))
                a = a.squeeze(0).float().cpu().numpy()
                data.ctrl[:6] = np.clip(jmap.real_to_sim(a) if jmap is not None else a, lo, hi)
                for _ in range(CONTROL_SUBSTEPS):
                    mujoco.mj_step(model, data)
                held = held + 1 if object_in_container(model, data, obj) else 0
                if held >= ev.SUCCESS_HOLD_FRAMES:
                    break
            ok = held >= ev.SUCCESS_HOLD_FRAMES
            g, i = analyse_states(model, np.array(states), np.array(zs), grasp)
            zs = np.array(zs)
            drop = None
            li = lift_index(zs, HALF)
            if li is not None and not ok:
                peak = li + int(np.argmax(zs[li:]))
                below = np.nonzero(zs[peak:] < HALF + 0.003)[0]
                drop = int(peak + below[0]) if len(below) else None
            print(f"  pose {e.get('pose_id', '?')} try {t + 1}: {'success' if ok else 'FAIL'}"
                  + (f", grip side {g['side_mm']:+.1f} mm height {g['height_mm']:+.1f} mm tilt {g['tilt_deg']:.0f} deg"
                     if g else ", no two-jaw grip at lift")
                  + (f", cube back on the table at step {drop}" if drop else ""))
            if not ok:
                vid = outdir / f"fail_pose{e.get('pose_id', 'x')}_try{t + 1}.mp4"
                ev.write_video(vid, frames, 1 / (2 * CONTROL_SUBSTEPS * model.opt.timestep))
                out.append({"pose_id": e.get("pose_id"), "pose": list(pose), "try": t + 1, "video": vid.name,
                            "lift_frame": i, "drop_step": drop, **(g or {"side_mm": None})})
                break
        else:
            print(f"  pose {e.get('pose_id', '?')}: succeeded in all {tries} tries")
    return out


def draw_map(path, demo_rows, roll_rows):
    from PIL import Image, ImageDraw
    s, W = 9, 600                                       # px per mm
    img = Image.new("RGB", (W, W + 70), "white")
    dr = ImageDraw.Draw(img)
    c = W // 2
    px = lambda u, v: (c + u * s, c - v * s)
    dr.rectangle([px(-20, 20), px(20, -20)], outline="black", width=2)
    dr.line([px(0, -20), px(0, 20)], fill=(200, 200, 200))
    dr.line([px(-20, 0), px(20, 0)], fill=(200, 200, 200))
    for rows, col, r in ((demo_rows, (40, 90, 220), 4), (roll_rows, (230, 40, 40), 7)):
        for g in rows:
            if g.get("side_mm") is None:
                continue
            x, y = px(g["side_mm"], g["height_mm"])
            dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    dr.text((10, W + 5), "gripped face, seen from the jaw (40 x 40 mm): each dot = centre of one grip at lift\n"
            "x = sideways along the face (0 = middle, edge at +-20 mm), y = height (top edge +20)\n"
            "blue = demonstrations, red = failed policy rollouts", fill="black")
    img.save(path)


def summary(name, rows):
    g = [r for r in rows if r.get("side_mm") is not None]
    if not g:
        print(f"{name}: no grips found")
        return
    side = np.abs([r["side_mm"] for r in g])
    h = np.array([r["height_mm"] for r in g])
    tilt = np.array([r["tilt_deg"] for r in g])
    print(f"\n{name}: {len(g)} grips ({len(rows) - len(g)} without a two-jaw grip at lift)")
    print(f"  |side offset|  median {np.median(side):.1f} mm, 90th pct {np.percentile(side, 90):.1f} mm, "
          f"max {side.max():.1f} mm   (edge = 20 mm)")
    for lim in (5, 10, 15):
        print(f"     >= {lim:2d} mm off centre: {np.sum(side >= lim):3d} ({100 * np.mean(side >= lim):.0f}%)")
    print(f"  height         median {np.median(h):+.1f} mm (range {h.min():+.1f} .. {h.max():+.1f})")
    print(f"  tilt           median {np.median(tilt):.0f} deg, >= 15 deg: {np.sum(tilt >= 15)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default="demos/cube/trimmed/*.npz")
    ap.add_argument("--failures", help="eval results JSON: re-run its failed poses")
    ap.add_argument("--ckpt", help="policy checkpoint for --failures")
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--out", default="eval/grasps")
    a = ap.parse_args()
    outdir = HERE / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    rows = demos(a.demos)
    summary("demonstrations", rows)
    worst = sorted([r for r in rows if r.get("side_mm") is not None], key=lambda r: -abs(r["side_mm"]))[:8]
    print("  most off-centre:", ", ".join(f"{r['file']} {r['side_mm']:+.0f} mm" for r in worst))
    roll = []
    if a.failures:
        print(f"\nre-running the failed poses of {a.failures}")
        roll = rerun_failures(a.failures, a.ckpt, a.tries, outdir)
        summary("failed policy rollouts", roll)
    draw_map(outdir / "grasp_map.png", rows, roll)
    (outdir / "grasps.json").write_text(json.dumps({"demos": rows, "failed_rollouts": roll}, indent=2))
    print(f"\nwrote {outdir / 'grasp_map.png'} and grasps.json")


if __name__ == "__main__":
    main()
