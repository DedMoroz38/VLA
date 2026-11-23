"""Run a trained policy (SmolVLA / ACT, real-arm units) on the real SO-101, on the same
fixed start poses as the sim eval, so real and sim success rates compare directly.

    cd sim
    CKPT=outputs/train/smolvla_cube/020000/pretrained_model
    python run_policy_real.py $CKPT --split test --shadow      # 1. dry run: the arm holds still
    python run_policy_real.py $CKPT --split test --mix         #    which camera's sim-to-real gap matters
    python run_policy_real.py $CKPT --split test --mix --filter both   # does filtering the real frames help?
    python run_policy_real.py $CKPT --split test --step --sim-wrist    # real side camera + sim-twin wrist view
    python run_policy_real.py $CKPT --split test --step        # 2. live, one 1.5 s chunk per SPACE
    python run_policy_real.py $CKPT --split test               # 3. live, continuous

Both modes start each pose the same way: the arm moves slowly to the scene's 'ready' pose
(the start pose of every demo) and the side-camera window shows where the cube must go
(yellow outline, projected from the sim's calibrated ext_cam). Place the real cube on it.

--shadow (nothing moves after 'ready'): the policy runs on the live cameras and the
  arm's measured joints, and its planned gripper path for the next 50 steps (1.5 s) is
  drawn on the side view (red). The same policy is also run on sim images rendered at
  the same arm pose and cube pose (green). If red follows green, the policy sees the real
  scene the way it saw the sim; if red wanders while green heads for the cube, the
  sim-to-real image gap is the problem. The mean gap between the paths is printed (mm).

live: SPACE starts the episode; the policy drives the follower at 33 Hz (the training
  rate). Each command is clipped to --max-step deg (gripper: %) from the measured
  position and to the sim's joint limits. Mark the outcome yourself:

    s      success (cube resting in the container)
    1      fail: never grasped the cube
    2      fail: grasped, but not placed in the container (dropped / missed)
    3      fail: other (stuck, collision, ...)
    x      abort: not counted, the same pose comes again (e.g. you had to intervene)
    SPACE  pause / resume (the arm holds where it is); after --max-seconds it pauses too

  Between poses:  SPACE start   n / b next / previous pose   q quit

Results are appended after every episode to <out>/results.json (re-running continues
with the poses that have no outcome yet; --redo starts over), with a video of what the
policy saw (side | wrist) and the per-step joints and actions (.npz) per episode.

Inference is synchronous: a SmolVLA chunk takes ~0.6 s on the Mac (MPS), so the arm holds
still ~0.6 s every 1.5 s. The loop's overruns are counted in the results.

Safety: keep a hand on the follower's power. q / Ctrl-C move the arm back to 'ready'
before the torque is released (it then rests under its own weight, as after teleop).
"""
import argparse
import datetime
import json
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "calib"))

import joint_units  # noqa: E402
from cameras import Grabber  # noqa: E402
from demo_io import CONTROL_SUBSTEPS, task_for  # noqa: E402
from eval_policy import checkpoint_joint_units, load_policy, wilson_interval  # noqa: E402
from pick_place_task import reset_to_ready, set_eraser_pose, task_object  # noqa: E402
from wrist_lens import WristLens, orient_wrist  # noqa: E402

FOLLOWER_PORT = "/dev/tty.usbmodem5B790809571"
JOINTS = joint_units.JOINTS
NAMES = joint_units.REAL_NAMES
OUTCOMES = {"s": "success", "1": "never grasped", "2": "grasped, not placed in container",
            "3": "other failure"}
W, H = 640, 480
WIN = "policy on the real arm"


# -- the sim side: start pose, cube outline, planned-path drawing, sim images --------------

class SimView:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.k = mujoco.MjData(self.m)                     # scratch data for path kinematics
        self.obj = task_object(self.m)
        reset_to_ready(self.m, self.d)
        self.ready_sim = self.m.key("ready").ctrl[:6].copy()
        self.lo, self.hi = self.m.actuator_ctrlrange[:6].T
        self.cam = self.m.camera("ext_cam").id
        self.fovy = float(self.m.cam_fovy[self.cam])
        self.site = self.m.site("gripperframe").id
        self.half = float(self.m.body(self.obj).pos[2])     # cube half size (rests at z = half)
        self.fps = 1.0 / (CONTROL_SUBSTEPS * self.m.opt.timestep)
        self.renderer = None
        self.lens = None

    def project(self, pts):
        """World points -> side-camera pixels (pinhole, principal point at the centre, as
        calibrated: calib/live_overlay.py)."""
        p, R = self.d.cam_xpos[self.cam], self.d.cam_xmat[self.cam].reshape(3, 3)
        c = (np.asarray(pts, float) - p) @ R
        f = (H / 2) / np.tan(np.radians(self.fovy) / 2)
        return np.stack([W / 2 + f * c[:, 0] / -c[:, 2], H / 2 - f * c[:, 1] / -c[:, 2]], 1)

    def cube_outline(self, x, y, yaw):
        """Pixel polygons of the cube's footprint and top face at a pose."""
        h = self.half
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        xy = sq @ R.T + (x, y)
        return [self.project(np.c_[xy, np.full(4, z)]) for z in (0.0, 2 * h)]

    def gripper_path(self, actions_sim):
        """(T, 6) sim joint targets -> (T, 3) world positions of the gripper frame."""
        out = []
        for q in actions_sim:
            self.k.qpos[:6] = q
            mujoco.mj_kinematics(self.m, self.k)
            out.append(self.k.site_xpos[self.site].copy())
        return np.array(out)

    def _render(self, d, cam):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, H, W)
            self.lens = WristLens()
        self.renderer.update_scene(d, camera=cam)
        img = self.renderer.render().copy()
        return orient_wrist(self.lens.distort(img)) if cam == "wrist_cam" else img

    def images(self, q_sim, pose):
        """What the policy saw in sim at this arm pose and cube pose (RGB, as in training)."""
        reset_to_ready(self.m, self.d)
        self.d.qpos[:6] = q_sim
        set_eraser_pose(self.m, self.d, *pose, object_body=self.obj)
        return {cam: self._render(self.d, cam) for cam in ("ext_cam", "wrist_cam")}

    # The twin: a sim copy of the episode, driven by the same joint commands as the real arm
    # (its position actuators track them like the real servos), with the cube's physics, so a
    # real grasp is a sim grasp too. --sim-wrist feeds its wrist view to the policy.
    def twin_reset(self, q_sim, pose):
        self.t = mujoco.MjData(self.m)
        reset_to_ready(self.m, self.t)
        self.t.qpos[:6] = q_sim
        self.t.ctrl[:6] = q_sim
        set_eraser_pose(self.m, self.t, *pose, object_body=self.obj)

    def twin_step(self, ctrl_sim):
        self.t.ctrl[:6] = ctrl_sim
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(self.m, self.t)

    def twin_wrist(self):
        return self._render(self.t, "wrist_cam")


# -- the real side ----------------------------------------------------------------------

class Follower:
    def __init__(self, port, max_step, jmap, sim):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id="so101_follower"))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit("follower motors don't match its calibration file -- run lerobot-calibrate first")
        self.bus.disable_torque()
        p0 = self.bus.sync_read("Present_Position")
        self.bus.sync_write("Goal_Position", p0)          # torque comes back on holding exactly here
        for k in range(5):                                # the bus sometimes drops one status packet
            try:
                self.robot.configure()
                break
            except ConnectionError as e:
                if k == 4:
                    raise
                print(f"motor bus glitch during configure ({e}); retrying")
                time.sleep(0.3)
        self.max_step = float(max_step)
        # the sim's joint limits in real units: the policy was never shown anything outside them
        a, b = jmap.sim_to_real(sim.lo), jmap.sim_to_real(sim.hi)
        self.lo, self.hi = np.minimum(a, b), np.maximum(a, b)
        print(f"follower {port}: torque on, commands clipped to {self.max_step:g} deg (gripper %) "
              f"per step and to the sim's joint limits")

    def read(self):
        p = self.bus.sync_read("Present_Position", num_retry=2)
        return np.array([p[j] for j in JOINTS], np.float32)

    def command(self, target, present):
        """Send a target (real units) clipped to the limits and to max_step from `present`."""
        t = np.clip(np.asarray(target, float), self.lo, self.hi)
        t = np.clip(t, present - self.max_step, present + self.max_step)
        self.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, t)})
        return t

    def move_to(self, target, seconds=3.0, fps=33.0, ui=None):
        """Slow straight-line move in joint space (for the start pose / going home)."""
        start = self.read()
        n = max(int(seconds * fps), 1)
        for i in range(1, n + 1):
            goal = start + (np.asarray(target) - start) * i / n
            self.command(goal, self.read())
            if ui is not None:
                ui()
            time.sleep(1.0 / fps)
        for _ in range(int(0.7 * fps)):                   # let it settle
            self.command(target, self.read())
            time.sleep(1.0 / fps)

    def close(self):
        self.bus.disconnect(disable_torque=True)


class Cameras:
    def __init__(self, side, wrist, filt="none"):
        self.side, self.wrist = Grabber(side), Grabber(wrist)
        self.filt, self.gain = filt, None
        print(f"cameras: side '{self.side.name}', wrist '{self.wrist.name}'"
              + (f", real frames filtered: {filt}" if filt != "none" else ""))

    def raw(self):
        """RGB frames exactly as the policy was trained on: ext 480x640, wrist rotated 90 deg
        anticlockwise to 640x480 (the real lens already has the distortion the sim adds)."""
        ext = cv2.cvtColor(self.side.get()[0], cv2.COLOR_BGR2RGB)
        wr = orient_wrist(cv2.cvtColor(self.wrist.get()[0], cv2.COLOR_BGR2RGB))
        return {"ext_cam": ext, "wrist_cam": wr}

    def read(self):
        imgs = self.raw()
        if self.filt in ("denoise", "both"):
            imgs = {c: cv2.bilateralFilter(im, 7, 30, 5) for c, im in imgs.items()}
        if self.filt in ("colour", "both") and self.gain is not None:
            for c, im in imgs.items():
                g, o = self.gain[c]
                lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32) * g + o
                imgs[c] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return imgs

    def fit_colour(self, sim_imgs):
        """Fixed per-camera colour correction, fitted once per pose: the LAB gain/offset that
        gives this moment's real frame the mean and spread of the sim render of the same scene."""
        if self.filt not in ("colour", "both"):
            return
        self.gain = None
        real = self.read()                                  # denoised first, if that is on too
        self.gain = {}
        for c in real:
            r = cv2.cvtColor(real[c], cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            s = cv2.cvtColor(sim_imgs[c], cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            g = s.std(0) / (r.std(0) + 1e-6)
            self.gain[c] = (g, s.mean(0) - g * r.mean(0))

    def close(self):
        self.side.close()
        self.wrist.close()


# -- policy ---------------------------------------------------------------------------------

class Policy:
    def __init__(self, ckpt, device, n_action_steps, task):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.policy, self.pre, self.post = load_policy(ckpt, self.device, n_action_steps)
        units = checkpoint_joint_units(self.pre)
        if units != "real":
            raise SystemExit(f"checkpoint {ckpt} uses '{units}' joint units; the real arm needs a "
                             "policy trained on real-unit data (export_lerobot.py --joint-units real)")
        self.task = task
        self.n_steps = self.policy.config.n_action_steps
        print(f"policy: {self.policy.config.type} on {self.device}, {self.n_steps} actions per chunk, "
              f"task \"{task}\"")

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _batch(self, state, imgs):
        from lerobot.policies.utils import prepare_observation_for_inference
        obs = {"observation.state": np.asarray(state, np.float32)}
        for cam, img in imgs.items():
            obs[f"observation.images.{cam}"] = np.ascontiguousarray(img)
        return self.pre(prepare_observation_for_inference(obs, self.torch.device(self.device),
                                                          self.task, "so101"))

    def step(self, state, imgs):
        """One action (real units), as in eval_policy.py; refills the chunk when empty."""
        with self.torch.inference_mode():
            a = self.post(self.policy.select_action(self._batch(state, imgs)))
        return a.squeeze(0).float().cpu().numpy()

    def just_planned(self):
        """Whether the last step() planned a new chunk (its queue is then full but one)."""
        from lerobot.utils.constants import ACTION
        return len(self.policy._queues[ACTION]) == self.n_steps - 1

    def queued(self):
        """The remaining actions of the current chunk, real units (T, 6)."""
        from lerobot.utils.constants import ACTION
        q = list(self.policy._queues[ACTION])
        if not q:
            return np.zeros((0, 6))
        with self.torch.inference_mode():
            a = self.post(self.torch.cat(q, 0))
        return a.float().cpu().numpy().reshape(-1, 6)

    def chunk(self, state, imgs):
        """A fresh full chunk for this observation, real units (T, 6); doesn't execute it."""
        self.reset()
        with self.torch.inference_mode():
            a = self.policy.predict_action_chunk(self._batch(state, imgs))[:, :self.n_steps]
            a = self.post(a)
        return a.float().cpu().numpy().reshape(-1, 6)


# -- display --------------------------------------------------------------------------------

def compose(imgs, lines=(), paths=(), polys=(), sim=None):
    """Side view (with overlays) | wrist view scaled to the same height, as BGR for imshow."""
    ext = cv2.cvtColor(imgs["ext_cam"], cv2.COLOR_RGB2BGR)
    for poly, col in polys:
        cv2.polylines(ext, [np.round(poly).astype(np.int32)], True, col, 2, cv2.LINE_AA)
    for path, col in paths:
        if len(path) and sim is not None:
            px = np.round(sim.project(path)).astype(np.int32)
            cv2.polylines(ext, [px], False, col, 2, cv2.LINE_AA)
            cv2.circle(ext, tuple(px[-1]), 5, col, -1, cv2.LINE_AA)
    wr = cv2.cvtColor(imgs["wrist_cam"], cv2.COLOR_RGB2BGR)
    wr = cv2.resize(wr, (round(wr.shape[1] * H / wr.shape[0]), H))
    out = np.concatenate([ext, wr], 1)
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def key():
    k = cv2.waitKey(1) & 0xFF
    return None if k == 255 else chr(k).lower() if k != 32 else " "


# -- results --------------------------------------------------------------------------------

def load_results(path, redo):
    if path.exists() and not redo:
        return json.loads(path.read_text())
    return {"episodes": []}


def report(results, poses):
    eps = [e for e in results["episodes"] if e["outcome"] != "abort"]
    n, k = len(eps), sum(e["outcome"] == "success" for e in eps)
    if not n:
        return
    lo, hi = wilson_interval(k, n)
    print(f"\nreal success: {k}/{n} = {100 * k / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%), "
          f"{len(poses) - n} poses left")
    for o in OUTCOMES.values():
        c = sum(e["outcome"] == o for e in eps)
        if c:
            print(f"  {c:4d}  {o}")


# -- main -----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a pretrained_model/ directory (trained in real-arm units)")
    ap.add_argument("--poses", default="eval/cube_poses.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", type=int, nargs="*", help="only these pose ids, in this order")
    ap.add_argument("--shadow", action="store_true",
                    help="dry run: the arm holds 'ready', draw the policy's planned path (real vs sim images)")
    ap.add_argument("--no-sim-compare", action="store_true", help="--shadow without the sim-image path")
    ap.add_argument("--mix", action="store_true",
                    help="camera-mixing test (arm holds still): plan from real/sim side x real/sim wrist")
    ap.add_argument("--samples", type=int, default=5, help="--mix: plans per mix")
    ap.add_argument("--sim-wrist", action="store_true",
                    help="live: feed the policy the wrist view of a sim twin (driven by the same commands, "
                         "cube physics included) instead of the real wrist camera; the side camera stays real")
    ap.add_argument("--filter", choices=("none", "colour", "denoise", "both"), default="none",
                    help="filter the real camera frames: denoise (bilateral) and/or colour-match to the sim "
                         "(LAB gain/offset fitted once per pose against the sim render of the same scene)")
    ap.add_argument("--step", action="store_true",
                    help="live, one chunk at a time: show each new plan (red) and run it only after SPACE")
    ap.add_argument("--follower", default=FOLLOWER_PORT)
    ap.add_argument("--side", default="C920", help="side camera (name fragment or OpenCV index)")
    ap.add_argument("--wrist", default="icspring", help="wrist camera (name fragment or OpenCV index)")
    ap.add_argument("--max-step", type=float, default=10.0,
                    help="max change per command from the measured position, deg (gripper: %%); "
                         "the demos move at most ~4 deg per step")
    ap.add_argument("--max-seconds", type=float, default=45.0, help="pause the episode after this long "
                    "(the sim eval's limit)")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: eval/real/<run name>_<checkpoint step>")
    ap.add_argument("--redo", action="store_true", help="ignore earlier results in --out")
    a = ap.parse_args()

    pose_file = json.loads((HERE / a.poses).read_text())
    poses = [p for p in pose_file["poses"] if a.split is None or p.get("split") == a.split]
    if a.ids:
        by_id = {p["id"]: p for p in pose_file["poses"]}
        poses = [by_id[i] for i in a.ids]
    if not poses:
        raise SystemExit(f"no poses with split '{a.split}' in {a.poses}")
    ckpt = pathlib.Path(a.checkpoint)
    sim = SimView(HERE / pose_file["scene"])
    task = task_for(sim.obj)
    jmap = joint_units.JointMap.load()
    ready = jmap.sim_to_real(sim.ready_sim)
    period = 1.0 / sim.fps

    run_name = f"{ckpt.parent.parent.name}_{ckpt.parent.name}" if ckpt.name == "pretrained_model" else ckpt.name
    out = pathlib.Path(a.out) if a.out else HERE / "eval/real" / run_name
    suffix = ("_mix" if a.mix else "_shadow" if a.shadow else "") + ("_simwrist" if a.sim_wrist and not a.mix else "") \
        + (f"_{a.filter}" if a.filter != "none" else "")
    if suffix:
        out = out.with_name(out.name + suffix)
    out.mkdir(parents=True, exist_ok=True)
    mix_path = out / "mix_results.json"
    mix_rows = json.loads(mix_path.read_text()) if a.mix and mix_path.exists() and not a.redo else []
    res_path = out / "results.json"
    results = load_results(res_path, a.redo)
    results.update({"checkpoint": str(ckpt), "poses": a.poses, "split": a.split, "task": task,
                    "mode": "shadow" if a.shadow else "live", "max_step": a.max_step,
                    "joint_map": jmap.meta.get("offsets_deg")})
    done = {e["pose_id"] for e in results["episodes"] if e["outcome"] != "abort"}

    policy = Policy(ckpt, a.device, a.n_action_steps, task)
    cams = Cameras(a.side, a.wrist, a.filter)
    arm = Follower(a.follower, a.max_step, jmap, sim)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def show(lines, **kw):
        imgs = cams.read()
        cv2.imshow(WIN, compose(imgs, lines, sim=sim, **kw))
        return imgs

    i = next((j for j, p in enumerate(poses) if p["id"] not in done), None)
    if i is None:
        print(f"all {len(poses)} poses already have an outcome in {res_path} (--redo to start over)")
        i = 0
    try:
        print("moving to the 'ready' pose")
        arm.move_to(ready, ui=lambda: show(["moving to 'ready' ..."]) and key())
        while True:
            p = poses[i]
            pose = (p["x"], p["y"], p["yaw_rad"])
            outline = [(poly, (0, 255, 255)) for poly in sim.cube_outline(*pose)]
            tag = (f"pose {i + 1}/{len(poses)} (id {p['id']}): x {p['x'] * 1000:.0f} y {p['y'] * 1000:.0f} mm, "
                   f"yaw {p['yaw_deg']:+.0f} deg" + ("  [done]" if p["id"] in done else ""))
            # ---- place the cube
            k = None
            while k not in (" ", "q", "n", "b"):
                show([tag, "place the cube on the yellow outline",
                      "SPACE " + ("mix test" if a.mix else "shadow run" if a.shadow else "start")
                      + "   n/b next/prev   q quit"],
                     polys=outline)
                k = key()
                arm.command(ready, arm.read())
            if k == "q":
                break
            if k in "nb":
                i = (i + (1 if k == "n" else -1)) % len(poses)
                continue
            cams.fit_colour(sim.images(jmap.real_to_sim(arm.read()), pose))   # --filter colour / both
            if a.mix:
                row, grid = mix_test(policy, cams, arm, sim, jmap, pose, outline, tag, ready, a.samples)
                row.update({"pose_id": p["id"], "time": datetime.datetime.now().isoformat(timespec="seconds")})
                mix_rows = [r for r in mix_rows if r["pose_id"] != p["id"]] + [row]
                (out / "mix_results.json").write_text(json.dumps(mix_rows, indent=2))
                cv2.imwrite(str(out / f"mix_pose{p['id']:04d}.png"), grid)
                for letter, name, _, _ in MIX:
                    v = row["mixes"][letter]["toward_cube_mm"]
                    print(f"  id {p['id']}  {letter} {name:24s} toward cube {np.mean(v):+5.0f} mm")
                k = None
                while k not in (" ", "n", "q"):
                    g = grid.copy()
                    cv2.putText(g, "SPACE / n next pose   q quit", (8, g.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(WIN, g)
                    k = key()
                    arm.command(ready, arm.read())
                    time.sleep(0.03)
                if k == "q":
                    break
                i = (i + 1) % len(poses)
                continue
            if a.shadow:
                shadow(policy, cams, arm, sim, jmap, pose, outline, tag, ready, not a.no_sim_compare)
                i = (i + 1) % len(poses)
                continue
            # ---- live episode
            ep = episode(policy, cams, arm, sim, jmap, outline, tag, a.max_seconds, period, out, p, a.step,
                         a.sim_wrist)
            ep.update({"sim_wrist": a.sim_wrist, "filter": a.filter})
            print(f"[{i + 1:3d}/{len(poses)}] id {p['id']}: {ep['outcome']}  ({ep['seconds']:.1f} s, "
                  f"{ep['overruns']} slow steps)")
            results["episodes"] = [e for e in results["episodes"] if e["pose_id"] != p["id"]] + [ep]
            results["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
            res_path.write_text(json.dumps(results, indent=2))
            arm.move_to(ready, ui=lambda: show([tag, "back to 'ready' ..."]) and key())
            if ep["outcome"] != "abort":
                done.add(p["id"])
                nxt = next((j for j in list(range(i + 1, len(poses))) + list(range(i))
                            if poses[j]["id"] not in done), None)
                if nxt is None:
                    print("all poses done")
                    break
                i = nxt
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        try:
            print("moving back to 'ready', then releasing the torque")
            arm.move_to(ready, seconds=3.0)
        finally:
            arm.close()
            cams.close()
            cv2.destroyAllWindows()
    if a.mix:
        if mix_rows:
            mix_summary(mix_rows)
            print(f"results -> {out / 'mix_results.json'} and mix_pose*.png")
    elif not a.shadow:
        report(results, poses)
        print(f"results -> {res_path}")


def episode(policy, cams, arm, sim, jmap, outline, tag, max_seconds, period, out, p, step_mode=False,
            sim_wrist=False):
    policy.reset()
    pending = None
    if sim_wrist:
        sim.twin_reset(jmap.real_to_sim(arm.read()), (p["x"], p["y"], p["yaw_rad"]))
    vid_path = out / f"pose{p['id']:04d}.mp4"
    writer = None
    states, actions, sent, times = [], [], [], []
    paused, outcome, overruns = False, None, 0
    t0 = time.perf_counter()
    next_t = t0
    path = np.zeros((0, 3))
    while outcome is None:
        state = arm.read()
        imgs = cams.read()
        if sim_wrist:
            imgs["wrist_cam"] = sim.twin_wrist()
        elapsed = time.perf_counter() - t0
        if not paused and pending is None:
            act = policy.step(state, imgs)
            if policy.just_planned():                          # draw each new chunk's gripper path
                path = sim.gripper_path(jmap.real_to_sim(np.vstack([act, policy.queued()])))
                if step_mode:                                  # show the plan, wait for SPACE
                    pending, paused = act, True
        elif not paused:
            act, pending = pending, None                       # the plan shown before SPACE
        if not paused:
            sent.append(arm.command(act, state))
            if sim_wrist:
                sim.twin_step(jmap.real_to_sim(sent[-1]))
            states.append(state)
            actions.append(act)
            times.append(elapsed)
            if writer is None:
                h, w = imgs["ext_cam"].shape[0], imgs["ext_cam"].shape[1] + round(imgs["wrist_cam"].shape[1] * H / imgs["wrist_cam"].shape[0])
                writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / period, (w, h))
            writer.write(compose(imgs))
            if len(times) * period > max_seconds:              # time actually run, not paused
                paused = True
        else:
            arm.command(state, state)                          # hold where it is
        if paused and pending is not None:
            status = "STEP: red = next 1.5 s plan.  SPACE run it   s/1/2/3 outcome   x abort"
        elif paused:
            status = "PAUSED: s success  1 not grasped  2 not placed  3 other  x abort  SPACE resume"
        else:
            status = "running: SPACE pause   s/1/2/3 outcome   x abort"
        cv2.imshow(WIN, compose(imgs, [tag, f"{elapsed:5.1f} s", status],
                                paths=[(path, (0, 0, 255))], polys=outline, sim=sim))
        k = key()
        if k == " ":
            paused = not paused
        elif k in OUTCOMES:
            outcome = OUTCOMES[k]
        elif k == "x":
            outcome = "abort"
        next_t += period
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        else:
            overruns += now - next_t > period / 2
            next_t = now                                       # don't try to catch up
    if writer is not None:
        writer.release()
    np.savez_compressed(out / f"pose{p['id']:04d}.npz", t=np.array(times), state=np.array(states),
                        action=np.array(actions), sent=np.array(sent))
    return {"pose_id": p["id"], "x": p["x"], "y": p["y"], "yaw_deg": p["yaw_deg"], "outcome": outcome,
            "seconds": round(times[-1] if times else 0.0, 2), "steps": len(times), "overruns": int(overruns),
            "video": vid_path.name, "time": datetime.datetime.now().isoformat(timespec="seconds")}


MIX = [("A", "real side + real wrist", ("real", "real"), (0, 0, 255)),
       ("B", "real side + SIM wrist", ("real", "sim"), (0, 140, 255)),
       ("C", "SIM side + real wrist", ("sim", "real"), (255, 120, 0)),
       ("D", "sim side + sim wrist", ("sim", "sim"), (0, 200, 0))]


def mix_test(policy, cams, arm, sim, jmap, pose, outline, tag, ready, samples):
    """Plan from the 4 mixes of real / sim camera images, same arm state, same moment.
    The arm holds still. Returns (per-mix metrics, image grid)."""
    arm.command(ready, arm.read())
    state = arm.read()
    real = cams.read()
    simimg = sim.images(jmap.real_to_sim(state), pose)
    start = sim.gripper_path(jmap.real_to_sim(state[None]))[0]
    cube = np.array([pose[0], pose[1], sim.half])
    d0 = float(np.linalg.norm(start - cube))
    rows, tiles = {}, []
    for letter, name, (side, wrist), col in MIX:
        imgs = {"ext_cam": (real if side == "real" else simimg)["ext_cam"],
                "wrist_cam": (real if wrist == "real" else simimg)["wrist_cam"]}
        paths, m = [], []
        for k in range(samples):
            cv2.imshow(WIN, compose(cams.read(), [tag, f"mix test: {letter} {name}, plan {k + 1}/{samples} "
                                                  "(the arm stays still)"], polys=outline, sim=sim))
            key()
            path = sim.gripper_path(jmap.real_to_sim(policy.chunk(state, imgs)))
            paths.append(path)
            m.append((d0 - float(np.linalg.norm(path[-1] - cube)),          # how much closer to the cube
                      float(np.linalg.norm(path - cube, axis=1).min()),     # closest approach
                      float(np.linalg.norm(path[-1] - start))))             # how far the gripper goes
        m = np.array(m) * 1000
        rows[letter] = {"name": name, "toward_cube_mm": m[:, 0].tolist(), "closest_mm": m[:, 1].tolist(),
                        "travel_mm": m[:, 2].tolist()}
        lines = [f"{letter}: {name}",
                 f"toward cube {m[:, 0].mean():+.0f} mm (sd {m[:, 0].std():.0f}), travel {m[:, 2].mean():.0f} mm",
                 f"closest approach {m[:, 1].mean():.0f} mm (start {d0 * 1000:.0f})"]
        tiles.append(compose(imgs, lines, paths=[(p, col) for p in paths], polys=outline, sim=sim))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return {"start_to_cube_mm": d0 * 1000, "mixes": rows}, grid


def mix_summary(all_rows):
    print(f"\nmix test over {len(all_rows)} pose(s): how much closer the 1.5 s plan takes the gripper "
          "to the cube (mm; mean over plans and poses)")
    for letter, name, _, _ in MIX:
        v = [x for r in all_rows for x in r["mixes"][letter]["toward_cube_mm"]]
        t = [x for r in all_rows for x in r["mixes"][letter]["travel_mm"]]
        print(f"  {letter}  {name:24s} toward cube {np.mean(v):+6.0f} mm   travel {np.mean(t):5.0f} mm")


def shadow(policy, cams, arm, sim, jmap, pose, outline, tag, ready, compare):
    """The arm holds 'ready'; keep re-planning from the live images and draw the plans."""
    gaps = []
    while True:
        state = arm.read()
        imgs = cams.read()
        real_plan = sim.gripper_path(jmap.real_to_sim(policy.chunk(state, imgs)))
        paths = [(real_plan, (0, 0, 255))]
        line = "red: plan from the real cameras"
        if compare:
            sim_plan = sim.gripper_path(jmap.real_to_sim(policy.chunk(state, sim.images(jmap.real_to_sim(state), pose))))
            paths.append((sim_plan, (0, 200, 0)))
            gaps.append(float(np.linalg.norm(real_plan - sim_plan, axis=1).mean()))
            line += f", green: from sim images; gap {gaps[-1] * 1000:.0f} mm (mean {np.mean(gaps) * 1000:.0f})"
        cube = np.array([pose[0], pose[1], sim.half])
        reach = float(np.linalg.norm(real_plan - cube, axis=1).min())
        lines = [tag, line, f"real plan's closest approach to the cube: {reach * 1000:.0f} mm",
                 "SPACE / n next pose   q back"]
        for _ in range(3):                                     # keep the view live between plans
            cv2.imshow(WIN, compose(cams.read(), lines, paths=paths, polys=outline, sim=sim))
            k = key()
            arm.command(ready, arm.read())
            if k in (" ", "n", "q"):
                if gaps:
                    print(f"{tag}: real vs sim plan gap mean {np.mean(gaps) * 1000:.0f} mm over {len(gaps)} plans; "
                          f"last real plan's closest approach to the cube {reach * 1000:.0f} mm")
                return
            time.sleep(0.03)


if __name__ == "__main__":
    main()
