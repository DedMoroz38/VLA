"""Keyboard-controlled Cartesian teleoperation for the SO-101 pick-and-place
scene, with demonstration recording.

Renders MuJoCo offscreen and displays frames in a Tkinter window -- same
approach as view_so101.py -- to avoid GLFW/NSWindow, which is broken on this
machine's macOS Tahoe. Full screen: a large OPERATOR view (from the container side, 45 deg
down, looking across at the robot, container in the foreground; arrow keys orbit
it, -/= zoom) plus the two recorded cameras (wrist_cam,
and ext_cam = the real side camera) small on the right. The operator view is display
only: demos store the sim state, and the dataset images are rendered from wrist_cam
and ext_cam at export, exactly as before.

Two ways to drive the arm:
  - keyboard (default): Cartesian jogging with the keys below
  - the real LEADER arm (--leader): the sim arm follows the leader's joints, converted
    with joint_units.py exactly as the dataset export converts them, so sim demos move
    like real teleop. After every reset the sim arm waits in the scene's start pose
    until you bring the leader close to it (the status line shows how far off each
    joint is), then follows, rate-limited so it never jumps. P only records while
    following. Needs the lerobot env (it talks to the leader over USB).

Fixed pose list (--poses eval/cube_poses.json, from make_cube_poses.py): steps through
the 'train' poses in order. A successful saved demo marks its pose done and moves on to
the next pose not done yet; a failed one is saved to <demo dir>/failed/ and the scene
resets to the same pose. Restarting resumes at the
first pose without a successful demo (found from the demos already saved). N / B go to
the next / previous pose by hand. Each demo stores its pose id.

Controls:
    W / S       move gripper +X / -X (forward / back)
    A / D       move gripper +Y / -Y (left / right)
    R / F       move gripper +Z / -Z (up / down)
    T / G       pitch gripper up / down
    Q / E       roll gripper left / right (wrist_roll joint)
    C / V       close / open gripper
    P           start recording / stop and save
    X           discard the recording in progress
    N / B       next / previous pose (--poses), or a new random pose (--randomize)
    Space / 0   reset to the "ready" pose (discards a recording in progress)
    Escape      quit

Recording writes one .npz per episode into demos/ -- see demo_io.py for the
format, and replay_demo.py to play one back and verify it.

Works with any of the pick-place scenes: the object is whichever of 'eraser'
or 'cube' the scene has. The cube scene (pick_place_scene_real.xml) has the
real 220 x 130 mm container (sim/container/build_container.py); the success test
reads its position and size from the scene.

The eraser's start pose can be set or randomised, which is how a demonstration
set covering different positions and rotations gets collected. Space re-runs
the *same* pose so a botched take can be retried; N draws a new one.

Usage:
    python pick_place_teleop.py
    # cube baseline demos, driven by the leader arm (lerobot env):
    /opt/anaconda3/envs/lerobot/bin/python pick_place_teleop.py --poses eval/cube_poses.json --leader
    python pick_place_teleop.py --poses eval/cube_poses.json          # same poses, keyboard
    python pick_place_teleop.py --randomize            # random pose, new each N
    python pick_place_teleop.py --randomize --seed 7   # reproducible sequence
    python pick_place_teleop.py --eraser 0.22 0.05 45  # x y yaw_degrees
    python pick_place_teleop.py --render-every 1    # smoother view, slower sim
    python pick_place_teleop.py --scene path/to/scene.xml
"""
import argparse
import json
import pathlib
import time
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageTk

import joint_units
from demo_io import CONTROL_SUBSTEPS, DemoRecorder, next_demo_path
from ik_solver import ArmIK
from wrist_lens import orient_wrist
from pick_place_task import (
    SCENE_PATH, eraser_pose_warnings, get_eraser_pose, has_container,
    object_in_container, reset_to_ready, sample_eraser_pose, set_eraser_pose,
    task_object,
)

CAM_WIDTH, CAM_HEIGHT = 320, 240          # windowed (--windowed) camera tiles
# operator view: a free camera on the container side, 45 deg down, looking across at the robot
OP_LOOKAT, OP_AZIMUTH, OP_ELEVATION, OP_DISTANCE = (0.13, -0.06, 0.03), 90.0, -45.0, 0.70
STATUS_PX = 230                              # screen height kept for the status text
DEMO_DIR = pathlib.Path(__file__).parent / "demos"
LEADER_PORT = "/dev/tty.usbmodem5B790804261"      # as calib/live_overlay.py
ENGAGE_TOL_DEG = 15.0     # leader within this of the start pose on every arm joint -> sim follows
LEADER_MAX_RATE = 2.5     # rad/s: the sim target never moves faster than this toward the leader

# Offscreen rendering costs ~16 ms per camera, versus ~0.05 ms for a whole
# control step of physics, so drawing every tick would pin the sim at roughly
# 0.4x real time and make teleoperating feel like wading. Control and
# recording run every tick; the display refreshes every RENDER_EVERY-th one.
RENDER_EVERY = 2

LIN_SPEED = 0.12       # m/s, Cartesian jog speed
PITCH_SPEED = 1.0      # rad/s
ROLL_SPEED = 1.5       # rad/s
GRIPPER_SPEED = 1.0    # ctrl units/s (gripper ctrlrange spans ~1.92)

HELP_TEXT = (
    "W/S +X/-X   A/D +Y/-Y   R/F +Z/-Z (up/down)\n"
    "T/G pitch up/down   Q/E roll\n"
    "C close gripper   V open gripper\n"
    "P start recording / stop and save   X discard recording\n"
    "N/B next/previous pose   Space/0 reset to ready pose   Esc quit"
)
HELP_TEXT_LEADER = (
    "LEADER ARM drives the sim arm (after each reset: match the start pose to engage)\n"
    "P start recording / stop and save   X discard recording\n"
    "N/B next/previous pose   Space/0 reset (same pose)   Esc quit\n"
    "arrows orbit the big view, -/= zoom (display only, not recorded)"
)


class Leader:
    """The real leader arm (torque off, moved by hand), read over USB in a background
    thread so the ~10 ms bus read never stalls the sim loop; read() returns the newest."""

    def __init__(self, port):
        import threading
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        self.arm = SO101Leader(SO101LeaderConfig(port=port, id="so101_leader"))
        self.arm.bus.connect()
        if not self.arm.bus.is_calibrated:
            self.arm.bus.disconnect()
            raise SystemExit("leader motors don't match its calibration file -- run lerobot-calibrate first")
        self.arm.configure()
        self.last = self.arm.get_action()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"leader arm on {port}")

    def _loop(self):
        while self.running:
            try:
                a = self.arm.get_action()
                with self.lock:
                    self.last = a
            except (ConnectionError, RuntimeError) as e:
                print(f"leader read glitch ({e}); keeping the last reading")
            time.sleep(0.005)

    def read(self):
        """{'shoulder_pan.pos': deg, ...}: the newest reading."""
        with self.lock:
            return dict(self.last)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.arm.bus.disconnect()


def load_pose_list(path, split):
    """The poses of one split from a make_cube_poses.py file, as dicts with id/x/y/yaw_rad."""
    data = json.loads(pathlib.Path(path).read_text())
    poses = [p for p in data["poses"] if p.get("split") == split]
    if not poses:
        raise SystemExit(f"no '{split}' poses in {path}")
    return data, poses


def done_pose_ids(demo_dir, pose_file):
    """(pose id, start pose) of every successful demo from this pose file in demo_dir."""
    done = []
    for f in sorted(pathlib.Path(demo_dir).glob("teleop_*.npz")):
        try:
            with np.load(f) as z:
                meta = json.loads(str(z["metadata_json"]))
        except Exception:
            continue
        if meta.get("success") and meta.get("pose_file") == pose_file and "pose_id" in meta:
            done.append((meta["pose_id"], meta.get("cube_start", meta.get("eraser_start"))))
    return done


def done_ids_matching(done_meta, poses):
    """Pose ids whose saved demo started where the pose file says (within 2 mm), so a
    regenerated pose file never counts demos recorded on the old poses."""
    by_id = {p["id"]: p for p in poses}
    ok = set()
    for pid, start in done_meta:
        p = by_id.get(pid)
        if p and start and abs(start["x"] - p["x"]) < 0.002 and abs(start["y"] - p["y"]) < 0.002:
            ok.add(pid)
    return ok


class TeleopViewer:
    def __init__(self, model_path, render_every=RENDER_EVERY, eraser_pose=None,
                 randomize=False, seed=None, poses_file=None, split="train",
                 leader_port=None, demo_dir=None, fullscreen=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.model)
        self.fullscreen = fullscreen
        # real-look scene (calibrated wrist lens): show the wrist view as the dataset stores it
        cid = self.model.camera("wrist_cam").id
        self.wrist_portrait = self.model.cam_sensorsize[cid][0] > 0
        self.lens = None
        if self.wrist_portrait:
            try:
                from wrist_lens import WristLens
                self.lens = WristLens()
                self.lens.distort(np.zeros((48, 64, 3), np.uint8))   # needs OpenCV (lerobot env)
            except ImportError:
                self.lens = None
                print("note: no OpenCV in this env -- the wrist view is shown rotated but without "
                      "the lens distortion (the dataset export applies it)")
        self.op_cam = mujoco.MjvCamera()
        self.op_cam.lookat[:] = OP_LOOKAT
        self.op_cam.azimuth, self.op_cam.elevation, self.op_cam.distance = OP_AZIMUTH, OP_ELEVATION, OP_DISTANCE
        self.scene_path = model_path
        self.obj = task_object(self.model)          # 'eraser' or 'cube'
        self.has_tray = has_container(self.model)
        self.demo_dir = pathlib.Path(demo_dir) if demo_dir else (
            DEMO_DIR / "cube" if self.obj == "cube" else DEMO_DIR)

        # leader arm: the sim follows it once it has been brought to the start pose
        self.leader = Leader(leader_port) if leader_port else None
        self.jmap = joint_units.JointMap.load() if self.leader else None
        self.engaged = False

        # fixed pose list (make_cube_poses.py)
        self.pose_file = None
        self.pose_list = None
        if poses_file:
            _, self.pose_list = load_pose_list(poses_file, split)
            self.pose_file = pathlib.Path(poses_file).name
            self.split = split
            self.done = done_ids_matching(done_pose_ids(self.demo_dir, self.pose_file), self.pose_list)
            self.pose_idx = next((i for i, p in enumerate(self.pose_list) if p["id"] not in self.done), 0)
            eraser_pose = self._list_pose()
            print(f"{len(self.pose_list)} '{split}' poses from {poses_file}: {len(self.done)} already done, "
                  f"starting at #{self.pose_idx + 1} (id {self.pose_list[self.pose_idx]['id']}); "
                  f"demos -> {self.demo_dir}")

        # One control step = one recorded frame. Driving the Tk timer off the
        # same period keeps the sim running at real time rather than the 0.9x
        # a fixed 30 Hz timer gave against a 30 ms control period.
        self.substeps = CONTROL_SUBSTEPS
        self.control_dt = self.substeps * self.model.opt.timestep
        self.tick_ms = max(1, int(1000 * self.control_dt))
        self.render_every = max(1, int(render_every))

        self.recorder = DemoRecorder(self.model, object_body=self.obj)
        self.recording = False
        self.message = ""
        self.ticks = 0
        self._deadline = None
        self._rt_anchor = None
        self._realtime_factor = 1.0

        # A fixed pose per take, so Space retries the same configuration rather
        # than silently moving the goalposts mid-attempt. N draws a new one.
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.eraser_pose = eraser_pose
        if self.eraser_pose is None and randomize:
            self.eraser_pose = sample_eraser_pose(self.rng, check_tray=self.has_tray, object_body=self.obj)

        self.reset()
        self.pressed = set()

        self.root = tk.Tk()
        self.root.title(f"SO-101 pick-and-place teleop — {pathlib.Path(model_path).name}")
        self.root.configure(bg="black")
        self._setup_display()

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.TOP)

        self.status = tk.Label(self.root, justify="left", anchor="w", font=("Menlo", 13))
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))

        self.rec_status = tk.Label(self.root, justify="left", anchor="w", font=("Menlo", 12))
        self.rec_status.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 0))

        tk.Label(self.root, text=HELP_TEXT_LEADER if self.leader else HELP_TEXT, justify="left", fg="gray",
                 font=("Menlo", 11)).pack(side=tk.TOP, anchor="w", padx=8, pady=(4, 8))

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.focus_force()

        self._tk_img = None

    # -- recording ---------------------------------------------------------

    def _list_pose(self):
        p = self.pose_list[self.pose_idx]
        return (p["x"], p["y"], p["yaw_rad"])

    def goto_pose(self, step):
        """Move through the pose list by `step` (+1 next, -1 previous) and reset to it."""
        self.pose_idx = (self.pose_idx + step) % len(self.pose_list)
        self.eraser_pose = self._list_pose()
        self.reset()
        p = self.pose_list[self.pose_idx]
        self.message = (f"pose #{self.pose_idx + 1}/{len(self.pose_list)} (id {p['id']})"
                        + ("  [already done]" if p["id"] in self.done else ""))
        print(self.message)

    def next_undone_pose(self):
        n = len(self.pose_list)
        for k in range(1, n + 1):
            i = (self.pose_idx + k) % n
            if self.pose_list[i]["id"] not in self.done:
                self.goto_pose(k)
                return
        self.message = f"all {n} poses have a successful demo"
        print(self.message)

    def start_recording(self):
        if self.leader and not self.engaged:
            self.message = "bring the leader to the start pose first (the sim arm isn't following yet)"
            return
        self.recorder.clear()
        self.recording = True
        # Read the pose off the sim rather than trusting self.eraser_pose: the
        # eraser may have been nudged since the reset that placed it.
        self.episode_start_pose = get_eraser_pose(self.model, self.data, self.obj)
        self.message = (f"recording — place the {self.obj} in the container, then press P"
                        if self.has_tray else "recording — press P to stop and save")

    def stop_recording(self):
        """Stop and write the episode out."""
        self.recording = False
        if len(self.recorder) == 0:
            self.message = "nothing recorded"
            return
        success = object_in_container(self.model, self.data, self.obj)
        x, y, yaw = self.episode_start_pose
        extra = {f"{self.obj}_start": {
            "x": x, "y": y, "yaw_rad": yaw, "yaw_deg": float(np.degrees(yaw))},
            "control": "leader" if self.leader else "keyboard"}
        if self.pose_list is not None:
            extra.update(pose_file=self.pose_file, pose_split=self.split,
                         pose_id=self.pose_list[self.pose_idx]["id"])
        # failed takes go to a failed/ subfolder, so demo_dir holds only good demos
        out_dir = self.demo_dir if (success or not self.has_tray) else self.demo_dir / "failed"
        path = self.recorder.save(
            next_demo_path(out_dir, prefix="teleop"),
            scene_path=self.scene_path, success=success, source="teleop",
            # Also recoverable from obj_pos[0]/obj_quat[0]; stored explicitly so
            # a demo set can be surveyed without loading every episode's arrays.
            extra_metadata=extra)
        outcome = (("SUCCESS" if success else f"{self.obj} NOT in container")
                   if self.has_tray else "no success test (no tray)")
        where = "" if out_dir == self.demo_dir else "failed/"
        self.message = f"saved {where}{path.name} — {len(self.recorder)} frames, {outcome}"
        print(self.message)
        self.recorder.clear()
        saved = self.message
        if self.pose_list is not None and success:
            self.done.add(self.pose_list[self.pose_idx]["id"])
            self.next_undone_pose()                  # resets to the next pose
        else:
            self.reset()                             # same pose again
            self.message = "retry this pose" if self.has_tray and not success else "reset"
        self.message = f"{saved}  ->  {self.message}"

    def discard_recording(self):
        if self.recording or len(self.recorder):
            self.message = f"discarded {len(self.recorder)} frames"
        self.recording = False
        self.recorder.clear()

    # -- sim ---------------------------------------------------------------

    def reset(self):
        if self.recording:
            self.discard_recording()
            self.message = "reset — recording in progress was discarded"
        reset_to_ready(self.model, self.data)
        if self.eraser_pose is not None:
            set_eraser_pose(self.model, self.data, *self.eraser_pose, object_body=self.obj)
        self.q_target = self.data.ctrl[:6].copy()
        self.engaged = False                        # leader: wait for it at the start pose again
        self.episode_start_pose = get_eraser_pose(self.model, self.data, self.obj)

    def new_eraser_pose(self):
        """Draw a fresh random object pose and restart from it."""
        if not self.randomize:
            self.message = "N needs --poses or --randomize (or pass --eraser x y yaw)"
            return
        self.eraser_pose = sample_eraser_pose(self.rng, check_tray=self.has_tray, object_body=self.obj)
        self.reset()
        x, y, yaw = self.eraser_pose
        self.message = f"new pose: x={x:.3f} y={y:+.3f} yaw={np.degrees(yaw):+.0f}°"
        print(self.message)

    def _on_key_press(self, event):
        k = event.keysym.lower()
        if k == "escape":
            self.close()
            return
        if k in ("space", "0"):
            self.reset()
            return
        if k == "p":
            self.stop_recording() if self.recording else self.start_recording()
            return
        if k == "x":
            self.discard_recording()
            return
        view = {"left": ("azimuth", -5), "right": ("azimuth", 5), "up": ("elevation", -3),
                "down": ("elevation", 3), "minus": ("distance", 0.05), "equal": ("distance", -0.05)}
        if k in view:                               # operator view only (display, not recorded)
            attr, dv = view[k]
            setattr(self.op_cam, attr, getattr(self.op_cam, attr) + dv)
            self.op_cam.elevation = float(np.clip(self.op_cam.elevation, -89, -5))
            self.op_cam.distance = float(np.clip(self.op_cam.distance, 0.25, 2.0))
            return
        if k in ("n", "b"):
            if self.recording:
                self.message = "stop (P) or discard (X) the recording first"
            elif self.pose_list is not None:
                self.goto_pose(+1 if k == "n" else -1)
            elif k == "n":
                self.new_eraser_pose()
            return
        if self.leader:
            return                                  # the leader drives the arm; motion keys are off
        self.pressed.add(k)

    def _on_key_release(self, event):
        self.pressed.discard(event.keysym.lower())

    def _axis(self, pos_key, neg_key, speed):
        return speed * (float(pos_key in self.pressed) - float(neg_key in self.pressed))

    def _leader_offsets_deg(self, reading):
        """Leader minus the start pose, per arm joint, in leader (LeRobot) degrees."""
        start = self.jmap.sim_to_real(self.model.key("ready").ctrl[:6] if self.model.nkey else np.zeros(6))
        now = np.array([reading[f"{j}.pos"] for j in joint_units.JOINTS])
        return (now - start)[:5]

    def _leader_action(self):
        """Follow the leader: its joints -> sim radians, clipped to the actuator ranges, and
        approached at most LEADER_MAX_RATE per second. Until engaged, hold the start pose."""
        reading = self.leader.read()
        if not self.engaged:
            off = self._leader_offsets_deg(reading)
            self.leader_offsets = off
            if np.abs(off).max() < ENGAGE_TOL_DEG:
                self.engaged = True
                self.message = "leader engaged: the sim arm follows it now (P to record)"
            return
        lo, hi = self.model.actuator_ctrlrange[:6].T
        goal = np.clip(self.jmap.real_dict_to_sim(reading), lo, hi)
        step = LEADER_MAX_RATE * self.control_dt
        self.q_target = self.q_target + np.clip(goal - self.q_target, -step, step)

    def _compute_action(self):
        """Update self.q_target from the keys currently held down (or the leader arm)."""
        if self.leader:
            self._leader_action()
            return
        lin_vel = np.array([
            self._axis("w", "s", LIN_SPEED),
            self._axis("a", "d", LIN_SPEED),
            self._axis("r", "f", LIN_SPEED),
        ])
        pitch_rate = self._axis("t", "g", PITCH_SPEED)
        roll_rate = self._axis("e", "q", ROLL_SPEED)
        grip_rate = self._axis("v", "c", GRIPPER_SPEED)

        if np.any(lin_vel) or pitch_rate:
            dq = self.ik.step(self.data, lin_vel, pitch_rate, q=self.q_target[:4])
            for i in range(4):
                lo, hi = self.model.actuator_ctrlrange[i]
                self.q_target[i] = float(
                    np.clip(self.q_target[i] + dq[i] * self.control_dt, lo, hi))

        for idx, rate in ((4, roll_rate), (5, grip_rate)):
            lo, hi = self.model.actuator_ctrlrange[idx]
            self.q_target[idx] = float(
                np.clip(self.q_target[idx] + rate * self.control_dt, lo, hi))

    def _tick(self):
        began = time.perf_counter()
        if self._deadline is None:
            self._deadline = began
        if self._rt_anchor is None:
            self._rt_anchor = (began, self.data.time)
        else:
            # Averaged over ~1 s: individual ticks alternate between cheap and
            # render-bound, so a per-tick ratio just flickers.
            wall, sim = began - self._rt_anchor[0], self.data.time - self._rt_anchor[1]
            if wall >= 1.0:
                self._realtime_factor = sim / wall
                self._rt_anchor = (began, self.data.time)

        # Same order as pick_place_task.PolicyRunner._apply, so a teleop demo
        # and a scripted one are the same kind of object: decide the action
        # from the current state, log (state, action), apply, then resync the
        # derived quantities with the integrated qpos.
        self._compute_action()
        if self.recording:
            self.recorder.capture(self.data, self.q_target)
        self.data.ctrl[:6] = self.q_target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if self.ticks % self.render_every == 0:
            self._render()
            self._update_status()
        self.ticks += 1

        # Schedule against a running deadline rather than "now + one period".
        # A render tick overruns the period, and the cheap ticks either side of
        # it have to absorb that for the sim to track real time -- waiting a
        # fresh full period after each one is what capped teleop below 1x.
        self._deadline += self.control_dt
        delay = self._deadline - time.perf_counter()
        if delay < -self.control_dt:
            # Too far behind to catch up (render is simply slower than the
            # control period); drop the backlog instead of spinning on it.
            self._deadline = time.perf_counter()
            delay = 0.0
        self.root.after(max(0, int(delay * 1000)), self._tick)

    def _setup_display(self):
        """Renderers sized for the window: full screen = big operator view + the two recorded
        cameras stacked on the right; windowed = the two recorded cameras side by side."""
        if not self.fullscreen:
            self.renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)
            return
        self.root.attributes("-fullscreen", True)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        h = min(sh - STATUS_PX, sw // 2) // 2 * 2          # operator view h x 4h/3, side column h/2 x 2h/3
        self.op_size = (h * 4 // 3, h)
        self.side_size = (h * 2 // 3, h // 2)
        rw, rh = min(self.op_size[0], 1280), min(self.op_size[1], 960)   # render size (upscaled if larger)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, rw)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, rh)
        # one renderer for all three views (a second GL context costs ~10 ms per switch);
        # the recorded views are rendered at its size and shrunk for the side column
        self.op_renderer = self.renderer = mujoco.Renderer(self.model, rh, rw)
        if self.wrist_portrait:                              # side column: portrait wrist over landscape ext
            sw = int(h / (4 / 3 + 3 / 4)) // 6 * 6           # multiple of 6: the wrist tile is exactly 4:3
            self.side_size = (sw, h)
            self.side_tiles = {"wrist_cam": (sw, sw * 4 // 3), "ext_cam": (sw, h - sw * 4 // 3)}
        else:
            self.side_tiles = {c: self.side_size for c in ("wrist_cam", "ext_cam")}

    def _recorded_view(self, cam, size=None):
        """A recorded camera exactly as the dataset gets it: for the real-look wrist_cam, the real
        lens distortion and the 90 deg rotation (wrist_lens.py), as export_lerobot.py does.
        size: final (w, h) to shrink to; the wrist view is shrunk before the (costly) distortion."""
        self.renderer.update_scene(self.data, camera=cam)
        img = self.renderer.render()
        if cam == "wrist_cam" and self.wrist_portrait:
            if size is not None:                        # landscape 4:3 at the tile's size, then distort
                img = np.asarray(Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR))
            if self.lens is not None:
                img = self.lens.distort(img)
            return orient_wrist(img)
        if size is not None:
            img = np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))
        return img

    def _render(self):
        if not self.fullscreen:
            frames = [self._recorded_view(c) for c in ("wrist_cam", "ext_cam")]
            h = frames[1].shape[0]
            frames[0] = np.asarray(Image.fromarray(frames[0]).resize(
                (round(frames[0].shape[1] * h / frames[0].shape[0]), h)))
            img = Image.fromarray(np.concatenate(frames, axis=1))
        else:
            self.op_renderer.update_scene(self.data, self.op_cam)
            op = Image.fromarray(self.op_renderer.render())
            if op.size != self.op_size:
                op = op.resize(self.op_size, Image.BILINEAR)
            # the small recorded views cost ~20 ms each: refresh one per frame, alternating
            if not hasattr(self, "_side"):
                self._side, self._side_i = {}, 0
            cams = ("wrist_cam", "ext_cam")
            todo = cams if len(self._side) < 2 else (cams[self._side_i % 2],)
            for cam in todo:
                self._side[cam] = self._recorded_view(cam, self.side_tiles[cam])
            self._side_i += 1
            img = Image.new("RGB", (self.op_size[0] + self.side_size[0], self.op_size[1]))
            img.paste(op, (0, 0))
            img.paste(Image.fromarray(np.concatenate([self._side[c] for c in cams], axis=0)), (self.op_size[0], 0))
        self._tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk_img)

    def _update_status(self):
        pos = self.ik.site_pos(self.data)
        lo, hi = self.model.actuator_ctrlrange[5]
        grip_frac = (self.q_target[5] - lo) / (hi - lo)
        ex, ey, eyaw = self.episode_start_pose
        lines = (f"gripper xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                 f"roll={self.q_target[4]:+.2f}  "
                 f"gripper={'OPEN' if grip_frac > 0.5 else 'CLOSED'} "
                 f"({grip_frac * 100:3.0f}%)\n"
                 f"{self.obj} start  x={ex:.3f} y={ey:+.3f} "
                 f"yaw={np.degrees(eyaw):+.0f}°")
        if self.pose_list is not None:
            p = self.pose_list[self.pose_idx]
            lines += (f"   pose #{self.pose_idx + 1}/{len(self.pose_list)} (id {p['id']})"
                      f"{'  DONE' if p['id'] in self.done else ''}   {len(self.done)} done")
        if self.leader:
            if self.engaged:
                lines += "\nLEADER: following"
            else:
                off = getattr(self, "leader_offsets", np.zeros(5))
                names = ("pan", "lift", "elbow", "w_flex", "w_roll")
                lines += (f"\nLEADER: bring it to the start pose (every joint within {ENGAGE_TOL_DEG:.0f}°); "
                          "off by: " + "  ".join(
                              f"{n} {o:+.0f}°" + ("" if abs(o) < ENGAGE_TOL_DEG else "!")
                              for n, o in zip(names, off)))
        self.status.configure(text=lines)

        if self.recording:
            secs = len(self.recorder) * self.control_dt
            prefix = f"● REC  {len(self.recorder):4d} frames ({secs:5.1f}s)"
            colour = "red"
        else:
            prefix = "○ idle  press P to record"
            colour = "gray"
        if self.has_tray:
            task = "task SOLVED" if object_in_container(self.model, self.data, self.obj) else "task not solved"
        else:
            task = "no tray"
        self.rec_status.configure(
            text=f"{prefix}   {task}   "
                 f"[{self._realtime_factor:.2f}x realtime]   {self.message}",
            fg=colour)

    def close(self):
        if self.recording:
            self.stop_recording()
        self.renderer.close()
        if self.leader:
            self.leader.close()
        self.root.destroy()

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None,
                    help="default: the --poses file's scene, else the eraser scene")
    ap.add_argument("--poses", default=None,
                    help="fixed pose list (make_cube_poses.py), stepped through in order")
    ap.add_argument("--split", default="train", help="which poses of --poses to use")
    ap.add_argument("--leader", nargs="?", const=LEADER_PORT, default=None, metavar="PORT",
                    help=f"drive the sim arm with the real leader arm (default port {LEADER_PORT})")
    ap.add_argument("--windowed", action="store_true",
                    help="small window with just the two recorded cameras (the old layout)")
    ap.add_argument("--demo-dir", default=None,
                    help="where demos are saved (default: demos/cube for the cube scene, else demos)")
    ap.add_argument("--render-every", type=int, default=RENDER_EVERY,
                    help="refresh the display every Nth control step "
                         "(1 = smoothest view, slower-than-realtime sim)")
    ap.add_argument("--eraser", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                    help="start the eraser at this pose instead of the scene's")
    ap.add_argument("--randomize", action="store_true",
                    help="start the eraser at a random reachable pose; N draws "
                         "a new one, Space retries the current one")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --randomize, to repeat a pose sequence")
    args = ap.parse_args()
    if args.scene is None:
        args.scene = SCENE_PATH
        if args.poses:
            rel = json.loads(pathlib.Path(args.poses).read_text()).get("scene")
            if rel:
                args.scene = str(pathlib.Path(__file__).parent / rel)

    pose = None
    if args.eraser:
        x, y, yaw_deg = args.eraser
        pose = (x, y, float(np.radians(yaw_deg)))
        for problem in eraser_pose_warnings(*pose):
            print(f"warning: eraser pose {problem}")

    viewer = TeleopViewer(args.scene, render_every=args.render_every,
                          eraser_pose=pose, randomize=args.randomize,
                          seed=args.seed, poses_file=args.poses, split=args.split,
                          leader_port=args.leader, demo_dir=args.demo_dir,
                          fullscreen=not args.windowed)
    x, y, yaw = viewer.episode_start_pose
    print(f"{viewer.obj} starts at x={x:.3f} y={y:+.3f} yaw={np.degrees(yaw):+.0f}°")
    viewer.run()


if __name__ == "__main__":
    main()
