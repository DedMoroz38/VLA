"""Replay a recorded demonstration and check it was captured correctly.

Two modes, which answer different questions:

  action (default)
      Restore only the episode's *first* MuJoCo state, then re-drive the sim
      with the recorded action sequence and nothing else. The recorded states
      are used purely as ground truth to diverge from. This is the test that
      matters: it proves the actions alone reproduce the demonstration, which
      is exactly what a policy trained on this data will have to do. If the
      control rate, the action semantics or the initial state were logged
      wrong, the trajectory drifts and this reports it.

  state
      mj_setState each recorded frame in turn. Reproduces the episode by
      construction, so it verifies nothing dynamical -- it checks that the
      stored states are self-consistent with the stored poses, and it is the
      mode to use with --view when you just want to watch what was recorded.

Usage:
    python replay_demo.py demos/scripted_000.npz
    python replay_demo.py demos/scripted_000.npz --view
    python replay_demo.py demos/scripted_000.npz --mode state --view
    python replay_demo.py "demos/*.npz"                    # batch check
"""
import argparse
import glob
import pathlib

import mujoco
import numpy as np

from demo_io import STATE_SPEC, load_demo, site_quat
from pick_place_task import object_in_container

# Replay is deterministic in principle (same solver, same initial state, same
# control sequence), so these absorb only floating-point noise, not a real
# recording bug.
TOLERANCES = {
    "ee_pos": (1e-4, "m", "end-effector position"),
    "joint_pos": (1e-4, "rad", "joint positions (max)"),
    "obj_pos": (1e-4, "m", "object position"),
    "ee_quat": (1e-6, "1-|dot|", "end-effector orientation"),
    "obj_quat": (1e-6, "1-|dot|", "object orientation"),
}


def load_model_for(demo, scene_override=None):
    scene = scene_override or demo.meta["scene_path"]
    if not pathlib.Path(scene).exists():
        raise FileNotFoundError(
            f"scene recorded in {demo.path.name} not found:\n  {scene}\n"
            f"The project tree may have moved -- pass --scene to override.")
    model = mujoco.MjModel.from_xml_path(scene)
    if model.nq != demo.meta["nq"] or model.nu != demo.meta["nu"]:
        raise ValueError(
            f"scene/demo mismatch: model has nq={model.nq} nu={model.nu}, "
            f"demo recorded nq={demo.meta['nq']} nu={demo.meta['nu']}")
    return model


class Replayer:
    """Steps a demo forward one control step at a time.

    Incremental rather than a single loop so the viewer can drive it from Tk's
    event loop and render as it goes, instead of buffering every frame of a
    half-minute episode in memory first.
    """

    def __init__(self, demo, model, mode="action"):
        if mode not in ("action", "state"):
            raise ValueError(f"unknown mode {mode!r}")
        self.demo = demo
        self.model = model
        self.mode = mode
        self.data = mujoco.MjData(model)
        self.substeps = demo.meta["control_substeps"]
        self.site_id = model.site(demo.meta["ee_site"]).id
        self.obj_bid = model.body(demo.meta["object_body"]).id
        self.k = 0
        self.errors = {key: [] for key in TOLERANCES}

        mujoco.mj_resetData(model, self.data)
        mujoco.mj_setState(model, self.data, demo.mj_state[0], STATE_SPEC)
        mujoco.mj_forward(model, self.data)

    @property
    def done(self):
        return self.k >= len(self.demo)

    def step(self):
        """Advance one control step. Returns False when the demo is exhausted."""
        if self.done:
            return False
        k, data, demo = self.k, self.data, self.demo

        if self.mode == "state":
            mujoco.mj_setState(self.model, data, demo.mj_state[k], STATE_SPEC)
            mujoco.mj_forward(self.model, data)

        # Compare before stepping: frame k is the state the recorded action[k]
        # was issued from (see demo_io's frame convention).
        obj_quat = data.xquat[self.obj_bid]
        ee_quat = site_quat(data, self.site_id)
        self.errors["ee_pos"].append(
            float(np.linalg.norm(data.site_xpos[self.site_id] - demo.ee_pos[k])))
        self.errors["joint_pos"].append(
            float(np.max(np.abs(data.qpos[:6] - demo.joint_pos[k]))))
        self.errors["obj_pos"].append(
            float(np.linalg.norm(data.xpos[self.obj_bid] - demo.obj_pos[k])))
        # q and -q are the same rotation, so compare |dot|.
        self.errors["obj_quat"].append(1.0 - abs(float(np.dot(obj_quat, demo.obj_quat[k]))))
        self.errors["ee_quat"].append(1.0 - abs(float(np.dot(ee_quat, demo.ee_quat[k]))))

        if self.mode == "action":
            data.ctrl[:self.model.nu] = demo.action[k]
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, data)
            # Mirrors PolicyRunner._apply: resync derived quantities with qpos.
            mujoco.mj_forward(self.model, data)

        self.k += 1
        return True

    def run(self):
        while self.step():
            pass
        return self

    def ee_error(self, k=None):
        k = self.k - 1 if k is None else k
        return self.errors["ee_pos"][k] if 0 <= k < len(self.errors["ee_pos"]) else 0.0

    def report(self):
        demo = self.demo
        print(f"\n{demo.path.name}  [{self.mode} replay]")
        print(f"  {len(demo)} frames @ {demo.meta['fps']:.2f} Hz "
              f"({demo.meta['duration_s']:.1f} s), source={demo.meta['source']}, "
              f"recorded success={demo.meta['success']}")
        print(f"  {'divergence from recording':<26} {'mean':>10} {'max':>10}   unit")

        checks = []
        for key, (tol, unit, label) in TOLERANCES.items():
            e = np.abs(np.array(self.errors[key]))
            ok = float(e.max()) <= tol
            checks.append((label, ok))
            print(f"  {label:<26} {e.mean():10.2e} {e.max():10.2e}   "
                  f"{unit:<8}{'PASS' if ok else 'FAIL'}")

        replayed = object_in_container(self.model, self.data, demo.meta["object_body"])
        print(f"  eraser ends in container: {replayed} (recorded: {demo.meta['success']})")
        checks.append(("outcome matches recording", replayed == demo.meta["success"]))

        ok = all(c for _, c in checks)
        print(f"  => {'REPLAY OK' if ok else 'REPLAY MISMATCH'}")
        for label, c in checks:
            if not c:
                print(f"     failed: {label}")
        return ok


def view(replayer, cameras=("wrist_cam", "ext_cam"), width=320, height=240,
         speed=1.0):
    """Drive the replay from a Tkinter window showing the scene's cameras.

    Offscreen rendering into Tk rather than mujoco.viewer, for the same reason
    pick_place_teleop.py does it: GLFW/NSWindow is broken on this machine's
    macOS.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    demo = replayer.demo
    root = tk.Tk()
    root.title(f"replay [{replayer.mode}] — {demo.path.name}")
    image_label = tk.Label(root)
    image_label.pack(side=tk.TOP)
    status = tk.Label(root, justify="left", anchor="w", font=("Menlo", 12))
    status.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    renderer = mujoco.Renderer(replayer.model, height, width)
    keep = {}
    delay = max(1, int(1000 * demo.meta["control_dt"] / max(speed, 1e-6)))

    def tick():
        if not replayer.step():
            renderer.close()
            status.configure(text=status.cget("text") + "    [done — close to exit]")
            return
        rendered = []
        for cam in cameras:
            renderer.update_scene(replayer.data, camera=cam)
            rendered.append(renderer.render())
        keep["img"] = ImageTk.PhotoImage(
            Image.fromarray(np.concatenate(rendered, axis=1)))
        image_label.configure(image=keep["img"])
        status.configure(
            text=f"frame {replayer.k}/{len(demo)}   t={replayer.data.time:6.2f}s   "
                 f"ee divergence={replayer.ee_error():.2e} m")
        root.after(delay, tick)

    root.after(0, tick)
    root.mainloop()
    return replayer


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="+", help=".npz demo files (quote globs)")
    ap.add_argument("--mode", choices=("action", "state"), default="action")
    ap.add_argument("--view", action="store_true", help="show the replay in a window")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--scene", default=None, help="override the recorded scene path")
    args = ap.parse_args()

    paths = []
    for pattern in args.demos:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    missing = [p for p in paths if not pathlib.Path(p).exists()]
    if missing:
        ap.error(f"no such demo file(s): {', '.join(missing)}")

    all_ok = True
    for path in paths:
        demo = load_demo(path)
        replayer = Replayer(demo, load_model_for(demo, args.scene), args.mode)
        view(replayer, speed=args.speed) if args.view else replayer.run()
        all_ok &= replayer.report()

    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
