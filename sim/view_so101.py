"""Interactive SO-101 viewer that avoids GLFW/NSWindow (broken on macOS 26 Tahoe).

Renders MuJoCo offscreen and displays frames in a Tkinter window instead of
using mujoco.viewer's native GLFW window.

Usage:
    python view_so101.py [path/to/scene.xml]
"""
import sys
import pathlib
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageTk

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else str(
    pathlib.Path(__file__).parent / "mujoco_menagerie/robotstudio_so101/scene.xml"
)
WIDTH, HEIGHT = 640, 480
FPS = 30


class So101Viewer:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.0
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self.cam.lookat[:] = [0.0, 0.0, 0.15]

        self._drag_prev = None

        self.root = tk.Tk()
        self.root.title(f"SO-101 — {pathlib.Path(model_path).name}")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.LEFT)
        self.image_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.image_label.bind("<B1-Motion>", self._on_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_drag_end)
        self.image_label.bind("<MouseWheel>", self._on_scroll)
        self.image_label.bind("<Button-4>", lambda e: self._zoom(-1))
        self.image_label.bind("<Button-5>", lambda e: self._zoom(1))

        controls = tk.Frame(self.root)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.sliders = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            lo, hi = self.model.actuator_ctrlrange[i]
            if lo == hi:
                lo, hi = -3.14, 3.14
            tk.Label(controls, text=name).pack(anchor="w")
            s = tk.Scale(
                controls, from_=lo, to=hi, resolution=0.01, orient=tk.HORIZONTAL,
                length=220, command=lambda v, idx=i: self._on_slider(idx, v),
            )
            s.set(0.0)
            s.pack(anchor="w", pady=(0, 8))
            self.sliders.append(s)

        tk.Button(controls, text="Reset", command=self.reset).pack(anchor="w", pady=(10, 0))
        tk.Label(
            controls, text="Drag = orbit camera\nScroll = zoom",
            justify="left", fg="gray",
        ).pack(anchor="w", pady=(20, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._tk_img = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for s in self.sliders:
            s.set(0.0)

    def _on_slider(self, idx, value):
        self.data.ctrl[idx] = float(value)

    def _on_drag_start(self, event):
        self._drag_prev = (event.x, event.y)

    def _on_drag_end(self, event):
        self._drag_prev = None

    def _on_drag(self, event):
        if self._drag_prev is None:
            return
        dx = event.x - self._drag_prev[0]
        dy = event.y - self._drag_prev[1]
        self._drag_prev = (event.x, event.y)
        self.cam.azimuth -= dx * 0.3
        self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.3, -89, 89))

    def _zoom(self, direction):
        self.cam.distance = float(np.clip(self.cam.distance * (1.1 ** direction), 0.1, 10.0))

    def _on_scroll(self, event):
        self._zoom(-1 if event.delta > 0 else 1)

    def _tick(self):
        for _ in range(max(1, int(1 / (FPS * self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.cam)
        frame = self.renderer.render()
        img = Image.fromarray(frame)
        self._tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk_img)

        self.root.after(int(1000 / FPS), self._tick)

    def close(self):
        self.renderer.close()
        self.root.destroy()

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


if __name__ == "__main__":
    So101Viewer(MODEL_PATH).run()
