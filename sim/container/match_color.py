"""Match the sim container's colours to the real side camera (C920).

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python container/match_color.py              # grab, fit the tape, write
    /opt/anaconda3/envs/lerobot/bin/python container/match_color.py --dry-run    # report only
    /opt/anaconda3/envs/lerobot/bin/python container/match_color.py --frame container/check/real_side.png

Averages a few C920 frames (or uses --frame), renders the cube scene's ext_cam, and
compares the median colour in the pixels where the sim sees the tape directly (eroded so
edges and slight misplacement don't count). The tape's RGB is then scaled channel by
channel until the render matches (MuJoCo shading is close to linear in the material
colour away from clipping, so it converges in a few rounds). The container's inside
(seen through the transparent shell) is compared too and, with --container, its tint is
fitted the same way. The result is written into build_container.py's TAPE_COLOR /
CONTAINER_COLOR and the scene is rebuilt. Writes container/check/color_match.png.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM / "calib"))
sys.path.insert(0, str(HERE))
import build_container as bc  # noqa: E402

W, H = 640, 480


def grab(n=15):
    from cameras import Grabber
    cam = Grabber("C920")
    time.sleep(2.0)                          # let auto-exposure / white balance settle
    frames, last = [], -1
    while len(frames) < n:
        f, k = cam.get()
        if k != last:
            frames.append(f.astype(np.float32))
            last = k
        time.sleep(0.01)
    cam.close()
    return np.mean(frames, 0).astype(np.uint8)


class Render:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(bc.SCENE))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("ready").id)
        j = self.m.body("cube").jntadr[0]
        self.d.qpos[self.m.jnt_qposadr[j]] = 5.0       # cube out of view (it is wherever in real)
        mujoco.mj_forward(self.m, self.d)
        self.rgb = mujoco.Renderer(self.m, H, W)
        self.seg = mujoco.Renderer(self.m, H, W)
        self.seg.enable_segmentation_rendering()

    def set_colors(self, tape, cont):
        self.m.mat_rgba[self.m.material("container_tape_mat").id][:3] = tape
        self.m.mat_rgba[self.m.material("container_mat").id][:3] = cont

    def image(self):
        self.rgb.update_scene(self.d, camera="ext_cam")
        return self.rgb.render()[..., ::-1].copy()      # BGR

    def masks(self):
        self.seg.update_scene(self.d, camera="ext_cam")
        s = self.seg.render()
        g = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        k = np.ones((7, 7), np.uint8)
        er = lambda m: cv2.erode(m.astype(np.uint8), k).astype(bool)
        tape = er(g == self.m.geom("container_tape").id)
        inside = er(g == self.m.geom("container_shell").id)
        return tape, inside


def med(img, m):
    return np.median(img[m].reshape(-1, 3), 0)          # BGR


def fit(r, real, mask, key, tape, cont, rounds=6):
    """Scale one material's RGB until the median render colour in `mask` matches the real one."""
    target = med(real, mask)[::-1]                       # RGB
    c = np.array(tape if key == "tape" else cont, float)
    for _ in range(rounds):
        r.set_colors(c if key == "tape" else tape, c if key == "cont" else cont)
        cur = med(r.image(), mask)[::-1]
        if np.abs(cur - target).max() < 1.0:
            break
        c = np.clip(c * (target + 1) / (cur + 1), 0.0, 1.0)
    r.set_colors(c if key == "tape" else tape, c if key == "cont" else cont)
    return c, target, med(r.image(), mask)[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", help="use this saved real frame instead of grabbing one")
    ap.add_argument("--container", action="store_true", help="also fit the container's tint")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    (HERE / "check").mkdir(exist_ok=True)
    if a.frame:
        real = cv2.imread(a.frame)
    else:
        real = grab()
        cv2.imwrite(str(HERE / "check/real_side.png"), real)
    r = Render()
    tape_m, in_m = r.masks()
    tape0, cont0 = np.array(bc.TAPE_COLOR, float), np.array(bc.CONTAINER_COLOR, float)
    before = r.image()
    print(f"tape pixels {tape_m.sum()}, container-inside pixels {in_m.sum()}")
    for name, m in (("tape", tape_m), ("container inside", in_m)):
        print(f"  {name:17s} real RGB {np.round(med(real, m)[::-1]).astype(int)}   "
              f"sim RGB {np.round(med(before, m)[::-1]).astype(int)}")

    tape, t_real, t_sim = fit(r, real, tape_m, "tape", tape0, cont0)
    print(f"tape colour {np.round(tape0, 3)} -> {np.round(tape, 3)}: render {np.round(t_sim).astype(int)} "
          f"vs real {np.round(t_real).astype(int)}")
    cont = cont0
    if a.container:
        cont, c_real, c_sim = fit(r, real, in_m, "cont", tape, cont0)
        print(f"container tint {np.round(cont0, 3)} -> {np.round(cont, 3)}: render {np.round(c_sim).astype(int)} "
              f"vs real {np.round(c_real).astype(int)}")
    after = r.image()

    def outline(img):
        o = img.copy()
        for m, col in ((tape_m, (0, 0, 255)), (in_m, (0, 165, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 1)
        return o
    tiles = [outline(real), before, after]
    for t, label in zip(tiles, ("real (measured areas)", "sim before", "sim after")):
        cv2.putText(t, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(HERE / "check/color_match.png"), np.hstack(tiles))
    print("wrote", HERE / "check/color_match.png")

    if a.dry_run:
        return
    bc.save_colors(tape=tape, container=cont if a.container else None)
    args = bc.default_args()                           # size/placement as saved; colours set below
    args.tape_color, args.color = tuple(tape), tuple(cont)
    bc.build(args)
    print("saved TAPE_COLOR" + (" and CONTAINER_COLOR" if a.container else "") + " in build_container.py; scene rebuilt")


if __name__ == "__main__":
    main()
