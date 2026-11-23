"""Shared geometry for building the sim floor from the real side camera (C920) and a photo."""
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
FLOOR_Z = -0.72          # real table top is ~72 cm above the floor; table top is z = 0
TABLE_EDGE_X = -0.021    # table back edge (from the tuned scene)
# world window of the floor texture (x behind the robot is negative), metres
X0, X1, Y0, Y1 = -3.2, 0.2, -2.2, 2.2
RES = 0.002              # metres per texture pixel


def ext_cam():
    xml = SCENE.read_text()
    el = re.search(r'<camera name="ext_cam"[^>]*/>', xml, re.S).group(0)
    a = lambda k: np.array([float(v) for v in re.search(rf'{k}="([^"]*)"', el).group(1).split()])
    pos, xy, fovy = a("pos"), a("xyaxes"), a("fovy")[0]
    x = xy[:3] / np.linalg.norm(xy[:3]); y = xy[3:] - x * (x @ xy[3:]); y /= np.linalg.norm(y)
    return pos, np.c_[x, y, np.cross(x, y)], fovy


def project(pts, W, H):
    """World points (..., 3) -> pixel u, v in a W x H image of the C920's 16:9 frame (same vertical FOV)."""
    pos, R, fovy = ext_cam()
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    c = (np.asarray(pts) - pos) @ R
    return W / 2 + f * c[..., 0] / -c[..., 2], H / 2 - f * c[..., 1] / -c[..., 2], c[..., 2] < 0


def grid():
    """Texture grid: rows run from far (x = X0) to near (x = X1), columns from robot-right (y = Y0) to left?
    No: columns from y = Y1 (camera-left... ) -- see note. Returned as world X, Y arrays (rows, cols)."""
    xs = np.arange(X0, X1, RES)
    ys = np.arange(Y0, Y1, RES)
    return np.meshgrid(xs, ys, indexing="ij")     # [i, j] = (xs[i], ys[j])


def floor_mask_in_image(W, H, margin_px=6):
    """Image pixels that see the floor: above the table's back edge (as the camera sees it)."""
    yy = np.linspace(-1.5, 1.5, 601)
    u, v, _ = project(np.c_[np.full_like(yy, TABLE_EDGE_X), yy, np.zeros_like(yy)], W, H)
    order = np.argsort(u)
    edge = np.interp(np.arange(W), u[order], v[order])
    vv = np.arange(H)[:, None]
    return vv < (edge[None, :] - margin_px)
