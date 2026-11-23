"""The fixed cube start poses for the cube-scene baseline: 120 poses, 100 for
demonstrations and 20 held out for testing, plus 100 extra sim-only test poses.

    cd sim
    python make_cube_poses.py                  # -> eval/cube_poses.json + eval/cube_poses.png
    python make_cube_poses.py --seed 7 --out eval/cube_poses_alt.json

The 120 are a Latin hypercube over the cube region (x 0.12-0.22, y 0.08-0.18: at least
8 cm to the robot's left, clear of the resting gripper, and reachable top-down within 3 mm
everywhere; pick_place_task.CUBE_SAMPLE_*) and
yaw -45..+45 deg (the cube repeats every 90 deg, so that covers every orientation).
A Latin hypercube is random but stratified: split each of x, y and yaw into 120 equal
slices and every slice holds exactly one pose, so there are no clusters or gaps the
way independent random draws leave them.

The 20 test poses are a random subset of the 120, chosen among 5000 random splits as
the one whose test poses are most spread out (largest smallest distance between two
of them, in x/y/yaw each scaled to its range), so the test set isn't bunched up in one
corner. The other 100 are the training poses, in random order (so the first N demos
already cover the region if you stop early).

'sim_test' adds 100 independent random poses from the same region for sim-only
evaluation: 20 test episodes give roughly +-20 points of uncertainty, 120 about +-9.

Every pose keeps the cube at least 10 mm clear of the container's tape. Each has an
"id" (0-119 for the main set, 1000+ for sim_test) that the teleop recorder stores
with the demo and eval reports.

Used by:  pick_place_teleop.py --poses eval/cube_poses.json   (steps through 'train')
          eval_policy.py CKPT --poses eval/cube_poses.json --split test
          calib/live_overlay.py --cube X Y YAW                  (place the real cube on a pose)
"""
import argparse
import datetime
import json
import pathlib

import mujoco
import numpy as np

from demo_io import task_for
from pick_place_task import CUBE_SAMPLE_X, CUBE_SAMPLE_Y, SAMPLE_YAW

HERE = pathlib.Path(__file__).parent
SCENE = "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
CUBE_HALF = 0.02
CLEARANCE = 0.010


def tape_box(scene):
    """(xmin, xmax, ymin, ymax) of the container's tape band on the table, from the scene."""
    m = mujoco.MjModel.from_xml_path(str(HERE / scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = m.geom("container_tape").id
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
    return w[:, 0].min(), w[:, 0].max(), w[:, 1].min(), w[:, 1].max()


def clear_of(box, x, y, yaw):
    h = CUBE_HALF * (abs(np.cos(yaw)) + abs(np.sin(yaw)))      # half-size of the rotated cube's box
    x0, x1, y0, y1 = box
    dx = max(x0 - (x + h), (x - h) - x1, 0.0)
    dy = max(y0 - (y + h), (y - h) - y1, 0.0)
    return np.hypot(dx, dy) >= CLEARANCE


def latin_hypercube(rng, n, bounds):
    cols = [lo + (rng.permutation(n) + rng.uniform(size=n)) / n * (hi - lo) for lo, hi in bounds]
    return np.stack(cols, 1)


def normalised(p, bounds):
    """x, y scaled to [0, 1]; yaw as a point on a circle of circumference 1 (90 deg periodic)."""
    (x0, x1), (y0, y1), (a0, a1) = bounds
    t = 2 * np.pi * (p[:, 2] - a0) / (a1 - a0)
    r = 1 / (2 * np.pi)
    return np.stack([(p[:, 0] - x0) / (x1 - x0), (p[:, 1] - y0) / (y1 - y0), r * np.cos(t), r * np.sin(t)], 1)


def spread(q):
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    return d[np.triu_indices(len(q), 1)].min()


def draw_map(path, poses, box):
    """Top-down sketch: the region, the container's tape, and every pose as a dot with a tick
    showing its yaw (train blue, test orange, sim_test grey)."""
    from PIL import Image, ImageDraw
    s = 1800                                         # px per metre
    ymax, xmin = 0.20, -0.03                         # world point at the image's top-left
    W, H = int(0.52 * s), int(0.40 * s)
    img = Image.new("RGB", (W, H + 40), "white")
    dr = ImageDraw.Draw(img)
    px = lambda x, y: ((ymax - y) * s, 40 + (x - xmin) * s)    # +y to the left, x down the image
    x0, x1, y0, y1 = box
    dr.rectangle([px(x0, y1), px(x1, y0)], outline=(30, 140, 70), width=3)
    dr.rectangle([px(CUBE_SAMPLE_X[0], CUBE_SAMPLE_Y[1]), px(CUBE_SAMPLE_X[1], CUBE_SAMPLE_Y[0])],
                 outline=(160, 160, 160), width=1)
    bx, by = px(0.0, 0.0)
    dr.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill="black")
    cols = {"sim_test": (190, 190, 190), "train": (40, 90, 220), "test": (240, 120, 20)}
    for split in ("sim_test", "train", "test"):
        r = 5 if split == "test" else 3
        for p in (q for q in poses if q["split"] == split):
            u, v = px(p["x"], p["y"])
            tu, tv = px(p["x"] + 0.012 * np.cos(p["yaw_rad"]), p["y"] + 0.012 * np.sin(p["yaw_rad"]))
            dr.line([u, v, tu, tv], fill=cols[split], width=2)
            dr.ellipse([u - r, v - r, u + r, v + r], fill=cols[split])
    n = {k: sum(p["split"] == k for p in poses) for k in cols}
    dr.text((10, 6), "top view, robot base = black dot (x down, +y left); tick = cube yaw (0 deg = along x)\n"
            f"blue train ({n['train']}), orange test ({n['test']}), grey sim_test ({n['sim_test']}); "
            "green = container tape; grey box = newest region", fill="black")
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-sim-test", type=int, default=100)
    ap.add_argument("--out", default="eval/cube_poses.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    bounds = (CUBE_SAMPLE_X, CUBE_SAMPLE_Y, SAMPLE_YAW["cube"])
    box = tape_box(SCENE)
    for _ in range(1000):                            # all poses clear of the container
        main_set = latin_hypercube(rng, a.n, bounds)
        if all(clear_of(box, *p) for p in main_set):
            break
    else:
        raise SystemExit("could not place the poses clear of the container")

    q = normalised(main_set, bounds)
    best, best_score, scores = None, -1.0, []
    for _ in range(5000):
        pick = rng.choice(a.n, a.n_test, replace=False)
        sc = spread(q[pick])
        scores.append(sc)
        if sc > best_score:
            best, best_score = pick, sc
    test = set(best.tolist())

    extra = []
    while len(extra) < a.n_sim_test:
        p = np.array([rng.uniform(*b) for b in bounds])
        if clear_of(box, *p):
            extra.append(p)

    def entry(i, p, split):
        return {"id": i, "x": float(p[0]), "y": float(p[1]), "yaw_rad": float(p[2]),
                "yaw_deg": float(np.degrees(p[2])), "split": split}

    poses = [entry(i, p, "test" if i in test else "train") for i, p in enumerate(main_set)]
    poses += [entry(1000 + i, p, "sim_test") for i, p in enumerate(extra)]
    out = {
        "description": "Cube start poses: 100 train (demos) + 20 test (held out) from one Latin "
                       "hypercube, plus 100 sim_test (independent, sim-only eval). make_cube_poses.py",
        "created": datetime.date.today().isoformat(),
        "seed": a.seed,
        "scene": SCENE,
        "object": "cube",
        "task": task_for("cube"),
        "region": {"x": list(CUBE_SAMPLE_X), "y": list(CUBE_SAMPLE_Y),
                   "yaw_deg": [float(np.degrees(v)) for v in SAMPLE_YAW["cube"]]},
        "container_tape_box": [float(v) for v in box],
        "train_demos": ["demos/cube/teleop_*.npz"],
        "test_spread": float(best_score),
        "counts": {s: sum(p["split"] == s for p in poses) for s in ("train", "test", "sim_test")},
        "poses": poses,
    }
    path = HERE / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    draw_map(path.with_suffix(".png"), poses, box)
    print(f"wrote {path} ({out['counts']}) and {path.with_suffix('.png').name}; "
          f"test spread (closest pair) {best_score:.3f} vs {np.median(scores):.3f} for a typical random split")


if __name__ == "__main__":
    main()
