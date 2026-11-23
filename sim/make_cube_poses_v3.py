"""Cube poses v3 (2026-11-21): add 50 'train' poses much further from the robot to the
v2 file (whose 100 train poses all have demos), since many v2 poses were awkwardly close.

    cd sim
    python make_cube_poses_v3.py          # appends to eval/cube_poses.json (v2 kept as eval/cube_poses_v2.json)

- region x 0.24-0.33, y 0.00-0.20 (v2 was x 0.16-0.26, y 0.08-0.18)
- reach: free gripper pitch (as when grabbing by hand with the leader), cube-centre height,
  within 3 mm (IK). A straight-down grasp only reaches ~27 cm out; tilted (~42 deg) the arm
  reaches past 36 cm, so here the side camera is the limit
- the whole cube (top corners) inside the side camera image with a 15 px margin
- >= 20 mm from the arm in its start pose, >= 10 mm from the container tape
- scrambled Halton over x/y/yaw (evenly spread); ids 400-449, split 'train', appended after
  the existing 100 train poses, so the teleop continues at #101
"""
import json
import pathlib

import mujoco
import numpy as np

import make_cube_poses as v1
from pick_place_task import FREE_SPACE_TOL, PolicyRunner, SAMPLE_YAW, load_scene, set_eraser_pose

HERE = pathlib.Path(__file__).parent
REGION_X, REGION_Y = (0.24, 0.33), (0.00, 0.20)
N_NEW, SEED, REACH_TOL, CAM_MARGIN_PX = 50, 20261123, 0.003, 15


def halton1(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Checks:
    def __init__(self):
        self.m, self.d = load_scene(str(HERE / v1.SCENE))
        m, d = self.m, self.d
        self.cube = m.geom("cube_geom").id
        lo, hi = m.body("base").id, m.body("moving_jaw_so101_v1").id
        self.robot = [g for g in range(m.ngeom) if lo <= m.geom_bodyid[g] <= hi and (m.geom_contype[g] or m.geom_conaffinity[g])]
        c = m.camera("ext_cam").id
        self.cp, self.R = d.cam_xpos[c].copy(), d.cam_xmat[c].reshape(3, 3).copy()
        self.f = 240 / np.tan(np.radians(m.cam_fovy[c] / 2))

    def in_view(self, x, y, yaw):
        cs, sn = np.cos(yaw), np.sin(yaw)
        for u, v in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            for z in (0.0, 0.04):
                p = np.array([x + 0.02 * (u * cs - v * sn), y + 0.02 * (u * sn + v * cs), z])
                c = (p - self.cp) @ self.R
                px, py = 320 + self.f * c[0] / -c[2], 240 - self.f * c[1] / -c[2]
                if not (CAM_MARGIN_PX < px < 640 - CAM_MARGIN_PX and CAM_MARGIN_PX < py < 480 - CAM_MARGIN_PX):
                    return False
        return True

    def arm_clearance(self, x, y, yaw):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        set_eraser_pose(self.m, self.d, x, y, yaw, object_body="cube")
        return min(mujoco.mj_geomDistance(self.m, self.d, g, self.cube, 0.3, None) for g in self.robot)


def reach_error(x, y):
    m, d = load_scene(str(HERE / v1.SCENE))
    adr = m.jnt_qposadr[m.body("cube").jntadr[0]]
    d.qpos[adr:adr + 3] = (0.6, 0.6, 0.02)
    mujoco.mj_forward(m, d)
    r = PolicyRunner(m, d)
    r.move_to(np.array([x, y, 0.08]), pos_tol=FREE_SPACE_TOL)
    r.move_to(np.array([x, y, 0.02]))                      # free pitch: as a hand-driven grasp
    return float(np.linalg.norm(r.ik.site_pos(d) - [x, y, 0.02]))


def main():
    path = HERE / "eval/cube_poses.json"
    data = json.loads(path.read_text())
    if any(p["id"] >= 400 and p["id"] < 500 for p in data["poses"]):
        raise SystemExit("eval/cube_poses.json already has the v3 poses (ids 400+)")
    rng = np.random.default_rng(SEED)
    shift = rng.uniform(size=3)
    bounds = (REGION_X, REGION_Y, SAMPLE_YAW["cube"])
    box = v1.tape_box(v1.SCENE)
    chk = Checks()
    new, i, rejected = [], 20, {"view": 0, "arm": 0, "tape": 0, "reach": 0}
    while len(new) < N_NEW:
        u = (np.array([halton1(i, b) for b in (2, 3, 5)]) + shift) % 1.0
        i += 1
        x, y, yaw = (lo + uu * (hi - lo) for uu, (lo, hi) in zip(u, bounds))
        if not chk.in_view(x, y, yaw):
            rejected["view"] += 1
        elif chk.arm_clearance(x, y, yaw) < 0.020:
            rejected["arm"] += 1
        elif not v1.clear_of(box, x, y, yaw):
            rejected["tape"] += 1
        elif reach_error(x, y) > REACH_TOL:
            rejected["reach"] += 1
        else:
            new.append({"id": 400 + len(new), "x": float(x), "y": float(y), "yaw_rad": float(yaw),
                        "yaw_deg": float(np.degrees(yaw)), "split": "train"})
    train = [p for p in data["poses"] if p["split"] == "train"]
    rest = [p for p in data["poses"] if p["split"] != "train"]
    data["poses"] = train + new + rest
    data["version"] = 3
    data["description"] += (" v3: + 50 train poses much further out (x 0.24-0.33, y 0.00-0.20; free-pitch "
                            "reach, fully in the side view), ids 400-449. make_cube_poses_v3.py")
    data["region_v3_extra"] = {"x": list(REGION_X), "y": list(REGION_Y), "yaw_deg": [-45.0, 45.0],
                               "reach": "free pitch, cube-centre height, within 3 mm (IK)"}
    data["counts"] = {s: sum(p["split"] == s for p in data["poses"]) for s in ("train", "test", "sim_test")}
    path.write_text(json.dumps(data, indent=2))
    v1.CUBE_SAMPLE_X, v1.CUBE_SAMPLE_Y = REGION_X, REGION_Y
    v1.draw_map(HERE / "eval/cube_poses.png", data["poses"], box)
    xs, ys = [p["x"] for p in new], [p["y"] for p in new]
    print(f"added {len(new)} train poses (x {min(xs):.3f}-{max(xs):.3f}, y {min(ys):.3f}-{max(ys):.3f}); "
          f"rejected {rejected}; counts now {data['counts']}")


if __name__ == "__main__":
    main()
