"""Re-score a cube eval_policy.py results file with a looser "inside the container" rule.

    cd sim
    python rescore_eval.py eval/results/smolvla_cube_dr_030000_test.json [more.json ...]

The eval's own rule (container_tol in the scene) needs the cube centre at least 5 mm away from
where the cube would touch a wall, so a cube resting flush against a wall counts as a failure.
This script also reports the looser rule, using each episode's final cube pose:
  - every corner of the cube's footprint (40 mm, at its final yaw) within the container's inner
    walls (+ TOUCH_MM), and
  - the cube low enough to be lying on the container floor (not tilted on the rim or held up).
Nothing is re-simulated, and the final pose is at the end of the episode (the 45 s timeout for a
failure), so a cube placed and then pushed out again counts as outside.
"""
import json
import pathlib
import sys

import numpy as np

CONTAINER_XY = np.array([0.2095, -0.2245])     # pick_place_scene_real.xml, body "container"
INNER_HALF = np.array([0.0995, 0.0635])        # inner faces of container_wall_px / _py
CUBE_HALF = 0.020
TOUCH_MM = 1.0                                 # contact / solver slack
FLOOR_Z_MAX = 0.030                            # flat on the floor: centre at ~0.022 m


def inside(ep):
    x, y, yaw = ep["final_eraser_pose"]
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    reach = CUBE_HALF * (c + s)                # footprint half-extent along x and along y
    d = np.abs(np.array([x, y]) - CONTAINER_XY)
    slack = d + reach - INNER_HALF             # > 0: past the wall
    return bool(np.all(slack <= TOUCH_MM / 1000) and ep["final_eraser_z"] < FLOOR_Z_MAX), slack * 1000


def main():
    for path in sys.argv[1:]:
        res = json.loads(pathlib.Path(path).read_text())
        eps = res["results"]
        strict = sum(e["success"] for e in eps)
        loose = 0
        print(f"\n{path}  ({res['checkpoint']})")
        for e in eps:
            ok, slack = inside(e)
            loose += e["success"] or ok
            if not e["success"]:
                x, y, yaw = e["final_eraser_pose"]
                where = "INSIDE (against a wall)" if ok else "outside / not on the floor"
                print(f"  pose {e['pose_id']}: {e['outcome']:<36} final x={x:.3f} y={y:+.3f} "
                      f"z={e['final_eraser_z']:.3f} yaw={np.degrees(yaw):+5.0f}  "
                      f"wall clearance x {-slack[0]:+5.1f} mm, y {-slack[1]:+5.1f} mm -> {where}")
        n = len(eps)
        print(f"  eval rule (5 mm wall margin): {strict}/{n} = {strict / n:.0%}")
        print(f"  cube anywhere inside, on the floor: {loose}/{n} = {loose / n:.0%}")


if __name__ == "__main__":
    main()
