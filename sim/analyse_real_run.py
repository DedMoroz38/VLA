"""Where did the real gripper go, relative to the cube? Reads a run_policy_real.py output folder.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python analyse_real_run.py eval/real/smolvla_cube_dr_030000

For each episode: the gripper frame's closest approach to the cube's start position while it is
low (z < 45 mm), split into radial (along base -> cube; negative = short of the cube, towards the
base) and tangential (positive = anticlockwise seen from above). Computed for the measured joints
(where the arm went) and for the policy's actions (where it was told to go), via forward
kinematics of the sim model after the joint map. The difference is the arm's tracking error.
The same metric on the 148 sim demos is printed as the reference.
"""
import argparse
import glob
import json
import pathlib

import numpy as np

import joint_units
from run_policy_real import SimView

HERE = pathlib.Path(__file__).resolve().parent
LOW_Z = 0.045
JOINTS = ("pan", "lift", "elbow", "wrist_flex", "wrist_roll")


def closest(sim, q_sim, cube_xy):
    """(radial mm, tangential mm, frame index) of the closest low approach, or None."""
    p = sim.gripper_path(q_sim)
    low = np.flatnonzero(p[:, 2] < LOW_Z)
    if not len(low):
        return None
    u = cube_xy / np.linalg.norm(cube_xy)             # the arm base is at the world origin
    v = np.array([-u[1], u[0]])
    rel = p[low, :2] - cube_xy
    i = int(np.argmin(np.linalg.norm(rel, axis=1)))
    return rel[i] @ u * 1000, rel[i] @ v * 1000, int(low[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="eval/real/<run> folder")
    ap.add_argument("--poses", default="eval/cube_poses.json")
    ap.add_argument("--demos", default="demos/cube/trimmed/*.npz")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    sim = SimView(HERE / json.loads((HERE / a.poses).read_text())["scene"])
    jmap = joint_units.JointMap.load()

    ref = [closest(sim, np.load(f)["joint_pos"][:, :6], np.load(f)["obj_pos"][0, :2])
           for f in sorted(glob.glob(str(HERE / a.demos)))]
    ref = np.array([r[:2] for r in ref if r])
    print(f"sim demos ({len(ref)}): radial median {np.median(ref[:, 0]):+.0f} mm "
          f"(p10 {np.percentile(ref[:, 0], 10):+.0f}, p90 {np.percentile(ref[:, 0], 90):+.0f}), "
          f"tangential median {np.median(ref[:, 1]):+.0f} mm\n")

    eps = json.loads((run / "results.json").read_text())["episodes"]
    eps = list(eps.values()) if isinstance(eps, dict) else eps
    print(f"{'pose':>4}  {'outcome':<14} {'measured r/t (mm)':>18} {'commanded r/t':>15}   "
          "joint error measured - commanded (deg): " + " ".join(f"{j:>6}" for j in JOINTS))
    rows = {"success": [], "fail": []}
    for e in eps:
        if not e.get("outcome") or not (run / f"pose{e['pose_id']:04d}.npz").exists():
            continue
        d = np.load(run / f"pose{e['pose_id']:04d}.npz")
        c = np.array([e["x"], e["y"]])
        m = closest(sim, jmap.real_to_sim(d["state"].astype(np.float32)), c)
        cmd = closest(sim, jmap.real_to_sim(d["action"].astype(np.float32)), c)
        if m is None or cmd is None:
            print(f"{e['pose_id']:>4}  {e['outcome']:<14} never low")
            continue
        w = slice(max(m[2] - 15, 0), m[2] + 1)            # the half second before the closest approach
        err = (d["state"][w] - d["action"][w]).mean(0)[:5]
        rows["success" if e["outcome"] == "success" else "fail"].append([m[0], m[1], cmd[0], cmd[1], *err])
        print(f"{e['pose_id']:>4}  {e['outcome']:<14} {m[0]:+8.0f} {m[1]:+5.0f}   {cmd[0]:+8.0f} {cmd[1]:+5.0f}   "
              f"{'':40}" + " ".join(f"{x:+6.1f}" for x in err))
    for k, r in rows.items():
        if r:
            r = np.median(np.array(r), 0)
            print(f"\n{k} ({len(rows[k])}), medians: measured radial {r[0]:+.0f} / tangential {r[1]:+.0f} mm, "
                  f"commanded {r[2]:+.0f} / {r[3]:+.0f} mm, joint error " +
                  ", ".join(f"{j} {x:+.1f}" for j, x in zip(JOINTS, r[4:])))


if __name__ == "__main__":
    main()
