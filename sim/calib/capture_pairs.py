"""Capture matched real / sim image pairs, to measure the sim-to-real gap per camera.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/capture_pairs.py                 # 5 spread test poses
    /opt/anaconda3/envs/lerobot/bin/python calib/capture_pairs.py --ids 300 319
    /opt/anaconda3/envs/lerobot/bin/python calib/capture_pairs.py --leader       # follower follows the leader

By default the follower's torque is OFF: move it by hand (the gripper too). With --leader the
follower is torqued and follows the leader arm: its target walks toward the leader by at most
--max-step deg per tick (it catches up smoothly if the two start apart) and never leads the
measured position by more than LEAD_CAP_DEG. Put the leader down to hold a pose. Close the
gripper only gently on the cube -- the follower squeezes as hard as the leader asks. On quit it
goes back to rest.

The cube goes on the yellow outline of the current pose. On every SPACE the real side and wrist frames are saved losslessly
(the wrist rotated exactly as the policy sees it), together with the measured joints, and the
sim twin renders the same two views at those joints with the cube on the pose. The window
shows real (top) and sim (bottom) live, so you can see they match.

What to capture, per pose (~8 snapshots): rest pose, mid-way, above the cube, low next to it,
jaws around it (open, then gently closed). Keep the cube ON THE TABLE: the sim cube stays on
its pose, so a lifted real cube would no longer match. Hold the arm still while pressing SPACE
(a snapshot taken while moving is refused).

    SPACE  save a pair        n / b  next / previous pose        q  quit

Output: calib/pairs/<date_time>/NNN_{real,sim}_{side,wrist}.png and pairs.json.
"""
import argparse
import datetime
import json
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))

import joint_units  # noqa: E402
import run_policy_real as rp  # noqa: E402

DEFAULT_IDS = [300, 305, 311, 319, 500]
LEADER_PORT = "/dev/tty.usbmodem5B790804261"
LEAD_CAP_DEG = 25.0      # --leader: the follower's target never leads its measured position by more
# The rate limit is on the TARGET, not on target-minus-measured: the servos are proportional, so a
# target only 2 deg from a joint under gravity load (the elbow) gives too little torque to move it,
# and a limit relative to the measured position would then stall that joint forever.
STILL_DEG = 0.6          # max joint change between the two reads around a snapshot


class LimpFollower:
    """The follower with its torque off: read-only."""

    def __init__(self, port):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id="so101_follower"))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit("follower motors don't match its calibration file -- run lerobot-calibrate first")
        self.bus.disable_torque()
        print(f"follower {port}: torque OFF, move it by hand")

    def read(self):
        p = self.bus.sync_read("Present_Position", num_retry=2)
        return np.array([p[j] for j in rp.JOINTS], np.float32)

    def close(self):
        self.bus.disconnect(disable_torque=True)


class Leader:
    """The leader arm (torque off, moved by hand), read synchronously."""

    def __init__(self, port):
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        self.arm = SO101Leader(SO101LeaderConfig(port=port, id="so101_leader"))
        self.bus = self.arm.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect()
            raise SystemExit("leader motors don't match its calibration file -- run lerobot-calibrate first")
        self.arm.configure()
        print(f"leader {port}: move it by hand, the follower follows")

    def read(self):
        p = self.bus.sync_read("Present_Position", num_retry=2)
        return np.array([p[j] for j in rp.JOINTS], np.float32)

    def close(self):
        self.bus.disconnect()


def follow_health(arm, ql, qf, log):
    """--leader: per-joint lag, servo error bits and load; logged every tick, and the lines worth
    showing (big lags, any error bit, load over 60%) returned for the window."""
    try:
        st = arm.bus.sync_read("Status", normalize=False, num_retry=1)
        ld = arm.bus.sync_read("Present_Load", normalize=False, num_retry=1)
    except ConnectionError:
        return ["(motor status read failed this tick)"]
    load = {j: (ld[j] & 0x3FF) / 10 for j in rp.JOINTS}           # % of max torque, bit 10 = direction
    log.write(f"{time.time():.3f}," + ",".join(
        f"{ql[k]:.1f},{qf[k]:.1f},{st[j]},{load[j]:.1f}" for k, j in enumerate(rp.JOINTS)) + "\n")
    out = []
    lag = [f"{j} {qf[k] - ql[k]:+.0f}" for k, j in enumerate(rp.JOINTS) if abs(qf[k] - ql[k]) > 5]
    if lag:
        out.append("follower lag (deg): " + ", ".join(lag))
    bad = [f"{j} status {st[j]:#b}" for j in rp.JOINTS if st[j]]
    if bad:
        out.append("SERVO ERROR: " + ", ".join(bad) + "  (32 = overload, 8 = current, 4 = temp)")
    if bad != follow_health.last:                                   # terminal: only on change
        print("SERVO ERROR: " + ", ".join(bad) if bad else "servo errors cleared", flush=True)
        follow_health.last = bad
    hot = [f"{j} {load[j]:.0f}%" for j in rp.JOINTS if load[j] > 60]
    if hot:
        out.append("high load: " + ", ".join(hot))
    return out


follow_health.last = []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default="eval/cube_poses.json")
    ap.add_argument("--ids", type=int, nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--follower", default=rp.FOLLOWER_PORT)
    ap.add_argument("--leader", nargs="?", const=LEADER_PORT, default=None, metavar="PORT",
                    help=f"follower follows the leader arm (default port {LEADER_PORT})")
    ap.add_argument("--max-step", type=float, default=2.0, help="--leader: deg (gripper %%) per tick")
    ap.add_argument("--side", default="C920")
    ap.add_argument("--wrist", default="icspring")
    ap.add_argument("--out", default=None, help="default: calib/pairs/<date_time>")
    a = ap.parse_args()

    pose_file = json.loads((SIM / a.poses).read_text())
    by_id = {p["id"]: p for p in pose_file["poses"]}
    poses = [by_id[i] for i in a.ids]
    sim = rp.SimView(SIM / pose_file["scene"])
    jmap = joint_units.JointMap.load()
    out = pathlib.Path(a.out) if a.out else HERE / "pairs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    index = {"poses_file": a.poses, "scene": pose_file["scene"], "joint_map": jmap.meta.get("offsets_deg"),
             "wrist_orientation": "rotated 90 deg anticlockwise, as the policy sees it", "pairs": []}

    cams = rp.Cameras(a.side, a.wrist)
    leader, flog, target = None, None, None
    if a.leader:
        flog = open(out / "follow_log.csv", "w")
        flog.write("t," + ",".join(f"leader_{j},follower_{j},status_{j},load_{j}" for j in rp.JOINTS) + "\n")
        arm = rp.Follower(a.follower, a.max_step, jmap, sim)
        try:
            leader = Leader(a.leader)
        except BaseException:
            arm.move_to(jmap.sim_to_real(sim.ready_sim))
            arm.close()
            raise
    else:
        arm = LimpFollower(a.follower)
    win = "real (top) vs sim (bottom)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i, n_saved, msg, msg_t = 0, 0, "", 0.0
    try:
        while True:
            p = poses[i]
            pose = (p["x"], p["y"], p["yaw_rad"])
            q = arm.read()
            follow = []
            if leader is not None:
                ql = leader.read()
                if target is None:
                    target = q.copy()
                target = target + np.clip(ql - target, -a.max_step, a.max_step)
                target = np.clip(target, arm.lo, arm.hi)
                target = np.clip(target, q - LEAD_CAP_DEG, q + LEAD_CAP_DEG)
                arm.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(rp.JOINTS, target)})
                follow = follow_health(arm, ql, q, flog)
            real = cams.raw()
            simimg = sim.images(jmap.real_to_sim(q), pose)
            outline = [(poly, (0, 255, 255)) for poly in sim.cube_outline(*pose)]
            here = sum(1 for e in index["pairs"] if e["pose_id"] == p["id"])
            lines = [f"pose {i + 1}/{len(poses)} (id {p['id']}): put the cube on the yellow outline; "
                     f"{here} saved here, {n_saved} in total",
                     ("follower follows the leader;  " if leader is not None else "")
                     + "SPACE save (hold still)   n/b next/prev pose   q quit"]
            lines += follow
            if msg and time.time() - msg_t < 2.0:
                lines.append(msg)
            top = rp.compose(real, lines, polys=outline, sim=sim)
            bottom = rp.compose(simimg, ["sim twin at the measured joints"], polys=outline, sim=sim)
            cv2.imshow(win, np.vstack([top, bottom]))
            k = rp.key()
            if k == "q":
                break
            if k in ("n", "b"):
                i = (i + (1 if k == "n" else -1)) % len(poses)
            elif k == " ":
                q0 = arm.read()
                real = cams.raw()
                q1 = arm.read()
                if np.abs(q1 - q0).max() > STILL_DEG:
                    msg, msg_t = "MOVING -- hold still and press SPACE again", time.time()
                    continue
                qm = (q0 + q1) / 2
                simimg = sim.images(jmap.real_to_sim(qm), pose)
                stem = f"{n_saved:03d}"
                files = {}
                for src, imgs in (("real", real), ("sim", simimg)):
                    for cam, tag in (("ext_cam", "side"), ("wrist_cam", "wrist")):
                        f = f"{stem}_{src}_{tag}.png"
                        cv2.imwrite(str(out / f), cv2.cvtColor(imgs[cam], cv2.COLOR_RGB2BGR))
                        files[f"{src}_{tag}"] = f
                grip = sim.gripper_path(jmap.real_to_sim(qm)[None])[0]
                index["pairs"].append({
                    "n": n_saved, "pose_id": p["id"], "pose": list(pose), "joints_real": qm.tolist(),
                    "gripper_to_cube_mm": float(np.linalg.norm(grip[:2] - pose[:2]) * 1000),
                    "gripper_height_mm": float(grip[2] * 1000), "files": files,
                    "time": datetime.datetime.now().isoformat(timespec="seconds")})
                (out / "pairs.json").write_text(json.dumps(index, indent=2))
                n_saved += 1
                msg, msg_t = f"saved pair {stem}", time.time()
    finally:
        if leader is not None:
            print("follower back to rest ...")
            arm.move_to(jmap.sim_to_real(sim.ready_sim))
            leader.close()
            flog.close()
        arm.close()
        cams.close()
        cv2.destroyAllWindows()
    print(f"{n_saved} pairs -> {out}")


if __name__ == "__main__":
    main()
