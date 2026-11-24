"""Read-only health check of both arms (no torque change, no motion).

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/check_motors.py
    /opt/anaconda3/envs/lerobot/bin/python calib/check_motors.py --watch     # live, Ctrl-C to stop

Prints, per joint: leader and follower position (real units) and their difference, and the
follower's torque-enable, status/error bits, load, current, temperature and voltage.
--watch prints both arms' positions (and the leader's raw encoder ticks) live, 5x a second:
move a joint of either arm by hand and its numbers must follow smoothly.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_policy_real as rp  # noqa: E402

LEADER_PORT = "/dev/tty.usbmodem5B790804261"


def watch(f, l):
    names = [j[:8] for j in rp.JOINTS]
    print("leader deg / leader raw ticks / follower deg      (Ctrl-C to stop)")
    print("          " + " ".join(f"{n:>9s}" for n in names))
    while True:
        try:
            pl = l.sync_read("Present_Position", num_retry=2)
            rl = l.sync_read("Present_Position", normalize=False, num_retry=2)
            pf = f.sync_read("Present_Position", num_retry=2)
        except ConnectionError as e:
            print(f"read failed: {e}")
            time.sleep(0.2)
            continue
        print("leader   " + " ".join(f"{pl[j]:9.1f}" for j in rp.JOINTS))
        print("  raw    " + " ".join(f"{rl[j]:9d}" for j in rp.JOINTS))
        print("follower " + " ".join(f"{pf[j]:9.1f}" for j in rp.JOINTS) + "\n")
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    f = SO101Follower(SO101FollowerConfig(port=rp.FOLLOWER_PORT, id="so101_follower")).bus
    l = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="so101_leader")).bus
    f.connect()
    l.connect()
    try:
        if a.watch:
            try:
                watch(f, l)
            except KeyboardInterrupt:
                return
        pf = f.sync_read("Present_Position", num_retry=2)
        pl = l.sync_read("Present_Position", num_retry=2)
        print(f"{'joint':14s} {'leader':>8s} {'follower':>8s} {'diff':>7s}  torque status load current temp volt")
        for j in rp.JOINTS:
            raw = {r: f.read(r, j, normalize=False, num_retry=2) for r in
                   ("Torque_Enable", "Status", "Present_Load", "Present_Current", "Present_Temperature",
                    "Present_Voltage")}
            print(f"{j:14s} {pl[j]:8.1f} {pf[j]:8.1f} {pf[j] - pl[j]:7.1f}  {raw['Torque_Enable']:6d} "
                  f"{raw['Status']:#06b} {raw['Present_Load']:4d} {raw['Present_Current']:7d} "
                  f"{raw['Present_Temperature']:4d} {raw['Present_Voltage'] / 10:4.1f}")
        print("status bits: 1 voltage, 2 sensor, 4 temperature, 8 current, 32 overload (0 = OK)")
    finally:
        f.disconnect(disable_torque=False)
        l.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
