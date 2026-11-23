"""Move the real follower arm through calibration poses and grab a side-cam frame at each.

    /opt/anaconda3/envs/lerobot/bin/python calib/capture_poses.py --port /dev/tty.usbmodemXXXX
    ... --plan calib/poses_20261119/plan.json --out calib/poses_20261119   # another plan / folder

Poses come from calib/poses/plan.json (LeRobot degrees). Every move goes via
the plan's 'up' waypoint, and every path was checked in the sim to stay well
clear of the table. Output: calib/poses/<name>.png and calib/poses/joints.json
with the joint angles actually *measured* at each frame.

Safety:
- the goal register is set to the present position before torque is enabled,
  so nothing jumps
- moves are linearly interpolated at <= 25 deg/s; the gripper is never commanded
- if any joint lags its target by more than ABORT_DEG (blocked or colliding),
  the arm holds where it is and the script stops with torque still on
- at the end the arm returns to its starting (rest) pose and torque is released
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from cameras import Grabber  # calib/cameras.py: open the side camera by name, not index

HERE = Path(__file__).resolve().parent
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
STEP_DEG, RATE_HZ, ABORT_DEG, SETTLE_S = 0.5, 50, 12.0, 1.5


class Abort(RuntimeError):
    pass


def present(bus):
    p = bus.sync_read("Present_Position")
    return np.array([p[j] for j in ARM])


def move(bus, target):
    start = present(bus)
    target = np.asarray(target, float)
    n = max(1, int(np.ceil(np.abs(target - start).max() / STEP_DEG)))
    for i, t in enumerate(np.linspace(0, 1, n + 1)[1:], 1):
        goal = start + (target - start) * t
        bus.sync_write("Goal_Position", dict(zip(ARM, goal)))
        time.sleep(1 / RATE_HZ)
        if i % 10 == 0 or i == n:
            err = np.abs(present(bus) - goal)
            if err.max() > ABORT_DEG:
                hold = present(bus)
                bus.sync_write("Goal_Position", dict(zip(ARM, hold)))
                raise Abort(f"{ARM[err.argmax()]} lags its target by {err.max():.1f} deg -- holding position")
    time.sleep(SETTLE_S)


def grab(cap):
    _, n0 = cap.get()
    t0 = time.time()
    while cap.get()[1] < n0 + 15 and time.time() - t0 < 3:   # let ~15 fresh frames pass (exposure settles)
        time.sleep(0.02)
    return cap.get()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--camera", default="C920", help="side camera name fragment, or an OpenCV index")
    ap.add_argument("--only", nargs="*", help="subset of pose names")
    ap.add_argument("--plan", default=str(HERE / "poses/plan.json"), help="pose plan (LeRobot degrees)")
    ap.add_argument("--out", default=str(HERE / "poses"), help="folder for the frames + joints.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text())
    names = args.only or list(plan["poses"])

    cap = Grabber(args.camera)
    print("camera:", cap.name)

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id="so101_follower"))
    bus = robot.bus
    bus.connect()
    if not bus.is_calibrated:
        raise SystemExit("motor calibration does not match the calibration file -- not moving")
    bus.disable_torque()
    p0 = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", p0)               # hold exactly here when torque comes on
    robot.configure()                                  # PID etc.; re-enables torque
    rest = present(bus)
    print("torque on, holding rest pose:", np.round(rest, 1))

    out = {}
    try:
        grab(cap)
        cv2.imwrite(str(out_dir / "rest.png"), grab(cap))
        out["rest"] = {**dict(zip(ARM, present(bus).round(2).tolist())),
                       "gripper": round(bus.sync_read("Present_Position")["gripper"], 2)}
        move(bus, plan["up"])
        for name in names:
            print(f"-> {name} {plan['poses'][name]}")
            move(bus, plan["poses"][name])
            cv2.imwrite(str(out_dir / f"{name}.png"), grab(cap))
            meas = bus.sync_read("Present_Position")
            out[name] = {k: round(v, 2) for k, v in meas.items()}
            print("   measured", {k: round(v, 1) for k, v in meas.items()})
            move(bus, plan["up"])
        print("returning to rest")
        move(bus, rest)
        bus.disable_torque()
        print("done, torque released at rest")
    except Abort as e:
        print("ABORT:", e)
        print("Arm is holding its position with torque ON. Not releasing it, so it doesn't drop.")
    finally:
        (out_dir / "joints.json").write_text(json.dumps(out, indent=2))
        cap.close()
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
