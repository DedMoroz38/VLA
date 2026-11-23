"""Joint-angle conversion between the sim (MuJoCo) and the real SO-101 (LeRobot).

    sim   MuJoCo qpos / ctrl: radians, 0 = the model's zero pose.
    real  LeRobot with use_degrees=True (the default for so101_follower/leader):
          the 5 arm joints in degrees, 0 = the pose held during lerobot-calibrate;
          the gripper in 0..100 over its calibrated range (0 = closed).

    arm      sim_rad = radians(real_deg + offset_deg)
    gripper  sim_rad = sim_closed + radians(real_pct / 100 * span_deg + offset_deg)

offset_deg per joint: tuned with calib/live_overlay.py (real arm vs sim overlay).
span_deg: the follower's calibrated gripper range, (range_max - range_min) * 360/4096,
since STS3215 servos have 4096 ticks per turn. sim_closed is the sim gripper's lower
joint limit. All of it lives in calib/joint_map.json, which is inside the repo, so
the conversion also works on machines without the arm's calibration file (Iridis).

Refresh joint_map.json after re-tuning or re-calibrating (live_overlay.py also does
this automatically on every change):

    python joint_units.py --update
    python joint_units.py            # print the current map and a round-trip check
"""
import argparse
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MAP_PATH = HERE / "calib/joint_map.json"
OVERLAY_PATH = HERE / "calib/live_overlay.json"
FIT_PATH = HERE / "calib/side_cam_fit.json"
FOLLOWER_CAL = (pathlib.Path.home() /
                ".cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REAL_NAMES = [f"{j}.pos" for j in JOINTS]       # LeRobot's feature names for the real arm
SIM_GRIPPER_CLOSED = -0.174533                  # so101 gripper joint lower limit (rad)


class JointMap:
    def __init__(self, offsets_deg, gripper_span_deg, gripper_closed_rad=SIM_GRIPPER_CLOSED, meta=None):
        self.off = np.array([offsets_deg[j] for j in JOINTS], float)
        self.span = float(gripper_span_deg)
        self.closed = float(gripper_closed_rad)
        self.meta = meta or {}

    @classmethod
    def load(cls, path=MAP_PATH):
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["offsets_deg"], d["gripper_span_deg"], d["gripper_closed_rad"], d)

    def sim_to_real(self, q):
        """(..., 6) sim radians -> (..., 6) LeRobot units (deg x5, gripper 0..100)."""
        q = np.asarray(q, float)
        out = np.degrees(q) - self.off
        out[..., 5] = (np.degrees(q[..., 5] - self.closed) - self.off[5]) / self.span * 100
        return out

    def real_to_sim(self, x):
        """(..., 6) LeRobot units -> (..., 6) sim radians. Inverse of sim_to_real."""
        x = np.asarray(x, float)
        out = np.radians(x + self.off)
        out[..., 5] = self.closed + np.radians(x[..., 5] / 100 * self.span + self.off[5])
        return out

    def real_dict_to_sim(self, d):
        """{'shoulder_pan': deg, ...} or {'shoulder_pan.pos': deg, ...} -> (6,) sim radians."""
        return self.real_to_sim([d[j] if j in d else d[f"{j}.pos"] for j in JOINTS])


def build_map():
    """Offsets from live_overlay.json (else side_cam_fit.json), gripper span from the follower calibration."""
    src = OVERLAY_PATH if OVERLAY_PATH.exists() else FIT_PATH
    offs = {j: 0.0 for j in JOINTS}
    offs.update(json.loads(src.read_text())["joint_offsets_deg"])
    cal = json.loads(FOLLOWER_CAL.read_text())["gripper"]
    span = (cal["range_max"] - cal["range_min"]) * 360 / 4096
    return {
        "description": "sim_rad = radians(real_deg + offset); gripper: sim_rad = gripper_closed_rad + "
                       "radians(real_pct/100 * gripper_span_deg + offset). See sim/joint_units.py.",
        "offsets_deg": {j: round(float(offs[j]), 3) for j in JOINTS},
        "gripper_span_deg": round(span, 3),
        "gripper_closed_rad": SIM_GRIPPER_CLOSED,
        "sources": {"offsets": str(src.relative_to(HERE)), "gripper_span": str(FOLLOWER_CAL),
                    "follower_gripper_range_raw": [cal["range_min"], cal["range_max"]]},
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def update_map(quiet=False):
    d = build_map()
    MAP_PATH.write_text(json.dumps(d, indent=2))
    if not quiet:
        print("wrote", MAP_PATH.relative_to(HERE))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rebuild calib/joint_map.json")
    args = ap.parse_args()
    if args.update or not MAP_PATH.exists():
        update_map()
    jm = JointMap.load()
    print(json.dumps({k: jm.meta[k] for k in ("offsets_deg", "gripper_span_deg", "updated")}, indent=2))
    q = np.array([0.0, -1.0, 1.2, 0.5, -0.3, 0.4])
    r = jm.sim_to_real(q)
    print("sim", q, "-> real", r.round(2), "-> sim", jm.real_to_sim(r).round(6))


if __name__ == "__main__":
    main()
