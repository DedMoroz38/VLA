"""Touch test: is the sim<->real joint conversion (calib/joint_map.json) accurate?

The script picks points on the table, solves in the sim for the joint angles that
put the FIXED JAW TIP straight above each point (pointing down, HOVER mm up),
converts them to real units with joint_units.py, and moves the real follower
there. You mark each point and measure where the real tip actually is. The
script then compares, and fits per-joint offset corrections that explain the
difference.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/touch_test.py --dry-run   # plan + checks only, no arm
    /opt/anaconda3/envs/lerobot/bin/python calib/touch_test.py             # the real test (follower only)

Before you start
  1. Tape a sheet of paper (A3 or two A4) on the table in front of the robot,
     pushed up against the base.
  2. Lay a ruler across the front of the base: it touches the base at two
     points (the base's front bulges forward on both sides of a notch in the
     middle). Draw along the ruler: that is the FRONT LINE.
     Mark the middle of the base on it (halfway between its left and right
     sides) and draw the CENTRELINE from there, perpendicular to the front
     line, running forward. Use a set square.
  3. Have a ruler and a pen ready. Start with the arm folded at rest; it is
     where the arm returns at the end.

During the test, at each point
  - the arm hovers with the fixed jaw tip about HOVER mm above the paper
  - make a small pen dot directly under the tip (look straight down; don't push
    the arm) and number it
  - type the tip's height above the paper in mm and press Enter (empty = skip)
After the arm is back at rest, the script asks for each dot's position:
  - forward: mm from the front line, measured along the centreline direction
  - lateral: mm from the centreline, + = the ROBOT's left (your left when you
    stand behind the robot facing the same way as the arm)

Safety: every path is checked in the sim for table contact before the real arm
moves, the target angles are checked against the real calibrated limits, moves
are slow (<= 25 deg/s), and if a joint lags its target by > 12 deg the arm holds
where it is (torque stays on) and the script stops. The gripper is never moved.
Ctrl-C also holds the arm in place with torque on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import joint_units  # noqa: E402

SCENE = HERE.parent / "mujoco_menagerie/robotstudio_so101/pick_place_scene_real.xml"
FOLLOWER_PORT = "/dev/tty.usbmodem5B790809571"
ARM = joint_units.JOINTS[:5]
# x forward, y = robot's left (m). Spread over the cube's workspace.
DEFAULT_POINTS = [(0.15, 0.00), (0.25, 0.00), (0.20, 0.10), (0.20, -0.10), (0.13, 0.12), (0.27, 0.08)]
UP_SIM = [0.0, -0.55, 0.25, 1.60, 0.0]   # transit waypoint: the scene's "ready" pose (sim rad)
ABOVE_Z = 0.06                      # approach height above each point (straight-down poses run
                                    # into the wrist_flex limit much higher than this)
LIMIT_MARGIN_DEG = 3.0
MEAS_SIGMA_MM = 2.0                 # assumed ruler accuracy, for the fit's error bars
PRIOR_SIGMA_DEG = 3.0               # corrections are expected to be small: a weak pull toward 0
                                    # that stops directions the data can't see from running away
REST_GUESS_REAL = [0.0, -100.0, 95.0, 70.0, 0.0]   # dry-run only: a typical folded rest


# ---------------------------------------------------------------- sim side
class SimArm:
    def __init__(self, jmap):
        self.jm = jmap
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for j in range(self.m.njnt):                       # task object out of the way
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[self.m.jnt_qposadr[j]] = 5.0
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in joint_units.JOINTS]
        self.tip = self.m.geom("fixed_jaw_sph_tip1").id
        self.gripper_body = self.m.body("gripper").id
        self.table = self.m.geom("table_top").id
        base = self.m.body("base").id
        self.robot_geoms = {g for g in range(self.m.ngeom)
                            if self.m.geom_bodyid[g] != base and self._in_robot(self.m.geom_bodyid[g])}
        self.lo = self.m.jnt_range[[self.m.joint(j).id for j in ARM]][:, 0]
        self.hi = self.m.jnt_range[[self.m.joint(j).id for j in ARM]][:, 1]
        self.grip = self.jm.real_to_sim([0, 0, 0, 0, 0, 10])[5]   # nearly closed, for collision checks

    def _in_robot(self, b):
        root = self.m.body("base").id
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def set(self, q5, grip=None):
        for a, v in zip(self.qadr[:5], q5):
            self.d.qpos[a] = v
        self.d.qpos[self.qadr[5]] = self.grip if grip is None else grip
        mujoco.mj_forward(self.m, self.d)

    def tip_pos(self, q5, grip=None):
        self.set(q5, grip)
        return self.d.geom_xpos[self.tip].copy()

    def approach(self):
        v = self.d.geom_xpos[self.tip] - self.d.xpos[self.gripper_body]
        return v / np.linalg.norm(v)

    def ik(self, target, roll=0.0):
        """Joint angles (sim rad, 5) putting the tip at target, pointing straight down."""
        target = np.asarray(target, float)
        radial = target[:2] / max(np.linalg.norm(target[:2]), 1e-6)
        pan = np.arctan2(target[1], target[0])

        def resid(q4):
            self.set(np.r_[q4, roll])
            return np.r_[self.d.geom_xpos[self.tip] - target, 0.1 * float(self.approach()[:2] @ radial)]

        best = None
        for lift, elbow, wrist in [(-0.55, 0.25, 1.6), (-1.0, 0.5, 1.4), (0.0, -0.5, 1.6), (-1.3, 1.2, 0.8)]:
            q4 = np.array([pan, lift, elbow, wrist])
            for _ in range(300):
                r = resid(q4)
                if np.abs(r).max() < 1e-7:
                    break
                J = np.stack([(resid(q4 + 1e-6 * e) - r) / 1e-6 for e in np.eye(4)], 1)
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(4), -J.T @ r)
                q4 = np.clip(q4 + np.clip(dq, -0.2, 0.2), self.lo[:4], self.hi[:4])
            q = np.r_[q4, roll]
            err = np.linalg.norm(self.tip_pos(q) - target)
            if best is None or err < best[1] - 1e-6:
                best = (q, err)
        return best

    def table_contact(self, q5):
        self.set(q5)
        for c in self.d.contact[:self.d.ncon]:
            pair = {c.geom1, c.geom2}
            if self.table in pair and (pair - {self.table}) & self.robot_geoms:
                return True
        return self.d.geom_xpos[self.tip][2] < 0.003

    def path_clear(self, qa, qb, n=60):
        return not any(self.table_contact(qa + (qb - qa) * t) for t in np.linspace(0, 1, n))


def real_limits():
    cal = json.loads(joint_units.FOLLOWER_CAL.read_text())
    lim = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / 4096 for j in ARM])
    lim[4] = 180.0                                   # wrist_roll: full turn
    return lim - LIMIT_MARGIN_DEG


def plan(sim, points, hover):
    """For each point: (above, hover) poses in sim rad, plus the shared UP waypoint."""
    up = np.array(UP_SIM)
    out = []
    for i, (x, y) in enumerate(points):
        qa, ea = sim.ik((x, y, ABOVE_Z))
        qh, eh = sim.ik((x, y, hover))
        if max(ea, eh) > 1e-3:
            raise SystemExit(f"point {i+1} ({x:.3f}, {y:.3f}) is out of reach (IK error {max(ea, eh)*1000:.1f} mm)")
        out.append((qa, qh))
    return up, out


def to_real5(jm, q5, grip_real=0.0):
    return jm.sim_to_real(np.r_[q5, jm.real_to_sim([0, 0, 0, 0, 0, grip_real])[5]])[:5]


def check_plan(sim, jm, up, poses, start_sim):
    lim = real_limits()
    seq = [start_sim, up]
    for qa, qh in poses:
        seq += [qa, qh, qa]
    seq += [up, start_sim]
    for k, q in enumerate(seq[1:-1], 1):            # not the start: the arm is already there
        r = to_real5(jm, q)
        bad = np.abs(r) > lim
        if bad.any():
            j = ARM[int(np.argmax(bad))]
            raise SystemExit(f"waypoint {k}: {j} = {r[ARM.index(j)]:.1f} deg is beyond the real limit "
                             f"+-{lim[ARM.index(j)]:.1f} (incl. {LIMIT_MARGIN_DEG} deg margin)")
    # A folded rest pose may rest on / touch the table, so the first and last
    # quarter of the segments leaving / returning to it aren't checked.
    last = len(seq) - 2
    for k in range(len(seq) - 1):
        t0, t1 = (0.25 if k == 0 else 0.0), (0.75 if k == last else 1.0)
        qa, qb = seq[k], seq[k + 1]
        if not sim.path_clear(qa + (qb - qa) * t0, qa + (qb - qa) * t1):
            raise SystemExit(f"segment {k}->{k+1} would touch the table in the sim -- not moving")
    return seq


# ---------------------------------------------------------------- fit
def fit_offsets(sim, records):
    """Least-squares offset corrections (deg, 5 arm joints) from the measured tip positions."""
    rows = [(np.asarray(r["q_meas_sim"]), r["meas"]) for r in records]

    def data_resid(delta_deg):
        out = []
        for q, meas in rows:
            q6 = q.copy()
            q6[:5] += np.radians(delta_deg)
            p = sim.tip_pos(q6[:5], q6[5]) * 1000
            out += [p[i] - meas[i] for i in range(3) if meas[i] is not None]
        return np.array(out)

    # wrist_roll barely moves the tip in these straight-down poses, so it isn't fitted
    free = [0, 1, 2, 3]
    w = MEAS_SIGMA_MM / PRIOR_SIGMA_DEG

    def resid(delta_deg):                  # data + prior rows (Gaussian prior on the corrections)
        return np.r_[data_resid(delta_deg), w * delta_deg[free]]

    delta = np.zeros(5)
    for _ in range(100):
        r = resid(delta)
        J = np.stack([(resid(delta + 1e-4 * np.eye(5)[i]) - r) / 1e-4 for i in free], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        step = np.clip(step, -2, 2)        # damped: small steps in a nonlinear problem
        delta[free] += step
        if np.abs(step).max() < 1e-4:
            break
    r = resid(delta)
    J = np.stack([(resid(delta + 1e-4 * np.eye(5)[i]) - r) / 1e-4 for i in free], 1)
    cov = MEAS_SIGMA_MM ** 2 * np.linalg.inv(J.T @ J)
    std = np.full(5, np.nan)
    std[free] = np.sqrt(np.diag(cov))
    return delta, std, data_resid(np.zeros(5)), data_resid(delta), free


def paper_to_robot(fwd, lat, front_x_mm, gap_mm=0.0, tilt_mm=0.0, span_mm=300.0):
    """A dot measured on the paper (forward from the edge, lateral from the centreline)
    -> robot x, y (mm). gap_mm: paper edge is that far in front of the base's front line.
    tilt_mm: over span_mm along the edge, the robot-RIGHT end of the edge is that much
    further from the robot than the left end (paper turned a little)."""
    n = np.hypot(span_mm, tilt_mm)
    f_hat = np.array([span_mm, tilt_mm]) / n          # paper 'forward' in robot x, y
    l_hat = np.array([-tilt_mm, span_mm]) / n         # paper 'left'
    o = np.array([front_x_mm + gap_mm, 0.0])
    p = o + (fwd or 0.0) * f_hat + (lat or 0.0) * l_hat
    return float(p[0]) if fwd is not None else None, float(p[1]) if lat is not None else None


def report(sim, records, jm_offsets):
    print("\n pt   predicted tip (x, y, z mm)      measured             error (x, y, z)   radial  tangential")
    for r in records:
        p, m_ = r["pred_mm"], r["meas"]
        err = [m_[i] - p[i] if m_[i] is not None else None for i in range(3)]
        meas_s = " ".join(f"{v:6.1f}" if v is not None else "   -  " for v in m_)
        err_s = " ".join(f"{e:+6.1f}" if e is not None else "   -  " for e in err)
        rt = ""
        if err[0] is not None and err[1] is not None:
            rh = np.array(p[:2]) / np.linalg.norm(p[:2])
            rt = f"  {err[0]*rh[0] + err[1]*rh[1]:+6.1f}  {-err[0]*rh[1] + err[1]*rh[0]:+6.1f}"
        print(f" {r['point']:>2}   {p[0]:6.1f} {p[1]:6.1f} {p[2]:6.1f}     {meas_s}    {err_s}{rt}")
    print("   radial = away from the base (+) / toward it (-); tangential = toward the robot's left (+)")
    n_xy = sum(r["meas"][0] is not None and r["meas"][1] is not None for r in records)
    if n_xy < 3:
        print("\nfewer than 3 dots with forward + lateral measured: no offset fit")
        return None
    delta, std, r0, r1, free = fit_offsets(sim, records)
    rms0, rms1 = np.sqrt(np.mean(r0 ** 2)), np.sqrt(np.mean(r1 ** 2))
    n_z = sum(r["meas"][2] is not None for r in records)
    print(f"\nRMS error with the current offsets: {rms0:.1f} mm  ({len(r0)} measurements, {n_z} heights)")
    print(f"RMS error after fitting corrections: {rms1:.1f} mm")
    print(f"suggested offset corrections (deg, +-1 sigma; ruler {MEAS_SIGMA_MM} mm, prior +-{PRIOR_SIGMA_DEG} deg):")
    new = dict(jm_offsets)
    for i, j in enumerate(ARM):
        if i in free:
            print(f"  {j:<14} {delta[i]:+6.2f} +- {std[i]:.2f}   -> offset {new[j] + delta[i]:+.2f}")
            new[j] = round(new[j] + float(delta[i]), 3)
    if n_z == 0:
        print("  (no heights measured: shoulder_lift / elbow_flex / wrist_flex are only weakly separated)")
    return dict(rms_mm_before=float(rms0), rms_mm_after=float(rms1),
                corrections_deg=dict(zip(ARM, delta.round(3).tolist())), suggested_offsets_deg=new)


def reanalyze(args):
    import copy
    res = json.loads(Path(args.reanalyze).read_text())
    jm = joint_units.JointMap.load()
    sim = SimArm(jm)
    fx = res["front_face_x_mm"]
    flip = set(args.flip_lateral or [])
    records = copy.deepcopy(res["records"])     # the saved measurements themselves are never changed
    for r in records:
        fwd = None if r["meas"][0] is None else r["meas"][0] - fx       # back to the raw ruler readings
        lat = r["meas"][1]
        if lat is not None and r["point"] in flip:
            lat = -lat
        if r["point"] in (args.drop or []):
            fwd = lat = None
            r["meas"][2] = None
        r["meas"][0], r["meas"][1] = paper_to_robot(fwd, lat, fx, args.gap_mm, args.tilt_mm, args.span_mm)
    print(f"re-analysis of {Path(args.reanalyze).name}: lateral sign flipped for {sorted(flip) or 'none'}, "
          f"dropped {args.drop or 'none'}, edge gap {args.gap_mm} mm, tilt {args.tilt_mm} mm over {args.span_mm} mm")
    out = report(sim, records, res["offsets_used"])
    if out:
        res["reanalysis"] = dict(out, flip_lateral=sorted(flip), drop=args.drop or [], gap_mm=args.gap_mm,
                                 tilt_mm=args.tilt_mm, span_mm=args.span_mm)
        Path(args.reanalyze).write_text(json.dumps(res, indent=2))
        print("saved into", Path(args.reanalyze).name, "(nothing applied)")


# ---------------------------------------------------------------- real side
def configure_with_retry(robot, tries=3):
    """robot.configure() writes every motor once with no retry; a single dropped status packet
    ('There is no status packet!') then aborts start-up. Retry a few times before giving up."""
    for k in range(tries):
        try:
            robot.configure()
            return
        except ConnectionError as e:
            if k == tries - 1:
                raise
            print(f"motor bus glitch during configure ({e}); retrying")
            time.sleep(0.3)


def ask_float(prompt):
    while True:
        s = input(prompt).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            print("  a number in mm, or empty to skip")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=FOLLOWER_PORT)
    ap.add_argument("--hover", type=float, default=15.0, help="tip height above the table, mm")
    ap.add_argument("--points", type=float, nargs="+", metavar="X_Y",
                    help="override points: x1 y1 x2 y2 ... in metres (robot frame)")
    ap.add_argument("--dry-run", action="store_true", help="plan and check in the sim only, no arm")
    ap.add_argument("--reanalyze", metavar="JSON", help="redo the analysis of a saved touch_test_*.json")
    ap.add_argument("--flip-lateral", type=int, nargs="*", help="(reanalyze) points whose lateral sign was entered wrong")
    ap.add_argument("--drop", type=int, nargs="*", help="(reanalyze) points to leave out")
    ap.add_argument("--gap-mm", type=float, default=0.0, help="(reanalyze) paper edge is this far in front of the base")
    ap.add_argument("--tilt-mm", type=float, default=0.0,
                    help="(reanalyze) robot-right end of the paper edge is this much further from the robot ...")
    ap.add_argument("--span-mm", type=float, default=300.0, help="(reanalyze) ... over this length along the edge")
    args = ap.parse_args()
    if args.reanalyze:
        return reanalyze(args)

    points = DEFAULT_POINTS if not args.points else list(zip(args.points[::2], args.points[1::2]))
    jm = joint_units.JointMap.load()
    sim = SimArm(jm)
    print(f"joint map: {joint_units.MAP_PATH.name} ({jm.meta.get('updated', '?')}), offsets {jm.meta['offsets_deg']}")
    up, poses = plan(sim, points, args.hover / 1000)

    # the FRONT LINE: a ruler across the front of the base touches its front-most
    # points (two flats either side of a central notch), at table level
    base = sim.m.body("base").id
    vx = []
    for g in range(sim.m.ngeom):
        if sim.m.geom_bodyid[g] == base and sim.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = sim.m.geom_dataid[g]
            v = sim.m.mesh_vert[sim.m.mesh_vertadr[mid]:sim.m.mesh_vertadr[mid] + sim.m.mesh_vertnum[mid]]
            w = v @ sim.d.geom_xmat[g].reshape(3, 3).T + sim.d.geom_xpos[g]
            vx.append(w[w[:, 2] < 0.02, 0])
    front_x = float(np.concatenate(vx).max())

    print(f"\nfront line (ruler across the front of the base) is at x = {front_x*1000:.1f} mm in the sim")
    print(f"{'pt':>3} {'forward':>8} {'lateral':>8}   real target (deg): " + " ".join(f"{j[:8]:>9}" for j in ARM))
    for i, ((x, y), (qa, qh)) in enumerate(zip(points, poses)):
        r = to_real5(jm, qh)
        side = "left" if y > 0 else "right" if y < 0 else "centre"
        print(f"{i+1:>3} {(x - front_x)*1000:7.0f}mm {y*1000:+7.0f}mm ({side:>6})  " + " ".join(f"{v:9.1f}" for v in r))

    if args.dry_run:
        start = jm.real_to_sim(np.r_[REST_GUESS_REAL, 0])[:5]
        check_plan(sim, jm, up, poses, start)
        print("\ndry run: all poses reachable, inside the real limits, and every path clears the table in the sim")
        return

    from capture_poses import Abort, move, present      # noqa: E402  (lerobot import)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id="so101_follower"))
    bus = robot.bus
    bus.connect()
    if not bus.is_calibrated:
        bus.disconnect(disable_torque=False)
        raise SystemExit("follower motors don't match its calibration file -- not moving")

    def read_all():
        p = bus.sync_read("Present_Position")
        return np.array([p[j] for j in joint_units.JOINTS])

    start_real = read_all()
    start_sim = jm.real_to_sim(start_real)[:5]
    check_plan(sim, jm, up, poses, start_sim)
    print("\nplan checked in the sim: reachable, inside the real limits, no table contact.")
    input("Paper taped, centreline drawn, arm at rest? Press Enter to start (Ctrl-C to cancel) ")

    bus.disable_torque()
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
    configure_with_retry(robot)                         # torque on, holding exactly here
    grip_now = start_real[5]
    to5 = lambda q: to_real5(jm, q, grip_now)           # noqa: E731

    records, held = [], False
    try:
        move(bus, to5(up))
        for i, (qa, qh) in enumerate(poses):
            move(bus, to5(qa))
            move(bus, to5(qh))
            meas_real = read_all()
            q_meas = jm.real_to_sim(meas_real)
            pred = sim.tip_pos(q_meas[:5], q_meas[5]) * 1000
            x, y = points[i]
            print(f"\npoint {i+1}: target forward {(x - front_x)*1000:.0f} mm, lateral {y*1000:+.0f} mm "
                  f"({'robot left' if y > 0 else 'robot right' if y < 0 else 'centre'}), height {args.hover:.0f} mm")
            print("  mark a dot under the fixed jaw tip and number it", i + 1)
            h = ask_float("  tip height above the paper (mm, empty = skip): ")
            records.append(dict(point=i + 1, target_mm=[x * 1000, y * 1000, args.hover],
                                real_deg=meas_real.round(2).tolist(), q_meas_sim=q_meas.tolist(),
                                pred_mm=pred.round(2).tolist(), meas=[None, None, h]))
            move(bus, to5(qa))
        move(bus, to5(up))
        move(bus, start_real[:5])
        bus.disable_torque()
        print("\nback at rest, torque released.")
    except (Abort, KeyboardInterrupt) as e:
        hold = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", hold)
        held = True
        print(f"\nSTOPPED ({e or 'Ctrl-C'}). The arm is holding its position with torque ON so it doesn't drop.")
        print("Support it by hand and power-cycle the follower to release it.")
    finally:
        bus.disconnect(disable_torque=False)

    if not records:
        return
    print("\nNow measure each dot (arm out of the way). Empty = not measured.")
    for r in records:
        f = ask_float(f"  dot {r['point']}: forward from the front line (mm): ")
        lat = ask_float(f"  dot {r['point']}: lateral from centreline (mm, + = robot's left): ")
        r["meas"][0] = None if f is None else front_x * 1000 + f
        r["meas"][1] = lat

    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "offsets_used": jm.meta["offsets_deg"],
           "hover_mm": args.hover, "front_face_x_mm": front_x * 1000, "records": records, "aborted": held}
    fit = report(sim, records, jm.meta["offsets_deg"])
    if fit:
        out.update(fit)
        new, rms0 = fit["suggested_offsets_deg"], fit["rms_mm_before"]
        if rms0 <= 5:
            print("\nwithin ~5 mm: the conversion is good enough for grasping a 40 mm cube.")
        if input("\napply the corrected offsets to live_overlay.json + joint_map.json? [y/N] ").strip().lower() == "y":
            st = json.loads(joint_units.OVERLAY_PATH.read_text())
            st["joint_offsets_deg"].update({j: new[j] for j in ARM})
            joint_units.OVERLAY_PATH.write_text(json.dumps(st, indent=2))
            joint_units.update_map()
            print("applied. Re-export the dataset to use them.")
    path = HERE / f"touch_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2))
    print("results saved to", path.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
