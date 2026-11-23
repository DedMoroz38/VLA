"""MimicGen-style augmentation: turn a few human demos into many.

The idea (Mandlekar et al., MimicGen): a pick-and-place demonstration is a
sequence of *object-centric* subtasks. The part of the trajectory that reaches
for and grasps the eraser is meaningful relative to the eraser; the part that
carries it to the tray is meaningful relative to the tray. Re-express each
segment in its reference object's frame, substitute a new object pose, and you
get a new trajectory that does the same thing to an object that has moved.

Segmentation here is automatic, from the recorded gripper command:

    phase A  [0, lift)        reach, grasp and lift  -- referenced to the ERASER
    phase B  [lift, end)      carry, release, retreat -- referenced to the TRAY

The cut is placed at the top of the lift rather than at the grasp itself, so
the two phases meet in free space at hover height. The bridge between them is a
straight interpolation, which would drag the eraser across the table if the
segments were joined down at grasp height instead.

Why the transform is simple here. The SO-101 has five arm joints, so it cannot
track an arbitrary 6-DOF pose -- but these are top-down grasps, where the
gripper points along world -Z, and in that configuration `wrist_roll` *is*
rotation about world Z. So a change in the eraser's yaw is absorbed exactly by
a roll offset, and the object-centric transform reduces to a planar rotation of
the waypoints plus that offset. It follows that this module is only valid in
the top-down regime; tilted approaches would need the real SE(3) machinery and
a 6-DOF arm to execute it.

The eraser is a box with a square cross-section, so yaw and yaw+180 deg are the
same configuration. Yaw deltas are therefore wrapped into (-90, +90] deg, which
also keeps the roll offset inside `wrist_roll`'s limits -- without the wrap a
-80 deg source and a +80 deg target would ask for 160 deg of extra roll and
saturate the joint.
"""
import dataclasses

import mujoco
import numpy as np

from demo_io import CONTROL_SUBSTEPS, DemoRecorder, TASK_DESCRIPTION
from ik_solver import pitch_axis
from pick_place_task import (
    DOWN, ERASER_REST_Z, PolicyRunner, eraser_qpos_adr, object_in_container,
    reset_to_ready, sample_eraser_pose, set_eraser_pose, wrap_half_turn,
)

# Fraction of the gripper command's range below which the jaws count as closed.
# The release in a human demo often only re-opens part way (0.60 against a 1.20
# start in teleop_003), so a midpoint threshold misses it entirely.
GRIPPER_CLOSED_FRAC = 0.2
# Phase A ends once the eraser has risen this fraction of its eventual lift.
LIFT_FRAC = 0.8
TRACK_MAX_SPEED = 0.35     # m/s ceiling on commanded gripper speed
BRIDGE_SPEED = 0.12        # m/s for the interpolated hop between phases
MIN_BRIDGE_FRAMES = 12
SETTLE_STEPS = 330         # matches the scripted policy: let the eraser come to rest


@dataclasses.dataclass
class Plan:
    """A retargeted trajectory: control-rate waypoints plus where its phases are."""
    pos: np.ndarray            # (N, 3) gripper site positions, world frame
    roll: np.ndarray           # (N,) wrist_roll targets, before pan correction
    pan: np.ndarray            # (N,) source shoulder_pan, for that correction
    grip: np.ndarray           # (N,) gripper targets
    d_yaw: float               # roll offset applied, radians
    lead: int                  # waypoints of lead-in before phase A
    grasp: int                 # index at which the jaws close
    bridge: int                # index at which the A->B bridge starts

    def __len__(self):
        return len(self.pos)


@dataclasses.dataclass
class Segmentation:
    """Where a source demo's phases begin and end."""
    grasp: int
    lift: int
    release: int
    n: int
    ok: bool = True
    reason: str = ""

    @property
    def phase_a(self):
        return slice(0, self.lift)

    @property
    def phase_b(self):
        return slice(self.lift, self.n)


def segment_demo(demo):
    """Split a demo into its eraser-referenced and tray-referenced phases.

    Returns a Segmentation with ok=False (and a reason) if the gripper opens
    and closes more than once -- a regrasp or a fumbled pick has no single
    well-defined grasp event, and augmenting from it would bake the correction
    into every generated trajectory.
    """
    grip = demo.action[:, 5]
    threshold = grip.min() + GRIPPER_CLOSED_FRAC * (grip.max() - grip.min())
    closed = grip < threshold
    transitions = np.flatnonzero(np.diff(closed.astype(int)) != 0) + 1
    n = len(demo)

    if len(transitions) != 2:
        return Segmentation(-1, -1, -1, n, ok=False,
                            reason=f"{len(transitions)} gripper transitions "
                                   f"(expected 2: one grasp, one release)")
    grasp, release = int(transitions[0]), int(transitions[1])

    obj_z = demo.obj_pos[:, 2]
    lift_height = obj_z[grasp:release].max() - obj_z[0]
    if lift_height < 0.03:
        return Segmentation(grasp, -1, release, n, ok=False,
                            reason=f"eraser only rose {lift_height * 1000:.0f} mm "
                                   f"-- the grasp did not take")
    risen = obj_z[grasp:release] - obj_z[0] >= LIFT_FRAC * lift_height
    lift = grasp + int(np.argmax(risen))
    return Segmentation(grasp, lift, release, n)


def demo_eraser_start(demo):
    """The eraser's (x, y, yaw) at the start of a demo, from its arrays.

    Read from obj_pos/obj_quat rather than the `eraser_start` metadata, which
    only exists on demos recorded after that field was added.
    """
    w, _, _, z = demo.obj_quat[0]
    return (float(demo.obj_pos[0][0]), float(demo.obj_pos[0][1]),
            float(2.0 * np.arctan2(z, w)))


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _bridge(from_pos, to_pos, speed=BRIDGE_SPEED):
    """Straight-line waypoints from one pose to another at `speed`."""
    gap = to_pos - from_pos
    frames = max(MIN_BRIDGE_FRAMES,
                 int(np.ceil(np.linalg.norm(gap) /
                             (speed * CONTROL_SUBSTEPS * 0.005))))
    t = np.linspace(0.0, 1.0, frames + 1)[1:, None]
    return from_pos + t * gap, frames


def retarget(demo, seg, target_pose, start_pos=None):
    """Adapt a source demo to a new eraser pose.

    Returns (pos, roll, grip) waypoint arrays at the control rate: a lead-in
    from where the arm actually is, phase A rotated and translated into the new
    eraser frame, a bridge across the gap to phase B, then phase B unchanged in
    world coordinates because the tray has not moved. The roll offset applies
    throughout -- changing it mid-carry would twist the eraser in the jaws.

    `start_pos` is the gripper's real starting position. Rotating phase A about
    the eraser moves its *first* waypoint too, so without a lead-in the arm
    begins the episode displaced from the trajectory it is tracking, chases it
    all the way down, and swats the eraser off the table instead of grasping
    it. Pass the arm's actual pose after reset.
    """
    src_x, src_y, src_yaw = demo_eraser_start(demo)
    tgt_x, tgt_y, tgt_yaw = target_pose
    d_yaw = wrap_half_turn(tgt_yaw - src_yaw)
    R = rot_z(d_yaw)
    src_origin = np.array([src_x, src_y, ERASER_REST_Z])
    tgt_origin = np.array([tgt_x, tgt_y, ERASER_REST_Z])

    a, b = seg.phase_a, seg.phase_b
    pos_a = (R @ (demo.ee_pos[a] - src_origin).T).T + tgt_origin
    pos_b = demo.ee_pos[b].copy()

    # Lead-in: close the gap between where the arm is and where the retargeted
    # trajectory begins, with the jaws already rolled and opened for the grasp.
    if start_pos is None:
        pos_lead, lead = np.empty((0, 3)), 0
    else:
        pos_lead, lead = _bridge(np.asarray(start_pos, dtype=float), pos_a[0])

    # Bridge phase A's (moved) end to phase B's (fixed) start. Both sit at
    # hover height, so a straight line between them stays in free space.
    pos_bridge, frames = _bridge(pos_a[-1], pos_b[0])

    def stitch(column, bridge_value):
        return np.concatenate([np.full(lead, column[a][0]), column[a],
                               np.full(frames, bridge_value(column)),
                               column[b]])

    roll = stitch(demo.action[:, 4], lambda c: c[b][0]) + d_yaw
    pan = stitch(demo.action[:, 0], lambda c: c[b][0])
    grip = stitch(demo.action[:, 5], lambda c: c[a][-1])
    pos = np.concatenate([pos_lead, pos_a, pos_bridge, pos_b])
    return Plan(pos=pos, roll=roll, pan=pan, grip=grip, d_yaw=d_yaw, lead=lead,
                grasp=lead + seg.grasp, bridge=lead + len(pos_a))


class TrajectoryRunner(PolicyRunner):
    """Tracks a control-rate waypoint sequence, holding the gripper top-down.

    One control step per waypoint, so a generated episode keeps the timing and
    length of the demo it came from, and records as the same kind of object.

    Each waypoint's joint command is solved to convergence on a scratch MjData
    rather than by taking a single velocity-IK step against the live state. A
    single step per waypoint is a proportional chase of a moving reference and
    always lags it -- measured at 22-29 mm mean during the approach, against a
    30 mm-wide eraser, so the jaws closed beside the block as often as on it,
    having nudged it out of position on the way down. Converging per waypoint
    also reproduces the source's own relationship between command and achieved
    pose: there too the position targets led the actual joint angles.
    """

    SOLVE_ITERS = 15
    # Backstop on how far a single control step may move the joint command, so
    # a hard-to-reach waypoint cannot make the recorded action jump.
    MAX_CMD_RATE = 2.0        # rad/s

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scratch = mujoco.MjData(self.model)

    def _pitch_rate(self, data, gain):
        """Rate about the local pitch axis that levels the gripper downward."""
        approach = self.ik.approach_dir(data)
        axis = pitch_axis(data, self.ik.base_body_id, self.ik.site_id)
        cos_a = float(np.clip(np.dot(approach, DOWN), -1.0, 1.0))
        sin_a = float(np.dot(np.cross(approach, DOWN), axis))
        return float(np.clip(gain * np.arctan2(sin_a, cos_a), -2.0, 2.0))

    def _solve(self, target_pos, iters, pitch_gain):
        """Joint command whose forward kinematics put the site on target_pos."""
        scratch = self._scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        q = self.q_target[:4].copy()
        for _ in range(iters):
            scratch.qpos[:4] = q
            mujoco.mj_forward(self.model, scratch)
            err = target_pos - scratch.site_xpos[self.ik.site_id]
            if np.linalg.norm(err) < 1e-4:
                break
            dq = self.ik.step(scratch, err / self.dt,
                              self._pitch_rate(scratch, pitch_gain), q=q)
            for i in range(4):
                q[i] = self._clip(q[i] + dq[i] * self.dt, i)
        return q

    def track(self, pos, roll, pan, grip, pitch_gain=4.0):
        """Follow the waypoints, holding the jaws at the source's world azimuth.

        With the gripper pointing straight down, the jaw-opening axis sits at
        world azimuth `wrist_roll - shoulder_pan + const` (measured). Moving the
        eraser to a new position changes the pan the arm reaches it with, so
        replaying the source's roll unchanged lands the jaws correctly placed
        but rotated by the pan difference -- they close beside the eraser and
        it never lifts. Re-deriving roll from the commanded pan each step holds
        the azimuth where the retargeted grasp needs it.
        """
        limit = self.MAX_CMD_RATE * self.dt
        for k in range(len(pos)):
            # The first waypoint may start far from the arm's pose; afterwards
            # each solve only has to correct a step's worth of motion.
            q = self._solve(pos[k], self.SOLVE_ITERS * (4 if k == 0 else 1),
                            pitch_gain)
            for i in range(4):
                step = float(np.clip(q[i] - self.q_target[i], -limit, limit))
                self.q_target[i] = self._clip(self.q_target[i] + step, i)
            self.q_target[4] = self._clip(
                roll[k] + (self.q_target[0] - pan[k]), 4)
            self.q_target[5] = self._clip(grip[k], 5)
            self._apply()

    def tracking_error(self, pos_target):
        return float(np.linalg.norm(self.ik.site_pos(self.data) - pos_target))


def generate_episode(model, demo, seg, target_pose, record=True):
    """Execute `demo` retargeted to `target_pose`. Returns (recorder, info)."""
    data = mujoco.MjData(model)
    reset_to_ready(model, data)
    set_eraser_pose(model, data, *target_pose)

    recorder = DemoRecorder(model) if record else None
    runner = TrajectoryRunner(model, data,
                              on_control_step=recorder.capture if record else None)
    plan = retarget(demo, seg, target_pose, start_pos=runner.ik.site_pos(data))
    runner.track(plan.pos, plan.roll, plan.pan, plan.grip)
    runner.hold(SETTLE_STEPS)

    adr = eraser_qpos_adr(model)
    success = object_in_container(model, data)
    info = {
        "source_demo": demo.path.name,
        "eraser_start": {"x": float(target_pose[0]), "y": float(target_pose[1]),
                         "yaw_rad": float(target_pose[2]),
                         "yaw_deg": float(np.degrees(target_pose[2]))},
        "yaw_delta_deg": float(np.degrees(plan.d_yaw)),
        "num_waypoints": int(len(plan)),
        "final_tracking_error_m": runner.tracking_error(plan.pos[-1]),
        "final_eraser_pos": [float(v) for v in data.qpos[adr:adr + 3]],
        "success": bool(success),
    }
    return recorder, info


def usable_sources(demos):
    """Split loaded demos into (usable, [(demo, reason), ...]) by segmentation."""
    usable, rejected = [], []
    for demo in demos:
        seg = segment_demo(demo)
        (usable if seg.ok else rejected).append(
            (demo, seg) if seg.ok else (demo, seg.reason))
    return usable, rejected
