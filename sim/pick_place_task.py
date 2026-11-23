"""Task definition for SO-101 eraser pick-and-place: scene, success test, and
a scripted reference policy.

The policy here is the same motion verify_pick_place.py has always driven, but
restructured to run at the *control* rate (one action held for
CONTROL_SUBSTEPS physics steps) instead of re-solving IK every physics step.
That matters because a demonstration is a sequence of actions at a fixed rate:
what gets recorded has to be what a policy could later be asked to reproduce.
"""
import copy
import pathlib

import mujoco
import numpy as np

from demo_io import CONTROL_SUBSTEPS
from ik_solver import ArmIK, pitch_axis

SCENE_PATH = str(pathlib.Path(__file__).parent /
                 "mujoco_menagerie/robotstudio_so101/pick_place_scene.xml")

DOWN = np.array([0.0, 0.0, -1.0])

CONTAINER_XY = np.array([0.18, -0.14])
# Footprint the tray occupies on the tabletop, for keeping the eraser out of it.
CONTAINER_FOOTPRINT_X = (0.115, 0.245)
CONTAINER_FOOTPRINT_Y = (-0.192, -0.088)
# Conservative inner footprint of the tray (true inner half-extents are
# ~0.057 x 0.045); z range accepts the eraser resting on the tray floor.
CONTAINER_TOL_XY = np.array([0.045, 0.035])
CONTAINER_Z_RANGE = (0.0, 0.05)
AT_REST_SPEED = 0.01


def load_scene(scene_path=SCENE_PATH):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    reset_to_ready(model, data)
    return model, data


def reset_to_ready(model, data):
    """Reset to the scene's 'ready' keyframe (or plain zero state if absent)."""
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)


# -- eraser start pose ------------------------------------------------------
#
# Where a demonstration may start the eraser. Measured by sweeping the IK over
# the tabletop (see demo_collection_notes.md): inside this box the gripper
# reaches a straight-down grasp at GRASP_Z to within ~2 mm, and outside it
# wrist_flex saturates at its +/-95 deg limit and the descent stalls
# centimetres short -- (0.10, 0.00) and (0.30, 0.14) both miss by ~40 mm.
ERASER_SAMPLE_X = (0.14, 0.26)
ERASER_SAMPLE_Y = (-0.04, 0.16)
# The eraser is a 70 x 30 x 30 mm box -- a square cross-section -- so yaw and
# yaw+180 deg are the same configuration and +/-90 deg covers every distinct
# one. Yaw is the only pose DOF that changes how it has to be grasped.
ERASER_SAMPLE_YAW = (-np.pi / 2, np.pi / 2)
# The cube repeats every 90 deg, so +/-45 deg covers every distinct orientation.
SAMPLE_YAW = {"eraser": ERASER_SAMPLE_YAW, "cube": (-np.pi / 4, np.pi / 4)}
# Cube start region (2026-11-21): at least 8 cm to the robot's left (y >= 0.08), clear of
# the arm's rest pose (jaws at y ~0.01); the box is the largest one there that a straight-down
# grasp reaches everywhere within 3 mm (IK sweep: error grows past ~28 cm from the base).
CUBE_SAMPLE_X = (0.12, 0.22)
CUBE_SAMPLE_Y = (0.08, 0.18)
SAMPLE_REGION = {"eraser": (ERASER_SAMPLE_X, ERASER_SAMPLE_Y), "cube": (CUBE_SAMPLE_X, CUBE_SAMPLE_Y)}
ERASER_REST_Z = 0.015      # half-height; the tabletop surface is at z=0
# Half-length of the eraser's long axis, for the tray-overlap check.
ERASER_HALF_LEN = 0.035


def task_object(model):
    """Name of the scene's pick object: 'eraser' in the original scenes, 'cube'
    in the cube scene (pick_place_scene_real.xml)."""
    for name in ("eraser", "cube"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    raise ValueError("scene has no 'eraser' or 'cube' body")


def has_container(model):
    """Whether the scene has the tray (the cube scene doesn't)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "container") >= 0


def object_rest_z(model, object_body="eraser"):
    """Height of the object's origin when lying on the table, taken from its
    body pos in the scene XML (0.015 for the eraser, 0.02 for the 40 mm cube)."""
    return float(model.body(object_body).pos[2])


def eraser_qpos_adr(model, object_body="eraser"):
    """qpos index of the eraser's free joint (7 entries: xyz + wxyz quat)."""
    return int(model.jnt_qposadr[model.body(object_body).jntadr[0]])


def set_eraser_pose(model, data, x, y, yaw, object_body="eraser"):
    """Place the eraser flat on the table at (x, y), rotated `yaw` rad about Z.

    Call after reset_to_ready, which restores the keyframe's eraser pose along
    with the arm's. Zeroes the free joint's velocity so the episode starts at
    rest regardless of what the eraser was doing before.
    """
    adr = eraser_qpos_adr(model, object_body)
    dof = int(model.jnt_dofadr[model.body(object_body).jntadr[0]])
    data.qpos[adr:adr + 3] = (x, y, object_rest_z(model, object_body))
    data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def wrap_half_turn(angle):
    """Fold an angle into (-pi/2, pi/2], the eraser's distinct yaw range."""
    return angle - np.pi * np.round(angle / np.pi)


def get_eraser_pose(model, data, object_body="eraser"):
    """Current eraser pose as (x, y, yaw_rad)."""
    adr = eraser_qpos_adr(model, object_body)
    w, _, _, z = data.qpos[adr + 3:adr + 7]
    return float(data.qpos[adr]), float(data.qpos[adr + 1]), float(
        2.0 * np.arctan2(z, w))


def sample_eraser_pose(rng=None, check_tray=True, object_body="eraser"):
    """Draw a random (x, y, yaw) from the reachable region, clear of the tray.
    The yaw range follows the object's symmetry (SAMPLE_YAW)."""
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(100):
        rx, ry = SAMPLE_REGION[object_body]
        pose = (rng.uniform(*rx), rng.uniform(*ry), rng.uniform(*SAMPLE_YAW[object_body]))
        if not eraser_pose_warnings(*pose, check_tray=check_tray):
            return pose
    raise RuntimeError("could not sample a clear eraser pose")   # unreachable


def eraser_pose_warnings(x, y, yaw, check_tray=True):
    """Reasons the pose is a bad place to start a demo. Empty list = fine."""
    problems = []
    if not ERASER_SAMPLE_X[0] <= x <= ERASER_SAMPLE_X[1]:
        problems.append(f"x={x:.3f} outside the reachable range {ERASER_SAMPLE_X}")
    if not ERASER_SAMPLE_Y[0] <= y <= ERASER_SAMPLE_Y[1]:
        problems.append(f"y={y:.3f} outside the reachable range {ERASER_SAMPLE_Y}")
    # Treat the eraser as its axis-aligned bounding box at this yaw.
    reach = ERASER_HALF_LEN * abs(np.sin(yaw)) + ERASER_REST_Z * abs(np.cos(yaw))
    if (check_tray and CONTAINER_FOOTPRINT_X[0] < x < CONTAINER_FOOTPRINT_X[1]
            and y - reach < CONTAINER_FOOTPRINT_Y[1]):
        problems.append("overlaps the tray")
    return problems


def container_xy(model):
    """Centre of the container on the tabletop: the 'container' body's position
    (CONTAINER_XY for the original tray)."""
    return np.array(model.body("container").pos[:2], float)


def container_tolerances(model):
    """(tol_xy, z_range) the success test uses: from the scene's
    <numeric name="container_tol" data="tol_x tol_y zmin zmax"> when it has one
    (the cube scene's container, sim/container/build_container.py), else the
    original tray's CONTAINER_TOL_XY / CONTAINER_Z_RANGE."""
    nid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "container_tol")
    if nid < 0:
        return CONTAINER_TOL_XY, CONTAINER_Z_RANGE
    adr = model.numeric_adr[nid]
    v = model.numeric_data[adr:adr + 4]
    return np.array(v[:2]), (float(v[2]), float(v[3]))


def object_in_container(model, data, object_body="eraser"):
    """True once the object is resting inside the container and has stopped
    moving. Always False in a scene without a container."""
    if not has_container(model):
        return False
    bid = model.body(object_body).id
    pos = data.xpos[bid]
    tol_xy, z_range = container_tolerances(model)
    inside_xy = np.all(np.abs(pos[:2] - container_xy(model)) < tol_xy)
    inside_z = z_range[0] < pos[2] < z_range[1]
    ang = float(np.linalg.norm(data.cvel[bid][:3]))
    nid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "container_rest")
    if nid < 0:                      # original scenes: angular speed only (cvel[:3] is angular)
        at_rest = ang < AT_REST_SPEED
    else:                            # cube scene: <numeric name="container_rest" data="lin_max ang_max">
        lin_max, ang_max = model.numeric_data[model.numeric_adr[nid]:model.numeric_adr[nid] + 2]
        at_rest = float(np.linalg.norm(data.cvel[bid][3:])) < lin_max and ang < ang_max
    return bool(inside_xy and inside_z and at_rest)


class PolicyRunner:
    """Drives the arm at the control rate, calling `on_control_step` once per
    action so a recorder can log exactly the frames that were commanded.

    The hook is called with (data, action) *before* the action takes effect,
    matching the frame convention in demo_io.
    """

    def __init__(self, model, data, on_control_step=None,
                 substeps=CONTROL_SUBSTEPS):
        self.model = model
        self.data = data
        self.ik = ArmIK(model)
        self.substeps = substeps
        self.dt = substeps * model.opt.timestep
        self.on_control_step = on_control_step
        self.q_target = data.ctrl[:6].copy()

    def _apply(self):
        """Log the pending action, then hold it for `substeps` physics steps.

        The trailing mj_forward is not optional. mj_step integrates qpos but
        leaves the derived quantities (xpos, xquat, site_xpos, cvel) describing
        the state *before* that integration, so without it both the IK and the
        recorder would read poses one physics step stale relative to the qpos
        recorded alongside them. Replay does the same forward in the same
        place, which is what keeps the two bit-identical.
        """
        if self.on_control_step is not None:
            self.on_control_step(self.data, self.q_target)
        self.data.ctrl[:6] = self.q_target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _clip(self, value, idx):
        lo, hi = self.model.actuator_ctrlrange[idx]
        return float(np.clip(value, lo, hi))

    def move_to(self, target_pos, max_speed=0.12, pos_tol=0.004,
                max_steps=700, settle_steps=8, chase_pitch=False,
                desired_approach=DOWN, pitch_gain=4.0, stall_steps=60):
        """Servo the gripper site to `target_pos`. Returns True if it arrived.

        chase_pitch=True actively orients the gripper straight down -- only
        reachable near the table (descent phases). During transit at hover
        height, a pitch target that isn't reachable there saturates a joint and
        the position task stalls short of the goal, so transit leaves pitch
        alone.

        Gives up (returns False) once the distance has not improved by 0.5 mm
        in `stall_steps` control steps: a saturated arm otherwise creeps for
        the full `max_steps` (21 s), and in a recorded demo that is 21 s of an
        arm holding still.
        """
        settled = 0
        best, best_step = np.inf, 0
        for step in range(max_steps):
            err = target_pos - self.ik.site_pos(self.data)
            dist = float(np.linalg.norm(err))
            if dist < pos_tol:
                settled += 1
                if settled > settle_steps:
                    return True
            else:
                settled = 0
            if dist < best - 5e-4:
                best, best_step = dist, step
            elif step - best_step > stall_steps:
                return False

            lin_vel = err / max(dist, 1e-6) * min(max_speed, dist * 5.0)

            pitch_rate = 0.0
            if chase_pitch:
                approach = self.ik.approach_dir(self.data)
                axis = pitch_axis(self.data, self.ik.base_body_id, self.ik.site_id)
                cos_a = float(np.clip(np.dot(approach, desired_approach), -1, 1))
                sin_a = float(np.dot(np.cross(approach, desired_approach), axis))
                pitch_rate = float(np.clip(pitch_gain * np.arctan2(sin_a, cos_a),
                                           -2.0, 2.0))

            dq = self.ik.step(self.data, lin_vel, pitch_rate, q=self.q_target[:4])
            for i in range(4):
                self.q_target[i] = self._clip(self.q_target[i] + dq[i] * self.dt, i)
            self._apply()
        return False

    def ramp_actuator(self, idx, value, steps=50):
        """Linearly ramp one actuator's target to `value` over `steps` control
        steps (a jump would be commandable but not demonstrable)."""
        value = self._clip(value, idx)
        start = self.q_target[idx]
        for i in range(steps):
            self.q_target[idx] = start + (value - start) * (i + 1) / steps
            self._apply()

    def set_gripper(self, value, steps=50):
        self.ramp_actuator(5, value, steps)

    def set_wrist_roll(self, value, steps=50):
        self.ramp_actuator(4, value, steps)

    def hold(self, steps):
        """Keep the current action for `steps` control steps (let things settle)."""
        for _ in range(steps):
            self._apply()


GRIPPER_CLOSED = -0.1
GRIPPER_OPEN = 1.2
# Pre-grasp hover and straight-up lift height. Near the base (x < ~0.17) the
# arm cannot get the gripper higher than this without saturating wrist_flex:
# at 0.12 it stalls 10-27 mm short, at 0.08 it arrives within ~2 mm.
APPROACH_Z = 0.08
# Carry / retreat height over the tray, whose walls top out at ~0.07. The
# eraser hangs ~11 mm below the gripper site, so this clears them by ~2 cm.
HOVER_Z = 0.11
# Gripper-site height for the grasp. At 0.02 the fingertips only pinch the top
# ~9 mm of the 30 mm eraser and it slips out on the lift about a quarter of the
# time; at 0.01 the jaws close around its middle (18/24 -> 23/24 lifts over
# random poses and +/-0.1 rad roll error) and still clear the table.
GRASP_Z = 0.01
RELEASE_Z = 0.07
# Aligns the jaw-opening axis with the eraser's long (X) axis for this approach
# pose -- found empirically; without it the jaws close along a diagonal and
# never pinch the block.
GRASP_WRIST_ROLL = 0.8
# The pose GRASP_WRIST_ROLL was tuned at: the keyframe eraser (0.18, 0.14,
# yaw 0), which the arm reaches with this shoulder_pan.
GRASP_REF_YAW = 0.0
GRASP_REF_PAN = -0.787
# Control steps for each wrist-roll / gripper ramp. Long enough that the motion
# is demonstrable, short enough not to fill the demo with a near-still arm.
RAMP_STEPS = 25
# Frames kept after the eraser first counts as settled in the tray.
SETTLED_EXTRA_STEPS = 10
# Arrival tolerance for waypoints in free space (above the eraser, the lift,
# the retreat). Near the base the arm stops 5-7 mm short of them, which is
# harmless -- only the descents need the default 4 mm.
FREE_SPACE_TOL = 0.01
# A straight carry from near the base (radius < ~0.19 m) to the tray passes
# over the base, where wrist_flex saturates and the IK swings shoulder_pan to
# its far limit. Such carries go via a point at this radius instead.
TRANSIT_RADIUS = 0.21


def transit_via(from_xy, to_xy):
    """Via point keeping a carry clear of the base, or None if it already is.

    The base sits at the world origin. If the straight segment comes closer to
    it than TRANSIT_RADIUS, route through the point at that radius halfway (in
    angle) between the two ends.
    """
    seg = to_xy - from_xy
    t = np.clip(-np.dot(from_xy, seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0)
    if np.linalg.norm(from_xy + t * seg) >= TRANSIT_RADIUS:
        return None
    a0, a1 = np.arctan2(from_xy[1], from_xy[0]), np.arctan2(to_xy[1], to_xy[0])
    mid = a0 + 0.5 * (a1 - a0)
    return TRANSIT_RADIUS * np.array([np.cos(mid), np.sin(mid)])


def grasp_wrist_roll(eraser_yaw, pan):
    """wrist_roll that lines the jaws up with an eraser at `eraser_yaw`.

    With the gripper pointing straight down, the jaw axis sits at world azimuth
    `wrist_roll - shoulder_pan + const` (measured for mimicgen.py), so the roll
    has to absorb both the eraser's rotation and the pan the arm reaches it
    with. The eraser is 180-deg symmetric, so the offset is folded into
    +/-90 deg, which keeps the roll inside its joint limits.
    """
    return GRASP_WRIST_ROLL + wrap_half_turn(
        (eraser_yaw - GRASP_REF_YAW) + (pan - GRASP_REF_PAN))


def choose_grasp_roll(runner, target_pos, eraser_yaw):
    """Pick between the two wrist rolls that align the jaws with the eraser.

    The eraser is 180-deg symmetric, so `grasp_wrist_roll` and the same roll
    half a turn away grasp it equally well. They do not reach equally well: the
    gripper site sits off the roll axis, and near the base (x < ~0.18) one of
    the two leaves elbow_flex and wrist_flex saturated 3-4 cm above the eraser
    while the other arrives within 2 mm. Which one depends on the pose, so
    each candidate is dry-run -- roll, then descend -- on a copy of the sim.

    The two are not interchangeable once they both reach, though: the jaws are
    asymmetric while open, and the flipped roll can bring the open moving jaw
    down onto the eraser and knock it centimetres away. So the nominal roll is
    kept whenever its descent arrives without disturbing the eraser, and
    otherwise the candidate with the smallest combined miss is used.
    """
    model = runner.model
    lo, hi = model.actuator_ctrlrange[4]
    roll = grasp_wrist_roll(eraser_yaw, runner.q_target[0])
    candidates = [roll] + [r for r in (roll - np.pi, roll + np.pi) if lo < r < hi]
    eraser = model.body("eraser").id
    start = runner.data.xpos[eraser][:2].copy()

    def miss(candidate):
        """Final site error plus how far the descent pushed the eraser, metres."""
        trial = PolicyRunner(model, copy.copy(runner.data))
        trial.q_target = runner.q_target.copy()
        trial.set_wrist_roll(candidate, steps=RAMP_STEPS)
        trial.move_to(target_pos, chase_pitch=True)
        reach = np.linalg.norm(trial.ik.site_pos(trial.data) - target_pos)
        pushed = np.linalg.norm(trial.data.xpos[eraser][:2] - start)
        return float(reach + pushed)

    misses = []
    for candidate in candidates:
        misses.append(miss(candidate))
        if misses[-1] < 0.006:
            return candidate
    return candidates[int(np.argmin(misses))]


def run_scripted_pick_place(runner, object_body="eraser", settle_steps=330):
    """Execute the reference pick-and-place. Returns a list of (stage, ok).

    Works for any eraser pose in the sampled region: the wrist is rolled to the
    eraser's yaw at hover height, before descending -- a gripper lowered
    unrotated onto a rotated eraser clips it and stalls short. After the
    release the arm holds still until the eraser has settled in the tray (at
    most `settle_steps`), so the demo ends in the state success is judged on.
    """
    model, data = runner.model, runner.data
    object_x, object_y, object_yaw = get_eraser_pose(model, data, object_body)
    object_xy = np.array([object_x, object_y])
    stages = []

    def stage(name, ok):
        stages.append((name, bool(ok)))
        return ok

    grasp_pos = np.array([*object_xy, GRASP_Z])
    stage("reach above eraser", runner.move_to(np.array([*object_xy, APPROACH_Z]),
                                               pos_tol=FREE_SPACE_TOL))
    runner.set_wrist_roll(choose_grasp_roll(runner, grasp_pos, object_yaw),
                          steps=RAMP_STEPS)
    stage("descend to eraser", runner.move_to(grasp_pos, chase_pitch=True))

    runner.set_gripper(GRIPPER_CLOSED, steps=RAMP_STEPS)

    z_before_lift = data.xpos[model.body(object_body).id][2]
    stage("lift", runner.move_to(np.array([*object_xy, APPROACH_Z]),
                                 pos_tol=FREE_SPACE_TOL, settle_steps=0))
    stage("eraser lifted off table",
          data.xpos[model.body(object_body).id][2] - z_before_lift > 0.03)

    drop_xy = container_xy(model)
    via = transit_via(object_xy, drop_xy)
    if via is not None:
        # Pass through without stopping: a loose tolerance and no settling.
        runner.move_to(np.array([*via, HOVER_Z]), pos_tol=0.02, settle_steps=0)
    stage("move over container", runner.move_to(np.array([*drop_xy, HOVER_Z])))
    stage("descend to container",
          runner.move_to(np.array([*drop_xy, RELEASE_Z]), chase_pitch=True))

    runner.set_gripper(GRIPPER_OPEN, steps=RAMP_STEPS)
    stage("retreat", runner.move_to(np.array([*drop_xy, HOVER_Z]),
                                    pos_tol=FREE_SPACE_TOL))

    # Let the eraser settle in the tray; keep recording so the demo ends in the
    # state the success check is evaluated on.
    for _ in range(settle_steps):
        if object_in_container(model, data, object_body):
            break
        runner.hold(1)
    runner.hold(SETTLED_EXTRA_STEPS)
    stage("eraser at rest in container", object_in_container(model, data, object_body))
    return stages
