"""Damped-least-squares velocity IK for the SO-101 arm.

The arm's first four joints (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex) are used to resolve a 4-DOF task: the 3D world position of the
gripper site plus one pitch angular-rate about the horizontal axis
perpendicular to the arm's current reach direction. wrist_roll (gripper roll)
and the gripper jaw are driven directly, outside the IK.
"""
import numpy as np
import mujoco

ARM_QVEL_IDX = [0, 1, 2, 3]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex
SITE_NAME = "gripperframe"
BASE_BODY_NAME = "base"


def site_jacobian(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp, jacr


def pitch_axis(data, base_body_id, site_id):
    """Horizontal axis perpendicular to the base->gripper radial direction.

    Rotating about this axis changes the gripper's pitch (tilt) without the
    shoulder_pan joint fighting it, since shoulder_pan's own angular velocity
    is purely about world +z and has ~zero projection onto this axis.
    """
    radial = data.site_xpos[site_id][:2] - data.xpos[base_body_id][:2]
    norm = np.linalg.norm(radial)
    radial_dir = radial / norm if norm > 1e-6 else np.array([1.0, 0.0])
    return np.array([-radial_dir[1], radial_dir[0], 0.0])


class ArmIK:
    def __init__(self, model):
        self.model = model
        self.site_id = model.site(SITE_NAME).id
        self.base_body_id = model.body(BASE_BODY_NAME).id
        n = len(ARM_QVEL_IDX)
        # actuator order matches joint/qpos order for this model (see so101.xml)
        self.qlo = np.array([model.actuator_ctrlrange[i][0] for i in range(n)])
        self.qhi = np.array([model.actuator_ctrlrange[i][1] for i in range(n)])

    def _solve_free(self, Jp_full, Jr_full, idx, lin_vel, pitch_rate, damping):
        """Task-priority (position, then pitch in the null space) restricted
        to the joint columns in idx."""
        Jp = Jp_full[:, idx]
        k = len(idx)
        H_p = Jp.T @ Jp + (damping ** 2) * np.eye(k)
        Jp_pinv = np.linalg.solve(H_p, Jp.T)  # (k, 3), damped pseudo-inverse
        dq_pos = Jp_pinv @ lin_vel

        Jr = Jr_full[idx]
        N = np.eye(k) - Jp_pinv @ Jp  # null space of the position task
        a_row = Jr @ N
        residual = pitch_rate - float(Jr @ dq_pos)
        denom = float(a_row @ a_row) + damping ** 2
        dq_null = N @ a_row * (residual / denom)
        return dq_pos + dq_null

    def step(self, data, lin_vel, pitch_rate, q=None, damping=0.08, max_joint_vel=3.0,
              limit_margin=1e-3):
        """Return joint-angle deltas (per unit time) for ARM_QVEL_IDX.

        lin_vel: desired world-frame linear velocity of the gripper site (3,)
        pitch_rate: desired angular rate about the local pitch axis (scalar)
        q: current joint angles for ARM_QVEL_IDX (defaults to data.qpos there).
           Used for joint-limit handling -- see below.

        Uses strict task-priority (position first, pitch resolved in the
        leftover null space) rather than stacking position (m/s) and
        rotation (rad/s) into one least-squares problem -- those have
        incommensurate units/scales, and naively combining them lets
        whichever task has the larger numeric magnitude dominate and starve
        the other of joint motion.

        This arm is small (short links, modest Jacobian magnitudes), so its
        minimum singular value is ~0.13-0.2 across most of the reachable
        workspace -- there's no "safe" pose where the naive un-damped
        least-squares solution stays well-behaved. damping is set high
        enough to tame that everywhere, and the result is additionally
        clamped to max_joint_vel as a hard backstop near true singularities.

        Joints already at their limit and being pushed further into it are
        dropped from the free set and the task is re-solved over the
        remaining joints (simple active-set approach). Without this, a
        saturated joint's column still participates in the pseudo-inverse as
        if it could keep moving, misallocating effort to the other joints
        and the commanded motion drifts instead of converging.
        """
        n = len(ARM_QVEL_IDX)
        if q is None:
            q = data.qpos[ARM_QVEL_IDX]

        jacp, jacr = site_jacobian(self.model, data, self.site_id)
        axis = pitch_axis(data, self.base_body_id, self.site_id)
        Jp_full = jacp[:, ARM_QVEL_IDX]
        Jr_full = (axis @ jacr)[ARM_QVEL_IDX]

        free = np.ones(n, dtype=bool)
        for _ in range(n):
            idx = np.where(free)[0]
            dq_free = self._solve_free(Jp_full, Jr_full, idx, lin_vel, pitch_rate, damping)
            newly_blocked = False
            for local_i, global_i in enumerate(idx):
                at_lo = q[global_i] <= self.qlo[global_i] + limit_margin and dq_free[local_i] < 0
                at_hi = q[global_i] >= self.qhi[global_i] - limit_margin and dq_free[local_i] > 0
                if at_lo or at_hi:
                    free[global_i] = False
                    newly_blocked = True
            if not newly_blocked:
                break

        dq = np.zeros(n)
        dq[idx] = dq_free
        return np.clip(dq, -max_joint_vel, max_joint_vel)

    def site_pos(self, data):
        return data.site_xpos[self.site_id].copy()

    def approach_dir(self, data):
        """Unit vector the gripper is pointing/reaching along (fingertip direction)."""
        grip_body_id = self.model.body("gripper").id
        v = data.site_xpos[self.site_id] - data.xpos[grip_body_id]
        return v / np.linalg.norm(v)
