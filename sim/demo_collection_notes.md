# Notes on the 10-demo collection plan

Assessment of: record 1 demo → verify → record 10 across the eraser-config table.

**Outcome:** 11 demos recorded, all replaying bit-exactly. Ten of them drive the
step-3 generator at an ~81% success rate; `teleop_009` is excluded (regrasp —
see below). The advice below is kept as the record of why the set looks as it
does.

## Verdict

The plan is sound and the 1-then-10 structure is right — 10 human demos is also
roughly what MimicGen-style methods use as a source set. Two things to fix
before it can actually run, and a few refinements to the table.

## ~~Blocker~~ Resolved: varying the eraser's start pose

The eraser pose was baked into the `ready` keyframe and `reset_to_ready` always
restored it, so every demo would have started identically. Teleop now takes
`--randomize` / `--eraser X Y YAW_DEG` / `--seed`, with **N** to draw a new pose
and **Space** to retry the current one. The pose is written to episode metadata
as `eraser_start` (also recoverable from `obj_pos[0]` / `obj_quat[0]` /
`mj_state[0]`, which is what step 3's augmentation keys off).

## Measured constraints (checked, not assumed)

**Reachable region for a straight-down grasp at z=0.02.** Swept the IK over the
tabletop; failures are `wrist_flex` saturating at its ±95° limit, not orientation
(gripper stayed <2° off vertical everywhere).

| region | residual | usable? |
| --- | --- | --- |
| x ∈ [0.14, 0.26], y ∈ [−0.04, +0.16] | ~2 mm | yes — keep demos here |
| (0.18, −0.06) | 14 mm | marginal, and right against the tray |
| (0.22, +0.20) | 14 mm | edge |
| (0.10, 0.00) | 40 mm | no — too close to the base |
| (0.30, +0.14) | 42 mm | no — too far |

Avoid y < −0.05 around x ∈ [0.12, 0.24]: that is the tray's own footprint.

**Rotation is not roll-limited.** `wrist_roll` spans [−157°, +163°], and the
tuned grasp roll is 0.8 rad (46°), so a +90° eraser yaw needs 2.37 rad — inside
the limit with margin. The 90° demos should be physically recordable.

**The eraser is 180°-symmetric.** It is a 70 × 30 × 30 mm box with a *square*
cross-section, so yaw θ and θ+180° are the same configuration. Distinct yaw
range is 0–180°, and yaw is the only pose DOF that changes the grasp. Sampling
is therefore uniform over ±90°.

**Align the jaws before descending.** Approaching a rotated eraser with roll
left at 0 makes the gripper clip the block and stall ~8 mm high; rolling the
wrist to match the eraser's yaw (`Q`/`E`) brings the descent back to ~1.4 mm.
Roll has room for it — `wrist_roll` spans [−157°, +163°] against a 46° nominal
grasp roll.

## Suggested edits to the table

- Make one of demos 4–6 a **negative** rotation (−30° to −45°). The object is
  symmetric but the arm is not, so +40° and −40° are genuinely different
  problems; three samples all on the same side wastes one.
- Rows 1–3 ("slightly different positions") are the lowest-value three. Position
  variation is precisely what step 3's augmentation generates for free; rotation
  and approach direction are where a human source demo beats the transform.
  Worth keeping for the cross-check, but don't spend care budget there.
- Rows 9–10 are the highest-value and the vaguest. **Pin down actual XY
  coordinates from the table above before recording**, rather than discovering
  mid-demo that a pose is unreachable.

## Keep these constant across all 10

Augmentation splices demos per object-centric subtask, so the 10 need to be
segmentable the same way: approach → grasp → lift → transport → release.

- No regrasps, no mid-demo corrections, no long pauses. A jittery source demo
  propagates into every trajectory generated from it. *(This one bit:
  `teleop_009` has four gripper transitions instead of two, so there is no
  single grasp event to anchor the object-centric phase on, and
  `generate_demos.py` skips it automatically.)*
- Let the eraser settle in the tray before pressing P — the success test requires
  speed < 0.01, and the demo should end in the state it is judged on.
- Check each one as it lands: `python replay_demo.py "demos/teleop_*.npz"`.
  Anything that isn't bit-exact and `success=True` should be re-recorded.

## One consequence worth knowing

The scripted policy cannot stand in for these. It grasps at the object's XY with
a hardcoded `GRASP_WRIST_ROLL = 0.8` tuned for the default pose, so it ignores
eraser yaw entirely and will fail on rows 4–8. The rotated demos genuinely have
to be human — and the scripted fallback source trajectory covers position
variation only.
