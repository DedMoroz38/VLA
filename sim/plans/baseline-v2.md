# Plan v2: a solid simulation baseline

Written 2 Nov 2026, after the first ACT run reached 42% on unseen poses (see `progress_summary.md`).

## What the first attempt actually tells us

The pipeline works end to end. Sim, demos, export, training and closed-loop eval are all built and checked, and evaluation renders match the training frames. What fell short is the **data** and the **training budget**, not the infrastructure. All of the code is reused below.

| Problem | Evidence | Consequence |
|---|---|---|
| Demos are full of pauses | 57–65% of pre-grasp frames have the arm held still (keyboard teleop) | The policy learned to "hold still" above the eraser |
| Only 10 distinct behaviours | 165 generated demos are copies of 10 teleop demos, about 16 each | Every quirk of one human demo is repeated 16 times |
| Undertrained | 40k steps × batch 8 = **2.5 epochs**. Standard ACT sim recipes use about 40 epochs | Loss was still falling; can't separate "bad data" from "not trained" |
| Noisy comparisons | Checkpoints compared on 20 poses, ±20 pp confidence interval | 30k vs 40k "plateau" is not a real finding |
| Slow loop | Export 1 h 47 min + training 2.5 h + eval, all at 640×480 | About 5 h per experiment, so few experiments |
| Two variables changed at once | Human data **and** a new pipeline were tested together | When it failed we couldn't tell which part was at fault |

## Principle for v2

Start from data that is known to be clean, prove the learning pipeline can hit a high number on it, then add the harder ingredients back **one at a time**, measuring each with the same eval.

## Stage 0: Fix the evaluation protocol (half a day)

> **Step 1 done 2 Nov 2026.** Fixed test set: `eval/test_poses.json` (100 poses, seed 20261022, written by `make_eval_poses.py`). The first 50 are v1's exact test poses. `eval_policy.py` uses it by default, and `-n 50` gives the v1 subset. The scripted expert solves 96/100; the 4 it fails are all in the base/tray corner.

Every number from here on uses the same protocol:

- **100 unseen poses**, fixed seed 20261022, uniform over the reachable region. Report the success rate with a 95% CI, failure categories, and success by region (y < 0.06 vs ≥ 0.06).
- Shard eval across parallel Iridis jobs (4 × 25 poses) so it takes about 3 min.
- **Evaluation is not deterministic.** Rerunning the same checkpoint on the same 100 poses changed the outcome of 15 poses (35% vs 34% overall). GPU differences get amplified by contact dynamics. One run on 100 poses is fine for large differences; for a close call (within ~10 points) run it 2–3 times and compare the means.
- Re-run the existing 40k checkpoint on the 100 poses, so v1 has a directly comparable number.

> **Done 2 Nov 2026.** `eval_policy.py --shard I/N` plus `merge_eval.py` (checks no pose is missing or repeated); the report now splits by region and by distance to training data. Job script: `iridis/eval_sharded.sh`. **v1 ACT (40k) on the fixed 100 poses: 35/100 = 35% (95% CI 26–45%); a rerun gave 34%.** By region: y < 0.06 15% (7/48) vs y ≥ 0.06 54% (28/52); |yaw| < 45° 25% vs |yaw| ≥ 45° 44%. So v2 has to beat about 35%, not 42%; the 42% was partly a lucky first half of the set.

## Stage 1: A yaw-aware scripted expert (1–2 days)

> **Done 2 Nov 2026.** Expert succeeds on 195/200 fresh poses (98%). 200 demos at random poses (seed 3) are in `demos/scripted_trimmed/`, all replaying bit-exactly. Still frames before the grasp: 13% (was 57–65%). Besides the yaw-matched roll, it needed a lower grasp (z 0.01), lower approach and lift heights near the base, a via point that carries around the base, and a dry-run choice between the two equivalent wrist rolls.

Replace keyboard teleop as the **primary** data source with a scripted expert. It produces smooth, pause-free, consistent demos at any pose, in unlimited quantity. This is how the ACT and ALOHA sim baselines were built.

Changes to `run_scripted_pick_place` (`pick_place_task.py`):

1. **Roll to match the eraser's yaw**: `roll = GRASP_WRIST_ROLL + wrap(yaw)` plus the shoulder-pan correction already measured in `mimicgen.py` (`MimicTracker.track`). Wrap yaw into (−90°, +90°].
2. **Rotate the wrist during the reach, at hover height**, not after descending. Descending unrotated clips the block (`demo_collection_notes.md`).
3. Shorten the dead time: gripper ramps of 25 steps instead of 50, and no 10 s final hold (`trim_demo.py` cuts the tail anyway).

**Gate:** the expert itself succeeds on **≥ 95%** of 200 random poses, run closed-loop in sim. Below that, fix the expert before generating anything. A policy can't beat its teacher.

Then generate **200 episodes** on random poses with `record_scripted_demo.py`, and trim them.

## Stage 2: A faster loop (1 day, in parallel with Stage 1)

- **Render at 320×240** instead of 640×480. That is 4× fewer pixels, so export, training and eval all get faster. ResNet-18 handles this easily, and the eraser still covers plenty of pixels in the wrist camera. Same resolution in export and eval (`eval_policy.py --width 320 --height 240`).
- **Shard the export** across parallel GPU jobs (e.g. 8 jobs × 25 episodes), then merge. Target under 20 min instead of 1 h 47 min.
- Check throughput (steps/s) in the first training job before choosing the step count for the long run.

> **Done 2 Nov 2026.** `export_lerobot.py --shard I/N`, job scripts in `iridis/`. The 200 scripted demos are exported at 320×240 as `lerobot_dataset/scripted_320` on Iridis: 200 episodes, 76,318 frames, 262 MB. 8 shards took 3–4 min each (about 12 min total, because the account is limited to 3 GPUs at once), against 1 h 47 min before. **Training throughput (ACT, 320×240):** batch 8 = 13.5 steps/s (108 samples/s); batch 32 = 4.6 steps/s (147 samples/s); v1 at 640×480 was about 36 samples/s. Data loading is not the bottleneck. One epoch is 76k samples ≈ 8.7 min at batch 32, so 20 epochs ≈ 48k steps ≈ 3 h and 30 epochs ≈ 72k steps ≈ 4.3 h. **Use batch 32.**

## Stage 3: Prove the learning pipeline (1–2 days)

**3a. Single-pose overfit test (about 1 h of GPU).** Train ACT on 50 scripted episodes at **one fixed pose**, then evaluate at that pose. It must reach **≥ 95%**. If it doesn't, there is a bug in the pipeline (normalisation, camera keys, action alignment, gripper scaling), not a data problem. Fix that first. This is cheap and removes the biggest uncertainty left over from v1.

> **3a done 2 Nov 2026: passed, but only after raising the learning rate.** 50 scripted demos at one pose (0.21, 0.07, 30°, ±3 mm / ±3°), batch 32, 8000 steps (13 epochs). With ACT's default learning rate 1e-5 it got **0/20** at the centre pose and **11/20** on its own training poses; the loss was still 0.053. A trace showed the policy's shoulder pan about 0.04 rad (about 8 mm) off the expert during the descent, so the jaw knocked the eraser sideways: not enough precision, not a pipeline bug. With **learning rate 1e-4** (loss 0.021): **20/20 at the centre pose and 20/20 on the training poses**. So v1's lr of 1e-5 was a likely cause of its poor result, and the full run uses 1e-4. Scripts: `iridis/train_act.sh`, `diag_onepose.py`.

**3b. Full ACT baseline** on the 200 scripted episodes:

- Train for **about 20–30 epochs**, not 2.5. Pick the step count from the measured throughput.
- Use a learning rate with warmup then cosine decay.
- Save a checkpoint every ~10% of training. Evaluate the last 3 checkpoints on the 100 poses.
- Keep `n_action_steps` = chunk size (100), the best setting in v1. Re-test temporal ensembling once, now that the data has no pauses.

**Gate: ≥ 80% on 100 unseen poses.** That is the solid baseline.

> **3b done 3 Nov 2026: 78–80%, right at the gate.** ACT on `scripted_320` (200 demos), batch 32, lr 1e-4, 48k steps (20 epochs, 2 h 57 min, final loss 0.021). Fixed 100 poses at 320×240: 19.2k steps 65%, **28.8k 80%**, 38.4k 72%, **48k 78% (CI 69–85%)**. Two runs of 48k were identical this time. Against **v1's 35%** on the same poses. Inference variants on 48k: re-plan every 50 steps 77%, every 25 steps 80%, temporal ensembling 72%, so none helps; keep the default 100. All failures are "never picked up": the jaw pushes the eraser 7–43 mm during the descent (same imprecision as the low-lr overfit run). 14 of 22 failures are on the far side (y > 0.09); 2 of the 4 poses the expert itself fails are among them. Next lever: data, not inference: recovery demos (3c) and/or more demos.

If it falls short: look at the failure categories first. If failures are mostly grasp misses, add **noise-injected demos** (Stage 3c). Otherwise, look at the failing region and add data there.

**3c (only if needed). Recovery data.** While recording, add small random noise to the **executed** actions but record the expert's **clean** corrective action (the DART technique). The policy then sees states slightly off the ideal path and learns how to get back. This is the standard fix for "policy drifts and misses the grasp".

## Stage 4: Bring human data back, as a controlled comparison (2–3 days)

With a known-good baseline, the human-data question becomes a clean experiment rather than a confound. Train identical ACT models on:

| Dataset | Purpose |
|---|---|
| A. 200 scripted | Baseline (Stage 3) |
| B. 165 MimicGen-style + 11 teleop, **pauses removed inside episodes** | Is the human-sourced data usable once cleaned? |
| C. A + B mixed | Does human data add anything to scripted data? |
| D. Scripted at 50 / 100 / 200 episodes | How much data is needed (scaling curve) |

Removing pauses for B means extending `trim_demo.py` to drop frames where the arm command is constant and the arm is at rest, while keeping short holds around gripper close and open.

If more human demos are needed later, record them with continuous input (gamepad, or an SO-101 leader arm if available) rather than the keyboard. That fixes the pauses at the source.

## Stage 5: SmolVLA

Train SmolVLA on the best dataset from Stage 4, with the same eval protocol, and compare directly with ACT. Only start once ACT has passed the 80% gate, so a SmolVLA failure can't be blamed on the data or the pipeline.

## Summary of gates

| Gate | Pass | If it fails |
|---|---|---|
| Scripted expert | ≥ 95% on 200 poses | Fix the expert; don't generate data yet |
| Single-pose overfit | ≥ 95% | Pipeline bug; debug before anything else |
| ACT on scripted data | ≥ 80% on 100 unseen poses | Failure analysis, then recovery data (3c) |
| Human data (Stage 4) | Report vs A; no pass mark | It's a research result either way |

Rough total: about 1.5–2 weeks to a solid ACT baseline (end of Stage 3), mostly limited by GPU queue time.
