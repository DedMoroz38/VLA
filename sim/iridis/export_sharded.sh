#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --array=0-7

# One shard of a LeRobot export: renders every Nth episode into its own dataset.
#   sbatch -J exp_<name> iridis/export_sharded.sh <name> "<demo glob>" <width> <height>
# Then merge with iridis/merge_dataset.sh (after the whole array finishes).
module load conda/python3
conda activate my-pytorch-env
export MUJOCO_GL=egl
cd ~/iridis101/sim_assets

NAME=$1; GLOB=$2; W=${3:-320}; H=${4:-240}
N=${SLURM_ARRAY_TASK_COUNT:-8}
python export_lerobot.py "$GLOB" \
  --scene ~/iridis101/sim_assets/robotstudio_so101/pick_place_scene.xml \
  --shard ${SLURM_ARRAY_TASK_ID}/${N} \
  --repo-id local/${NAME}_shard${SLURM_ARRAY_TASK_ID} \
  --root ~/iridis101/sim_assets/lerobot_dataset/${NAME}_shards/shard${SLURM_ARRAY_TASK_ID} \
  --width $W --height $H --vcodec h264 --overwrite
