#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=08:00:00

# Train ACT on a LeRobot dataset under lerobot_dataset/.
#   sbatch -J act_<name> iridis/train_act.sh <dataset name> <run name> <steps> <batch size> [extra lerobot-train flags]
# Checkpoints every ~10% of the run in outputs/train/<run name>/checkpoints/.
module load conda/python3
conda activate my-pytorch-env
cd ~/iridis101/sim_assets

NAME=$1; RUN=$2; STEPS=$3; BS=$4; shift 4
lerobot-train \
  --dataset.repo_id=local/$NAME \
  --dataset.root=$HOME/iridis101/sim_assets/lerobot_dataset/$NAME \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/$RUN --job_name=$RUN \
  --batch_size=$BS --steps=$STEPS --save_freq=$((STEPS / 10)) --log_freq=200 \
  "$@"
