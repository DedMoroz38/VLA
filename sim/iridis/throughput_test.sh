#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=00:20:00

# Time a few hundred ACT training steps, to choose a batch size and step count.
#   sbatch -J tp_bs<B> iridis/throughput_test.sh <dataset name> <batch size> [steps]
module load conda/python3
conda activate my-pytorch-env
cd ~/iridis101/sim_assets

NAME=$1; BS=$2; STEPS=${3:-400}
lerobot-train \
  --dataset.repo_id=local/$NAME \
  --dataset.root=$HOME/iridis101/sim_assets/lerobot_dataset/$NAME \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/tmp/tp_${NAME}_bs${BS} --job_name=tp_${NAME}_bs${BS} \
  --batch_size=$BS --steps=$STEPS --save_freq=100000 --log_freq=50
