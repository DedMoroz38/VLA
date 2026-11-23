#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=01:00:00

# Merge the shard datasets of one export into lerobot_dataset/<name>.
#   sbatch --dependency=afterok:<array job id> iridis/merge_dataset.sh <name> <number of shards>
module load conda/python3
conda activate my-pytorch-env
cd ~/iridis101/sim_assets

NAME=$1; N=${2:-8}
ROOT=$HOME/iridis101/sim_assets/lerobot_dataset
IDS=""; ROOTS=""
for i in $(seq 0 $((N - 1))); do
  IDS="$IDS'local/${NAME}_shard$i',"; ROOTS="$ROOTS'$ROOT/${NAME}_shards/shard$i',"
done
rm -rf $ROOT/$NAME
python ~/robot/lerobot/src/lerobot/scripts/lerobot_edit_dataset.py \
  --new_repo_id local/$NAME --new_root $ROOT/$NAME \
  --operation.type merge \
  --operation.repo_ids "[${IDS%,}]" --operation.roots "[${ROOTS%,}]"
python ~/robot/lerobot/src/lerobot/scripts/lerobot_edit_dataset.py \
  --repo_id local/$NAME --root $ROOT/$NAME --operation.type info
