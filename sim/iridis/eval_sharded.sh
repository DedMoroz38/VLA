#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --array=0-3

# One shard of the fixed 100-pose evaluation. Submit with the array size you want
# (it must match the N in --shard I/N below, default 4):
#   sbatch -J eval_<tag> iridis/eval_sharded.sh <tag> <checkpoint pretrained_model dir> [eval_policy.py flags]
# then, once all shards are done:
#   python merge_eval.py eval/<tag>_shard*.json --expect 100 --out eval/<tag>_100.json
module load conda/python3
conda activate my-pytorch-env
export MUJOCO_GL=egl
cd ~/iridis101/sim_assets

TAG=$1; CKPT=$2; shift 2
N=${SLURM_ARRAY_TASK_COUNT:-4}
python eval_policy.py "$CKPT" \
  --scene ~/iridis101/sim_assets/robotstudio_so101/pick_place_scene.xml \
  --shard ${SLURM_ARRAY_TASK_ID}/${N} \
  --out eval/${TAG}_shard${SLURM_ARRAY_TASK_ID}.json \
  --video-dir eval/videos_${TAG} --videos 2 \
  "$@"
