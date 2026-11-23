#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --open-mode=append

# Fine-tune SmolVLA (lerobot/smolvla_base) on a LeRobot dataset under lerobot_dataset/.
#   sbatch -J svla_<name> iridis/train_smolvla.sh <dataset name> <run name> <steps> <batch size> [extra lerobot-train flags]
#
# - cameras: the dataset's ext_cam / wrist_cam are renamed to the pretrained model's camera1 /
#   camera2 (camera3 is left empty); the rename is saved in the checkpoint's preprocessor, so
#   eval_policy.py feeds ext_cam / wrist_cam unchanged
# - runs offline: smolvla_base and the SmolVLM2 backbone must be in the HF cache already
#   (download them once on the login node: see iridis/README_smolvla.md)
# - resumable: re-submit the same command after a timeout or preemption (Iridis disables --requeue);
#   it continues from the last checkpoint (saved every 1000 steps)
module load conda/python3
conda activate my-pytorch-env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd ~/iridis101/sim_assets

NAME=$1; RUN=$2; STEPS=$3; BS=$4; shift 4
OUT=outputs/train/$RUN
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
  echo "resuming $RUN from $(readlink $OUT/checkpoints/last)"
  lerobot-train --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json --resume=true
else
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/$NAME \
    --dataset.root=$HOME/iridis101/sim_assets/lerobot_dataset/$NAME \
    --rename_map='{"observation.images.ext_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2"}' \
    --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=$OUT --job_name=$RUN \
    --batch_size=$BS --steps=$STEPS --save_freq=1000 --log_freq=100 \
    --num_workers=8 --wandb.enable=false \
    "$@"
fi
