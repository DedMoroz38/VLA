#!/bin/bash -l
#SBATCH -p scavenger_l4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=00:20:00
module load conda/python3
conda activate my-pytorch-env
export MUJOCO_GL=egl
cd ~/iridis101/sim_assets
python diag_onepose.py "$@" ~/iridis101/sim_assets/robotstudio_so101/pick_place_scene.xml
