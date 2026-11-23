# Do this ONCE
module load conda
mkdir -p /scratch/$USER/conda/pkgs /scratch/$USER/conda/envs
conda config --add envs_dirs /scratch/$USER/conda/envs
conda config --add pkgs_dirs /scratch/$USER/conda/pkgs
conda init

# These steps create an environment. You might repeat this with different environments
conda create --channel defaults -n "my-pytorch-env" python=3.13
conda activate my-pytorch-env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install packaging torchbearer