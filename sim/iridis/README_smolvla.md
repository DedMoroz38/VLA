# SmolVLA fine-tuning on Iridis

One-off setup on the login node (it has internet; compute nodes run offline):

    module load conda/python3 && conda activate my-pytorch-env
    pip show lerobot | head -2              # needs a version with --rename_map and SmolVLA (>= 0.4)
    pip install "lerobot[smolvla]"          # if SmolVLA's extras are missing
    huggingface-cli download lerobot/smolvla_base
    huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct

Data: the exported dataset is copied to `~/iridis101/sim_assets/lerobot_dataset/<name>`:

    rsync -a sim/lerobot_dataset/so101-cube/ em21g23@loginx002.iridis.soton.ac.uk:iridis101/sim_assets/lerobot_dataset/so101-cube/

Train (SmolVLA docs: batch 64, 20k steps):

    sbatch -J svla_cube iridis/train_smolvla.sh so101-cube smolvla_cube 20000 64

Checkpoints every 1000 steps in `outputs/train/<run>/checkpoints/`; preempted jobs requeue and
resume from the last one. Evaluate a checkpoint locally (eval_policy.py reads the camera
rename from the checkpoint):

    python eval_policy.py <ckpt>/pretrained_model --poses eval/cube_poses.json --split test

Checked locally (2026-11-22): a 2-step CPU fine-tune with these exact flags trains (100M of 450M
parameters trainable: the action expert), saves a checkpoint whose preprocessor holds the
camera rename, and eval_policy.py runs it closed-loop on the cube scene.
