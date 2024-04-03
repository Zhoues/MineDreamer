# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/inference_MineDreamer.sh
source ~/.bashrc
conda activate imaginator
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:MineDreamer
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/diffusers0202_unet
wandb disabled
export WANDB_DISABLED=true

python3 imaginator/test/inference_single_image_MLLMSD.py \
    --save_dir inference_valid_MineDreamer \
    --ckpt_dir data/weights/minedreamer/imaginator \
    --input_image img/mine_block_log.jpg \
    --text_prompt "look at the sky" \
    --steps 20000 \
    --sd_qformer_version v1.1-7b
