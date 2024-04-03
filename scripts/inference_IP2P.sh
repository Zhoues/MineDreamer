# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/inference_IP2P.sh
source ~/.bashrc
conda activate imaginator
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:MineDreamer
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/diffusers0202_unet

wandb disabled
export WANDB_DISABLED=true


# No need to change sd_path and pipeline_path 
python3 imaginator/test/inference_single_image_IP2P.py \
    --sd_path data/weights/sd/stable-diffusion-v1-5 \
    --pipeline_path data/weights/instruct-pix2pix \
    --save_dir inference_valid_IP2P \
    --pretrain_unet data/weights/minedreamer/IP2P/unet-15000/adapter_model.bin \
    --input_image img/mine_block_log.jpg \
    --text_prompt "chop a tree" \