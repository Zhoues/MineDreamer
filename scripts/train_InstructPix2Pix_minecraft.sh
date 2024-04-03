# srun -p AI4Good_L --gres=gpu:8 --quotatype=auto bash scripts/train_InstructPix2Pix_minecraft.sh 
source ~/.bashrc
conda activate imaginator
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:MineDreamer
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/diffusers0202_unet

wandb disabled
export WANDB_DISABLED=true
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28458 imaginator/train/InstructPix2Pix_minecraft.py \
    --bf16 True \
    --output_dir output/trained_IP2P \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 True \
    --dataloader_num_workers 16 \
    --dinov2_proj_dim 16 \
    --is_position_embeddings False \
    --is_dino_v2 False \
    --MinecraftDataset_path data/mllm_diffusion_dataset/goal_drift_dataset \
    --clip_path data/weights/clip/clip-vit-large-patch14 \
    --sd_path data/weights/sd/stable-diffusion-v1-5 \
    --deepspeed scripts/zero2_mixed.json \
