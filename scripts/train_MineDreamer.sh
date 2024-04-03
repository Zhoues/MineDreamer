# srun -p AI4Good_L --gres=gpu:8 --quotatype=auto bash scripts/train_MineDreamer.sh
source ~/.bashrc
conda activate imaginator
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:MineDreamer
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/diffusers0202_unet
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/llava


wandb disabled
export WANDB_DISABLED=true
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28456 imaginator/train/MLLMSD_train.py \
   --model_name_or_path data/weights/LLM/Vicuna/vicuna-7b-v1.1 \
   --sd_qformer_version "v1.1-7b" \
   --bf16 True \
   --tf32 True \
   --output_dir output_mllm/trained_MineDreamer_7B \
   --num_train_epochs 20 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --evaluation_strategy 'no' \
   --save_strategy 'steps' \
   --save_steps 5000 \
   --save_total_limit 3 \
   --learning_rate 1e-5 \
   --lr_scheduler_type 'cosine' \
   --weight_decay 0. \
   --warmup_ratio 0.001 \
   --logging_steps 1 \
   --model_max_length 2048 \
   --gradient_checkpointing True \
   --dataloader_num_workers 16 \
   --ddp_find_unused_parameters True \
   --unet_full True \
   --unet_ckpt data/weights/minedreamer/IP2P/unet-15000/adapter_model.bin \
   --SD_QFormer_conversation_33tokens data/weights/minedreamer/Qformer/checkpoint-100000.bin \
   --MinecraftDataset_path data/mllm_diffusion_dataset/goal_drift_dataset \
   --is_convert False \
   --is_InstructDiffusion True \
   --deepspeed scripts/zero2_mixed.json \