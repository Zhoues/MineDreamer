# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/minedreamer_backend_MLLMSD.sh
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
python3 imaginator/test/minedreamer_backend_MLLMSD.py \
    --model_name_or_path data/weights/LLM/Vicuna/vicuna-7b-v1.1 \
    --data_path img \
    --ckpt_dir data/weights/minedreamer/MineDreamer-7B \
    --steps 20000 \
    --sd_qformer_version v1.1-7b