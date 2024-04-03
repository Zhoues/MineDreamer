# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/pre_llava.sh
source ~/.bashrc
conda activate imaginator
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:MineDreamer
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/diffusers0202_unet
export PYTHONPATH=$PYTHONPATH:MineDreamer/imaginator/llava

python3 -m imaginator.llava.model.apply_delta \
    --base data/weights/LLM/Vicuna/vicuna-7b-v1.1 \
    --target data/weights/LLM/LLaVA-7B-v1.1 \
    --delta data/weights/LLM/LLaVA-Lightning-7B-delta-v1-1