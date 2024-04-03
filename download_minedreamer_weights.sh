# Create the directory structure if it doesn't exist
mkdir -p data/weights

mkdir -p data/weights/minedreamer
mkdir -p data/mllm_diffusion_dataset

mkdir -p data/weights/clip
mkdir -p data/weights/sd
mkdir -p data/weights/LLM



###### MineDreamer ckpt

# Pretrain-QFormer for training Imaginator
git clone https://huggingface.co/Zhoues/Pretrained-QFormer-7B data/weights/minedreamer/Qformer

# Goal Drift Dataset trained InstructPix2Pix Unet
git clone https://huggingface.co/Zhoues/MineDreamer-InstructPix2Pix-Unet data/weights/minedreamer/IP2P

# Trained MineDreamer Imaginator
git clone https://huggingface.co/Zhoues/MineDreamer-7B data/weights/minedreamer/MineDreamer-7B


###### pretrain ckpt

# CLIP
git clone https://huggingface.co/openai/clip-vit-large-patch14 data/weights/clip/clip-vit-large-patch14

# InstructPix2Pix
git clone https://huggingface.co/timbrooks/instruct-pix2pix data/weights/instruct-pix2pix

# Stable-Diffusion
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 data/weights/sd/stable-diffusion-v1-5

# LLM 7B
git clone https://huggingface.co/lmsys/vicuna-7b-v1.1 data/weights/LLM/Vicuna/vicuna-7b-v1.1

# MLLM 7B (including LoRA, we need to remove it)
git clone https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1 data/weights/LLM/LLaVA-Lightning-7B-delta-v1-1
