import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    # LLM -> Vicuna
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    # config for added token
    num_new_tokens: int = 32

    # config for sd
    sd_model_name_or_path: str = "data/weights/sd/stable-diffusion-v1-5"

    # config for pretrained clip text model
    clip_path: str = "data/weights/clip/clip-vit-large-patch14"
    clip_max_length: int = 77

    # config for qformer that link to sd
    sd_qformer_num_layers: int = 6
    sd_qformer_cross_attention_freq: int = 2
    pretrain_sd_qformer: str = None

    # pretrained model: unet_ckpt
    sd_qformer_version: str = "v1.1-7b"
    SD_QFormer_conversation_33tokens: str = "data/weights/Qformer/checkpoint-100000.bin"
    LLaVA_00001: str = "data/weights/LLM/LLaVA-7B-v1.1/pytorch_model-00001-of-00002.bin"
    LLaVA_00002: str = "data/weights/LLM/LLaVA-7B-v1.1/pytorch_model-00002-of-00002.bin"
    LLaVA_model_path: str = "data/weights/LLM/LLaVA-7B-v1.1"
    unet_ckpt: str = "output/wo_dig_andd_dirt_dataset/unet-10000/adapter_model.bin"



@dataclass
class DataArguments:
    # Minecraft dataset
    MinecraftDataset_path: str = "data/mllm_diffusion_dataset/goal_drift_dataset"
    MinecraftDataset_resolution_ViT: int = 224
    MinecraftDataset_resolution_for_SD: int = 256

    # Instruction tuning template
    editing_template: str = "imaginator/data/ConversationTemplateMinecraft.txt"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    # max_length -> editing=280, LLaVA=2048
    model_max_length: int = 2048
    editing_max_length: int = 512
    mm_projection_length: int = 256

    # training settings
    llm_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    unet_full: bool = False
    is_convert: bool = False
    is_MagicBrush: bool = False
    is_InstructDiffusion: bool = False

    # choose datasets
    is_editing: bool = False
    is_all: bool = False
    is_all_more_editing: bool = False
    is_all_more_editing_20231111: bool = False
    is_all_more_editing_20231114: bool = False