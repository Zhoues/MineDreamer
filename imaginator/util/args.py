import transformers
from dataclasses import dataclass, field



@dataclass
class ModelArguments:
    # config for pretrained clip text model
    clip_path: str = "data/weights/clip/clip-vit-large-patch14"
    clip_hidden_size: int = 768
    clip_max_length: int = 77

    # config for sd
    sd_path: str = "data/weights/sd/stable-diffusion-v1-5"


@dataclass
class DataArguments:
    # Minecraft dataset
    MinecraftDataset_path: str = "data/mllm_diffusion_dataset/goal_drift_dataset"
    MinecraftDataset_resolution_for_SD: int = 256


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    dinov2_proj_dim: int = 16

    # some variants
    is_position_embeddings: bool = False
    is_dino_v2: bool = False
    diffusion_loss_weight: float = 1.0

    # choose dataset
    is_editing: bool = False
    is_editing_more_MB: bool = False
    is_editing_with_seg: bool = False
    is_seg: bool = False

    # 2023-11-17 methods ablation
    is_InstructPix2Pix_231117: bool = False
    is_MagicBrush_231117: bool = False
    is_InstructDiffusion_231117: bool = False