import os
import pathlib
import torch
import torch.nn as nn
import transformers

from imaginator.util.trainer import LLMSDTrainer, safe_save_model_for_hf_trainer
from imaginator.util.args import ModelArguments, DataArguments, TrainingArguments
from imaginator.util.dataset import Minecraft_Dataset

# import which model...
from imaginator.model.InstructPix2Pix_model import InstructPix2Pix_model
from diffusers.models.attention_processor import AttnProcessor2_0



def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.sd_path = os.path.abspath(model_args.sd_path)
    model_args.clip_path = os.path.abspath(model_args.clip_path)
    data_args.MinecraftDataset_path = os.path.abspath(data_args.MinecraftDataset_path)

    local_rank = training_args.local_rank
    InstructPix2Pix_model_ = InstructPix2Pix_model(
        SD_path=model_args.sd_path,
        CLIP_path=model_args.clip_path
    )

    # dtype
    bf16_ = torch.bfloat16
    fp32_ = torch.float32

    # initialize Stable Diffusion
    is_position_embeddings = training_args.is_position_embeddings
    InstructPix2Pix_model_.init_sd_vae_unet(is_position_embeddings=is_position_embeddings,
                                            diffusion_loss_weight=training_args.diffusion_loss_weight)
    InstructPix2Pix_model_.vae.requires_grad_(False)
    InstructPix2Pix_model_.vae.to(bf16_)

    # initialize CLIP text encoder
    InstructPix2Pix_model_.init_CLIP_text_encoder()
    InstructPix2Pix_model_.CLIP_text_encoder.to(bf16_)
    InstructPix2Pix_model_.CLIP_text_encoder.requires_grad_(False)
    CLIP_tokenizer = InstructPix2Pix_model_.CLIP_tokenizer

    ####################################################################################
    # https://huggingface.co/docs/diffusers/optimization/torch2.0 -> align with InstructPix2Pix hugging-face unet
    
    in_channels = 8
    out_channels = InstructPix2Pix_model_.unet.conv_in.out_channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, InstructPix2Pix_model_.unet.conv_in.kernel_size, InstructPix2Pix_model_.unet.conv_in.stride, InstructPix2Pix_model_.unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(InstructPix2Pix_model_.unet.conv_in.weight)
        InstructPix2Pix_model_.unet.conv_in = new_conv_in
    InstructPix2Pix_model_.unet.set_attn_processor(AttnProcessor2_0())
    InstructPix2Pix_model_.unet.requires_grad_(True)
    InstructPix2Pix_model_.unet.to(fp32_)

    # check model dtype
    print("1.vae.dtype: ", InstructPix2Pix_model_.vae.dtype)
    print("2.unet.dtype: ", InstructPix2Pix_model_.unet.dtype)
    print("3.CLIP text-encoder.dtype: ", InstructPix2Pix_model_.CLIP_text_encoder.dtype)
    # 1.vae.dtype:  torch.bfloat16
    # 2.unet.dtype:  torch.float32
    # 3.CLIP text-encoder.dtype:  torch.bfloat16

    params_no_grad = [n for n, p in InstructPix2Pix_model_.named_parameters() if not p.requires_grad]
    params_requires_grad = [n for n, p in InstructPix2Pix_model_.named_parameters() if p.requires_grad]
    print(params_requires_grad)
    print(sum([p.nelement() for p in InstructPix2Pix_model_.parameters()]))
    # no dino_v2: 1,066,246,827 // with dino_v2: 1,152,885,691

    ####################################################################################
    Minecraft_train_dataset = Minecraft_Dataset(
        Minecraft_Dataset_path=data_args.MinecraftDataset_path,
        Minecraft_Dataset_resolution_for_SD=data_args.MinecraftDataset_resolution_for_SD,
        CLIP_tokenizer=CLIP_tokenizer)

    Minecraft_train_dataloader = torch.utils.data.DataLoader(Minecraft_train_dataset, batch_size=1, num_workers=8)
    print(Minecraft_train_dataset, Minecraft_train_dataloader)
    print('Checking Minecraft train dataset...', len(Minecraft_train_dataset))
    index = 0
    for step, batch_data in enumerate(Minecraft_train_dataloader):
        # batch_data.keys() -> dict_keys(['original_img', 'edited_img', 'input_ids'])
        print(batch_data['original_img'].shape, batch_data['original_img'].dtype)  # FloatTensor=float32
        print(batch_data['edited_img'].shape, batch_data['edited_img'].dtype)  # FloatTensor=float32
        print(batch_data['input_ids'], batch_data['input_ids'].shape, batch_data['input_ids'].dtype)  # LongTensor=int64
        # [bs, 3, SD_resolution, SD_resolution], [bs, 3, SD_resolution, SD_resolution], [bs, CLIP_length=77], [bs, 3, 224, 224]
        index = index + 1
        if index == 1:
            break

    data_module = dict(train_dataset=Minecraft_train_dataset, eval_dataset=None)
    trainer = LLMSDTrainer(model=InstructPix2Pix_model_, tokenizer=CLIP_tokenizer, args=training_args, **data_module)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

