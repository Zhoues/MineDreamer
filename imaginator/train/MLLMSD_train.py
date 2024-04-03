import os
import pathlib
import torch
import torch.nn as nn

import transformers
from imaginator.util.args_mllm import ModelArguments, DataArguments, TrainingArguments
from imaginator.util.dataset import MLLM_Minecraft_Dataset, DataCollatorForLLaVADataset

from imaginator.model.MLLMSD_model import LLMSD
from peft import LoraConfig
from imaginator.util.llm_util import get_peft_state_maybe_zero_3
from imaginator.util.trainer import LLMSDTrainer_Dreamer, safe_save_model_for_hf_trainer, save_llama_lora_config
from imaginator.model.LoraLLaMAUnetPeftModel import get_peft_model
from diffusers.models.attention_processor import AttnProcessor2_0



def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.model_name_or_path = os.path.abspath(model_args.model_name_or_path)
    model_args.sd_model_name_or_path = os.path.abspath(model_args.sd_model_name_or_path)
    model_args.clip_path = os.path.abspath(model_args.clip_path)
    model_args.SD_QFormer_conversation_33tokens = os.path.abspath(model_args.SD_QFormer_conversation_33tokens)
    model_args.LLaVA_00001 = os.path.abspath(model_args.LLaVA_00001)
    model_args.LLaVA_00002 = os.path.abspath(model_args.LLaVA_00002)
    model_args.LLaVA_model_path = os.path.abspath(model_args.LLaVA_model_path)
    model_args.unet_ckpt = os.path.abspath(model_args.unet_ckpt)
    data_args.MinecraftDataset_path = os.path.abspath(data_args.MinecraftDataset_path)
    data_args.editing_template = os.path.abspath(data_args.editing_template)


    local_rank = training_args.local_rank
    model_ = LLMSD.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_.config.use_cache = False
    # LLMSD: sum([p.nelement() for p in model_.parameters()]) -> 6,738,415,616

    # load first stage pre-training from corresponding LLaVA version
    sd_qformer_version = model_args.sd_qformer_version
    if sd_qformer_version == "v1.1-7b" or "v1.1-13b":
        LLaVA_model_path = model_args.LLaVA_model_path
        # init and freeze vit image encoder -> CLIP-ViT
        model_.init_visual_features_extractor(LLaVA_model_path=LLaVA_model_path, sd_qformer_version=sd_qformer_version)
        model_.vision_tower.requires_grad_(False)
        model_.vision_tower.to(torch.float32)
        # sum([p.nelement() for p in model_.vision_tower.parameters()]) -> 303,179,776

    # init llm tokenizer -> LlamaTokenizer
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    # setup new llm tokens -> conversation system num_new_tokens=33: "<img>"(system message) + " <img_0> ... <img_31>" -> len(llm_tokenizer)=32033/original:32000
    model_.setup_tokens_for_conversation(llm_tokenizer, num_new_tokens=model_args.num_new_tokens, tune_new_embeddings=True, editing_template=data_args.editing_template, editing_max_length=training_args.editing_max_length)

    dtype = torch.bfloat16

    # freeze mm_projector
    for p in model_.mm_projector.parameters():
        p.requires_grad = False
    model_.mm_projector.to(torch.float32)
    # model_.vision_tower.dtype -> torch.float32
    # model_.mm_projector.weight.dtype -> torch.float32

    # LLaMA lora hyper-parameters
    lora_attention_dim_llama = 8
    lora_alpha_llama = 16
    lora_target_modules_llama = ['q_proj', 'v_proj']
    lora_dropout_llama = 0.05
    task_type_llama = "CAUSAL_LM"
    lora_bias_llama = 'none'
    llama_lora_config = LoraConfig(
        r=lora_attention_dim_llama,
        lora_alpha=lora_alpha_llama,
        target_modules=lora_target_modules_llama,
        lora_dropout=lora_dropout_llama,
        task_type=task_type_llama,
        bias=lora_bias_llama
    )
    save_llama_lora_config(llama_lora_config, training_args.output_dir)
    model_.model = get_peft_model(model_.model, llama_lora_config)
    model_.model.print_trainable_parameters()
    print('LoraLLaMA_model -> lora training')
    # trainable params: 4,194,304 || all params: 6,611,681,280 || trainable%: 0.06343778265125327

    # embed_tokens -> requires_grad
    for p in model_.get_model().embed_tokens.parameters():
        p.requires_grad = True

    # init LLM with LoRA and put on bfloat16 -> RuntimeError: Expected q_dtype == torch::kFloat16 || ((is_sm8x || is_sm90) && q_dtype == torch::kBFloat16) to be true, but got false
    model_.model.to(dtype)
    model_.model.embed_tokens.to(torch.float32)
    model_.lm_head.to(torch.float32)

    # init q-former that link SD
    model_.init_sd_qformer(
        num_query_token=model_args.clip_max_length,
        num_hidden_layers=model_args.sd_qformer_num_layers,
        cross_attention_freq=model_args.sd_qformer_cross_attention_freq
    )

    # init and freeze vae -> "runwayml/stable-diffusion-v1-5" -> no training(unet&vae) 
    model_.init_sd_vae_unet(model_args.sd_model_name_or_path)
    model_.vae.requires_grad_(False)
    model_.vae.to(dtype)

    # load first stage pre-training from corresponding LLaVA version
    if sd_qformer_version == "v1.1-7b" or "v1.1-13b":
        SD_QFormer_conversation_33tokens = model_args.SD_QFormer_conversation_33tokens
        LLaVA_00001 = model_args.LLaVA_00001
        LLaVA_00002 = model_args.LLaVA_00002
        # load pretrained -> SD_QFormer_conversation_33tokens, LLaVA_00001, LLaVA_00002
        model_.load_pretrain_LLaVA_ckpt_v1_1(SD_QFormer_conversation_33tokens=SD_QFormer_conversation_33tokens,
                                             LLaVA_00001=LLaVA_00001, LLaVA_00002=LLaVA_00002)
        print(SD_QFormer_conversation_33tokens, LLaVA_00001, LLaVA_00002)

    # align with InstructPix2Pix hugging-face
    in_channels = 8
    out_channels = model_.unet.conv_in.out_channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, model_.unet.conv_in.kernel_size, model_.unet.conv_in.stride, model_.unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(model_.unet.conv_in.weight)
        model_.unet.conv_in = new_conv_in
    model_.unet.set_attn_processor(AttnProcessor2_0())
    model_.unet.to(torch.float32)

    # load pretrained unet from InstructPix2Pix baseline
    unet_full = training_args.unet_full
    # unet key need to be change
    model_.load_pretrain_from_unet(unet_ckpt=model_args.unet_ckpt, is_convert=training_args.is_convert,
                                   is_MagicBrush=training_args.is_MagicBrush)
    
    if unet_full == True:
        model_.unet.requires_grad_(True)
        print('Fine-tuning total unet...')
    else:
        # init sd-unet -> DreamBooth LoRA SD-unet training example default...
        lora_attention_dim_sd = 16
        lora_alpha_sd = 27
        UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
        lora_dropout_sd = 0.0
        lora_bias_sd = 'none'
        unet_lora_config = LoraConfig(
            r=lora_attention_dim_sd,
            lora_alpha=lora_alpha_sd,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=lora_dropout_sd,
            bias=lora_bias_sd
        )
        model_.unet = get_peft_model(model_.unet, unet_lora_config)
        model_.unet.print_trainable_parameters()
        model_.unet.to(torch.float32)
        print('model.unet -> lora training')
        print('Training unet with LoRA...')
        # trainable params: 1,594,368 || all params: 861,115,332 || trainable%: 0.18515150535027286

    # init CLIP for null-text embeddings
    model_.init_CLIP_text_encoder(CLIP_path=model_args.clip_path)

    # setup loss weight
    model_.config.llm_loss_weight = training_args.llm_loss_weight
    model_.config.diffusion_loss_weight = training_args.diffusion_loss_weight

    # check model dtype
    print("1.model.vision_tower.dtype: ", model_.vision_tower.dtype)
    print("2.model.mm_projector.dtype: ", model_.mm_projector.weight.dtype)
    print("3.1.model.model.model(LLaMA).embed_tokens.dtype: ", model_.model.embed_tokens.weight.dtype)
    print("3.2.model.model.model(LLaMA).dtype: ", model_.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype, model_.model.layers[0].self_attn.q_proj.weight.dtype)
    print("3.3.model.lm_head.dtype: ", model_.lm_head.weight.data.dtype)
    print("4.1.model.sd_query_tokens.dtype: ", model_.sd_query_tokens.data.dtype)
    print("4.2.model.sd_qformer.dtype: ", model_.sd_qformer.dtype)
    print("5.1.model.vae.dtype: ", model_.vae.dtype)
    print("5.2.model.unet.dtype: ", model_.unet.dtype)
    print("model.get_input_embeddings().dtype: ", model_.get_input_embeddings().weight.data.dtype)
    # 1.model.vision_tower.dtype:  torch.float32
    # 2.model.mm_projector.dtype:  torch.float32
    # 3.1.model.model.model(LLaMA).embed_tokens.dtype: torch.float32
    # 3.2.model.model.model(LLaMA).dtype: torch.bfloat16, torch.bfloat16
    # 3.3.model.lm_head.dtype: torch.float32
    # 4.1.model.sd_query_tokens.dtype: torch.float32
    # 4.2.model.sd_qformer.dtype: torch.float32
    # 5.1.model.vae.dtype: torch.bfloat16
    # 5.2.model.unet.dtype: torch.float32
    # model.get_input_embeddings().dtype: torch.float32

    params_no_grad = [n for n, p in model_.named_parameters() if not p.requires_grad]
    params_requires_grad = [n for n, p in model_.named_parameters() if p.requires_grad]
    print(params_requires_grad)
    print(sum([p.nelement() for p in model_.parameters()]))

    ####################################################################################
    Minecraft_train_dataset = MLLM_Minecraft_Dataset(
        Minecraft_Dataset_path=data_args.MinecraftDataset_path,
        Minecraft_Dataset_resolution_ViT=data_args.MinecraftDataset_resolution_ViT,
        Minecraft_Dataset_resolution_for_SD=data_args.MinecraftDataset_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)

    # DataCollatorForLLaVADataset
    data_collator_train_dataset = DataCollatorForLLaVADataset(LLM_tokenizer=llm_tokenizer)

    # Check dataset
    Minecraft_train_dataloader = torch.utils.data.DataLoader(Minecraft_train_dataset, batch_size=1, num_workers=8)
    print('Checking Merged-Dataset train dataset...')
    index = 0
    for step, batch_data in enumerate(Minecraft_train_dataloader):
        if int(batch_data['is_editing_task'][0].item()) == 0:
            print(batch_data['original_img'].shape, batch_data['original_img'].dtype)  # FloatTensor=float32
            print(batch_data['original_img_for_vae'].shape, batch_data['original_img_for_vae'].dtype)  # FloatTensor=float32
            print(batch_data['edited_img'].shape, batch_data['edited_img'].dtype)  # FloatTensor=float32
            print(batch_data['input_ids'], batch_data['input_ids'].shape, batch_data['input_ids'].dtype)  # LongTensor=int64
            print(batch_data['input_attention_mask'], batch_data['input_attention_mask'].shape, batch_data['input_attention_mask'].dtype)  # torch.bool
            print(batch_data['generated_caption_targets'], batch_data['generated_caption_targets'].shape, batch_data['generated_caption_targets'].dtype)  # LongTensor=int64
            print(batch_data['generated_caption_encoder_attention_mask'], batch_data['generated_caption_encoder_attention_mask'].shape, batch_data['generated_caption_encoder_attention_mask'].dtype)  # torch.bool
            print(batch_data['is_editing_task'], batch_data['is_editing_task'].shape, batch_data['is_editing_task'].dtype)
            # [bs, 3, 224, 224], [bs, 3, 256, 256], [bs, 3, 256, 256]
            # [bs, (editing_max_length-mm_projection_length)=256], [bs, (editing_max_length-mm_projection_length)=256], [bs, (editing_max_length-mm_projection_length)=256], [bs, (editing_max_length-mm_projection_length)=256]
            index = index + 1
        else:
            print(batch_data['original_img'].shape, batch_data['original_img'].dtype)  # FloatTensor=float32
            print(batch_data['original_img_for_vae'].shape, batch_data['original_img_for_vae'].dtype)  # FloatTensor=float32
            print(batch_data['edited_img'].shape, batch_data['edited_img'].dtype)  # FloatTensor=float32
            print(batch_data['input_ids'], batch_data['input_ids'].shape, batch_data['input_ids'].dtype)  # LongTensor=int64
            print(batch_data['input_attention_mask'], batch_data['input_attention_mask'].shape, batch_data['input_attention_mask'].dtype)  # torch.bool
            print(batch_data['generated_caption_targets'], batch_data['generated_caption_targets'].shape, batch_data['generated_caption_targets'].dtype)  # LongTensor=int64
            print(batch_data['generated_caption_encoder_attention_mask'], batch_data['generated_caption_encoder_attention_mask'].shape, batch_data['generated_caption_encoder_attention_mask'].dtype)  # torch.bool
            print(batch_data['is_editing_task'], batch_data['is_editing_task'].shape, batch_data['is_editing_task'].dtype)  # FloatTensor=float32
            index = index + 1
        if index == 2:
            break

    data_module_ = dict(train_dataset=Minecraft_train_dataset, eval_dataset=None, data_collator=data_collator_train_dataset)
    trainer = LLMSDTrainer_Dreamer(model=model_, tokenizer=llm_tokenizer, args=training_args, **data_module_)

    # trainer need to check whether have pretrained checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # savve trainer_state.json
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # imaginator -> train_LLaMALoRA.py
    state_dict = get_peft_state_maybe_zero_3(
        model_.named_parameters(), lora_bias_llama)
    if training_args.local_rank == 0:
        model_.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()

