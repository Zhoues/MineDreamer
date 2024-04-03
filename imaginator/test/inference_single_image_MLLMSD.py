import argparse
import os
import random
import torch
from PIL import Image
import transformers
from imaginator.util.dataset import generate_LLaVA_image
import PIL
from SDPipeIP2P_variant1 import SDIP2PPipeline_variant1
from imaginator.model.MLLMSD_model import LLMSD

torch.set_grad_enabled(False)

editing_template = os.path.abspath('imaginator/data/ConversationTemplateMinecraft.txt')

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    

    # Unet and LLM with LoRA
    adapter_name = 'default'
    ckpt_dir = args.ckpt_dir
    LLM_sub_dir = ckpt_dir + '/LLM-' + f'{args.steps}'
    embeddings_qformer_Path = ckpt_dir + '/embeddings_qformer/checkpoint-' + f'{args.steps}' + '_embeddings_qformer.bin'


    # "v1.1-7b" -> 30098/40007
    sd_qformer_version = args.sd_qformer_version
    if sd_qformer_version == "v1.1-7b":
        model_name_or_path = "data/weights/LLM/Vicuna/vicuna-7b-v1.1"
        LLaVA_model_path = "data/weights/LLM/LLaVA-7B-v1.1"
        LLaVA_00002_weights = "data/weights/LLM/LLaVA-7B-v1.1/pytorch_model-00002-of-00002.bin"
        SD_IP2P_path = "data/weights/instruct-pix2pix"
    # "v1.1-13b" -> 40005
    elif sd_qformer_version == "v1.1-13b":
        model_name_or_path = "data/weights/LLM/Vicuna/vicuna-13b-v1.1"
        LLaVA_model_path = "data/weights/LLM/LLaVA-13B-v1.1"
        LLaVA_00002_weights = "data/weights/LLM/LLaVA-13B-v1.1/pytorch_model-00002-of-00002.bin"
        SD_IP2P_path = "data/weights/instruct-pix2pix"

    # Load LLMSD
    model_ = LLMSD.from_pretrained(
        os.path.abspath(model_name_or_path),
        cache_dir=None
    )

    # init llm tokenizer -> LlamaTokenizer
    model_max_length = 512
    cache_dir = None
    LLM_tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.abspath(model_name_or_path),
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )

    LLM_tokenizer.pad_token = LLM_tokenizer.unk_token

    # init CLIP-ViT feature extractor
    model_.init_visual_features_extractor(LLaVA_model_path=os.path.abspath(LLaVA_model_path), sd_qformer_version=sd_qformer_version)

    # setup new llm tokens -> conversation system num_new_tokens=35: "<img>"(system message) + 32001='<im_start>', 32002='<im_end>' + " <img_0> ... <img_31>" -> len(llm_tokenizer)=32035
    editing_max_length = 512
    num_new_tokens = 32
    model_.setup_tokens_for_conversation(
        LLM_tokenizer,
        num_new_tokens=num_new_tokens,
        tune_new_embeddings=True,
        editing_template=editing_template,
        editing_max_length=editing_max_length)

    # init q-former that link SD
    model_.init_sd_qformer(
        num_hidden_layers=6
    )

    # Add LoRA for LLaMA
    # Load LLM with lora checkpoint -> type(model_.model)
    from imaginator.test.PeftForLoRA import PeftModel_for_LLM
    model_.model = PeftModel_for_LLM.from_pretrained(model_.model, LLM_sub_dir, adapter_name=adapter_name)
    LLM_sub_dir = LLM_sub_dir + '/adapter_model.bin'
    model_.load_pretrained_LLaMA_for_inference(pretrained_LLaMA=LLM_sub_dir)
    # pretrained checkpoint for SD-QFormer
    model_.load_pretrained_for_inference(pretrain_model=embeddings_qformer_Path, LLaVA_00002_weights=os.path.abspath(LLaVA_00002_weights))

    # inference preparation
    print('LLM vocabulary size:', LLM_tokenizer.vocab_size)
    model_.to(dtype=torch.float32, device="cuda")
    model_.eval()

    # loading inference type
    original_image_ViT_resolution = 224

    total_save_path = args.save_dir
        
    # prepare image for llava
    original_img_path = os.path.abspath(args.input_image)

    input_image = PIL.Image.open(original_img_path)

    instruction = args.text_prompt
    
    original_image_for_ViT = PIL.Image.open(original_img_path).convert('RGB')
    original_image_for_ViT = original_image_for_ViT.resize((original_image_ViT_resolution, original_image_ViT_resolution), resample=Image.Resampling.BICUBIC)
    original_image_for_ViT = model_.vision_tower.image_processor.preprocess(original_image_for_ViT, return_tensors='pt')['pixel_values'].to(dtype=torch.float32, device="cuda")
    CLIP_image_features_llm_input = model_.vision_tower(original_image_for_ViT)
    CLIP_image_features_llm_input = model_.mm_projector(CLIP_image_features_llm_input)
    # [1, mm_projection_length=256, LLM_hidden_size=4096]

    # original image for SD
    original_image_for_SD = PIL.Image.open(original_img_path)
    original_image_for_SD = PIL.ImageOps.exif_transpose(original_image_for_SD)
    original_image_for_SD = original_image_for_SD.convert("RGB")

    both_condition_embeddings = \
        generate_LLaVA_image(CLIP_image_features_llm_input, instruction, LLM_tokenizer, model_, args.temperature, editing_template)

        
    del model_
    torch.cuda.empty_cache()

    # generated image by LLMSD -> image editing
    # Load SD-v1.5

    pipe = SDIP2PPipeline_variant1.from_pretrained(
        SD_IP2P_path,
        torch_dtype=torch.float16,
        safety_checker=None)
    pipe = pipe.to("cuda")

    # load unet checkpoint
    unet_sub_dir = ckpt_dir + '/unet-' + f'{args.steps}'
    unet_ckpt = unet_sub_dir + '/adapter_model.bin'
    unet_ckpt = torch.load(unet_ckpt)
    unet_ckpt_new = {}
    for k, v in unet_ckpt.items():
        if 'unet.' in k:
            unet_ckpt_new[k[len('unet.'):]] = v
    pipe.unet.load_state_dict(unet_ckpt_new, strict=True)
    pipe.unet.to(dtype=torch.float16, device="cuda")
    pipe.unet.eval()
    print('Loading unet checkpoint:', pipe.unet.load_state_dict(unet_ckpt_new, strict=True))
 
    # Minecraft testing inference pipeline
    final_save_dir =total_save_path

    idx = 0
    input_image.save(os.path.join(final_save_dir, f'{(idx + 1):04d}_input.png'))

    text_guidance_scale_list = [7.5]
    image_guidance_scale_list = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

    seed = random.randint(0, 100000)
    generator = torch.Generator("cuda").manual_seed(seed)
    for text_ in text_guidance_scale_list:
        for image_ in image_guidance_scale_list:
            text_guidance_scale = text_
            image_guidance_scale = image_
            image_output = pipe(
                prompt_embeds=both_condition_embeddings,
                image=original_image_for_SD,
                num_inference_steps=100,
                guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator).images[0]
            image_output.save(os.path.join(final_save_dir, f'{(idx + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
    print('Editing image %d' % (idx + 1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. ./vicuna-7b-v1
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="data/weights/LLM/Vicuna/vicuna-7b-v1.1",
    )
    # 2. save_dir
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
    )
    # 3. ckpt_dir
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        required=True,
    )
    # 4. steps
    parser.add_argument(
        '--steps',
        type=int,
        required=True,
    )
    # 5. temperature
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
    )
    # 6. input_image path
    parser.add_argument(
        '--input_image',
        type=str,
        required=True,
    )
    # 7. text_prompt
    parser.add_argument(
        '--text_prompt',
        type=str,
        required=True,
    )
    # 8. sd_qformer_version
    parser.add_argument(
        '--sd_qformer_version',
        type=str,
        default="v1.1-7b"
    )
    args = parser.parse_args()
    main()
