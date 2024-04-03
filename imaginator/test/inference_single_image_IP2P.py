import argparse
import numpy as np
import os
import random
import torch
from PIL import Image
torch.set_grad_enabled(False)
import random
import torch.nn as nn
from IP2P_pipeline import StableDiffusionInstructPix2PixPipeline


def main():

    os.makedirs(args.save_dir, exist_ok=True)

    # Load InstructPix2Pix SD-v1.5
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(os.path.abspath(args.pipeline_path),
                                                                  torch_dtype=torch.float32,
                                                                  safety_checker=None)
    pipe = pipe.to("cuda")

    # load dir
    unet_ckpt = os.path.abspath(args.pretrain_unet)
    unet_ckpt = torch.load(unet_ckpt, map_location="cpu")

    SD_path = os.path.abspath(args.sd_path)
    from diffusers import UNet2DConditionModel
    pipe.unet = UNet2DConditionModel.from_pretrained(SD_path, subfolder="unet")

    # align with IP2P
    dino_image_features = None
    in_channels = 8
    # align with InstructPix2Pix hugging-face
    out_channels = pipe.unet.conv_in.out_channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, pipe.unet.conv_in.kernel_size, pipe.unet.conv_in.stride, pipe.unet.conv_in.padding)
        pipe.unet.conv_in = new_conv_in

    # load new unet checkpoint
    unet_ckpt_new = {}
    for k, v in unet_ckpt.items():
        if 'unet.' in k:
            unet_ckpt_new[k[len('unet.'):]] = v
    pipe.unet.load_state_dict(unet_ckpt_new, strict=True)
    pipe.unet.to(dtype=torch.float32, device="cuda")
    pipe.unet.eval()
    print('Loading unet checkpoint:', pipe.unet.load_state_dict(unet_ckpt_new, strict=True))
    # Loading unet checkpoint: <All keys matched successfully>

    image_path = os.path.abspath(args.input_image)

    # test 1 time
    pair_id = 0
    
    # 1. Get input image
    input_image = Image.open(image_path)    
    
    # 2. convert input image into torch style
    Minecraft_Dataset_source_img = Image.open(image_path)   
    Minecraft_Dataset_source_img = np.transpose(Minecraft_Dataset_source_img, (2, 0, 1)) # (3, 256, 256)

    Minecraft_Dataset_source_img = torch.tensor(Minecraft_Dataset_source_img)
    Minecraft_Dataset_source_img = 2 * (Minecraft_Dataset_source_img / 255) - 1

    # 3. Edited Prompt input_ids
    text_prompt = args.text_prompt

    input_image.save(os.path.join(save_dir, f'{(pair_id + 1):04d}_input.png'))

    text_guidance_scale_list = [7.5]
    image_guidance_scale_list = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

    seed = random.randint(0, 100000)
    generator = torch.Generator("cuda").manual_seed(seed)
    text_guidance_scale_list = text_guidance_scale_list
    image_guidance_scale_list = image_guidance_scale_list
    for text_ in text_guidance_scale_list:
        for image_ in image_guidance_scale_list:
            text_guidance_scale = text_
            image_guidance_scale = image_
            image_out = pipe(
                prompt=text_prompt,
                image=Minecraft_Dataset_source_img,
                num_inference_steps=100,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=text_guidance_scale,
                generator=generator,
                dino_image_features=dino_image_features).images[0]
            image_out.save(os.path.join(save_dir, f'{(pair_id + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
    print('Understanding Scenes Editing image %d' % (pair_id + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. save_dir
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True)
    # 2. pretrain_unet
    parser.add_argument(
        '--pretrain_unet',
        type=str,
        required=True,
    )
    # 3. input_image path
    parser.add_argument(
        '--input_image',
        type=str,
        required=True,
    )
    # 4. text_prompt
    parser.add_argument(
        '--text_prompt',
        type=str,
        required=True,
    )
    # 5. sd_path
    parser.add_argument(
        '--sd_path',
        type=str,
        default='data/weights/sd/stable-diffusion-v1-5',
    )
    # 6. pipeline_path
    parser.add_argument(
        '--pipeline_path',
        type=str,
        default='data/weights/instruct-pix2pix',
    )
    args = parser.parse_args()

    main()

