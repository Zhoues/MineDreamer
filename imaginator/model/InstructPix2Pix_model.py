# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from typing import List, Optional
from transformers.utils import ModelOutput
from transformers import CLIPTextModel, CLIPTokenizer



class InstructPix2PixOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

# InstructPix2Pix_model
class InstructPix2Pix_model(nn.Module):
    def __init__(self,
                 SD_path,
                 CLIP_path):
        super(InstructPix2Pix_model, self).__init__()
        self.SD_path_ = SD_path
        self.CLIP_path_ = CLIP_path

    # 1. initialize Stable Diffusion
    def init_sd_vae_unet(self, is_position_embeddings, diffusion_loss_weight):
        SD_path = self.SD_path_
        self.vae = AutoencoderKL.from_pretrained(SD_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(SD_path, subfolder="scheduler")
        self.diffusion_loss_weight = diffusion_loss_weight

        # using position embeddings or not
        if is_position_embeddings == True:
            # Diffusers-0.20.2 unet
            # <class 'imaginator.diffusers0202_unet.unet_2d_condition.UNet2DConditionModel'>
            from imaginator.diffusers0202_unet.unet_2d_condition import UNet2DConditionModel
            self.unet = UNet2DConditionModel.from_pretrained(SD_path, subfolder="unet")
        else:
            from diffusers import UNet2DConditionModel
            self.unet = UNet2DConditionModel.from_pretrained(SD_path, subfolder="unet")

    # 2. initialize CLIP text encoder
    def init_CLIP_text_encoder(self):
        CLIP_path = self.CLIP_path_
        self.CLIP_tokenizer = CLIPTokenizer.from_pretrained(CLIP_path)
        self.CLIP_text_encoder = CLIPTextModel.from_pretrained(CLIP_path)


    def forward(
            self,
            original_img: torch.FloatTensor = None,
            edited_img: torch.FloatTensor = None,
            dino_image: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. classifier-free guidance for image and text embeddings
        batch_size = original_img.shape[0]
        random_p = torch.rand(batch_size, device=original_img.device)
        InstructPix2Pix_dropout_prob = 0.05

        # CFG-1. Sample masks for the edit prompts
        text_embeddings = self.CLIP_text_encoder(input_ids)[0]
        # [bs, 77, 768]
        null_text_prompt = ""
        null_text_prompt = self.CLIP_tokenizer(null_text_prompt,
                                               max_length=self.CLIP_tokenizer.model_max_length,
                                               padding="max_length",
                                               truncation=True,
                                               return_tensors="pt")
        null_text_prompt_ids = null_text_prompt.input_ids.to(input_ids.device)
        null_text_embeddings = self.CLIP_text_encoder(null_text_prompt_ids)[0]

        # Final text conditioning
        prompt_mask = random_p < 2 * InstructPix2Pix_dropout_prob
        prompt_mask_embeds = prompt_mask.reshape(batch_size, 1, 1)
        text_embeddings = torch.where(prompt_mask_embeds, null_text_embeddings, text_embeddings)
        text_embeddings = text_embeddings.to(torch.float32)
        # [bs, 77, 768]

        # 2. Diffusion loss: Convert images to latent space
        edited_img = edited_img.to(torch.bfloat16)
        latents = self.vae.encode(edited_img).latent_dist.sample().detach()
        latents = latents * self.vae.config.scaling_factor
        # [bs, 4, 32, 32]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz, ), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(torch.float32)
        # [bs, 4, 32, 32]

        # Get the additional image embedding for conditioning -> Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_img = original_img.to(torch.bfloat16)
        original_image_embeds = self.vae.encode(original_img).latent_dist.mode()
    
        # CFG-2. Sample masks for the original images
        image_mask_dtype = original_img.dtype
        image_mask = 1 - ((random_p >= InstructPix2Pix_dropout_prob).to(image_mask_dtype) * (random_p < 3 * InstructPix2Pix_dropout_prob).to(image_mask_dtype))
        image_mask = image_mask.reshape(batch_size, 1, 1, 1)
        original_image_embeds = image_mask * original_image_embeds
        original_image_embeds = original_image_embeds.to(torch.float32)
        # [bs, 4, 32, 32]

        # Concatenate the original_image_embeds with the noisy_latents
        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
        # [bs, 8, 32, 32]

        # Predict the noise residual
        model_pred = self.unet(sample=concatenated_noisy_latents,
                               timestep=timesteps,
                               encoder_hidden_states=text_embeddings).sample
        # noise image prediction -> [bs, 4, 32, 32]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        SD_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") * self.diffusion_loss_weight

        # Final loss: Diffusion loss
        loss = SD_loss

        return InstructPix2PixOutput(
            loss=loss
        )

