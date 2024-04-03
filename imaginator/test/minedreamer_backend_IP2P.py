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


############# Flask
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import io
import base64
app = Flask(__name__)
############# Flask


parser = argparse.ArgumentParser()
# 1. data_path
parser.add_argument(
    '--data_path',
    type=str,
    default='img'
)
# 2. pretrain_unet
parser.add_argument(
    '--pretrain_unet',
    type=str,
    required=True,
)
# 3. sd_path
parser.add_argument(
    '--sd_path',
    type=str,
    default='data/weights/sd/stable-diffusion-v1-5',
)
# 4. pipeline_path
parser.add_argument(
    '--pipeline_path',
    type=str,
    default='data/weights/instruct-pix2pix',
)
args = parser.parse_args()


os.makedirs(args.data_path, exist_ok=True)

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



seed = random.randint(0, 100000)
generator = torch.Generator("cuda").manual_seed(seed)


# choose
UPLOAD_FOLDER = input_path = args.data_path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
text_guidance_scale_list = [7.5]
# image_guidance_scale_list = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
image_guidance_scale_list = [1.6]

def get_image_base64(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return encoded_img


@app.route('/goal_image', methods=['POST'])
def goal_image():

    if 'current_image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    current_image = request.files['current_image']

    if current_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    unique_id = str(uuid.uuid4())
    current_image_name = secure_filename(unique_id + '.png')
    vision_path = os.path.join(app.config['UPLOAD_FOLDER'], current_image_name)
    current_image.save(vision_path)

    # 1. Get input image and GT image
    input_image = Image.open(vision_path)
    input_image_np = np.array(input_image)
    input_image = cv2.resize(input_image_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        # convert input image into torch style
    input_image = np.transpose(input_image, (2, 0, 1)) # (3, 256, 256)
    

    input_image = torch.tensor(input_image)
    input_image = 2 * (input_image / 255) - 1


    text_prompt = request.form.get('text_prompt')

    goal_image_list = []

    for text_ in text_guidance_scale_list:
        for image_ in image_guidance_scale_list:
            text_guidance_scale = text_
            image_guidance_scale = image_
            image_out = pipe(
                prompt=text_prompt,
                image=input_image,
                num_inference_steps=100,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=text_guidance_scale,
                generator=generator,
                dino_image_features=dino_image_features).images[0]

            goal_image_list.append(get_image_base64(image_out))

    is_del = request.form.get('is_del')
    if is_del is not None and int(is_del) == 1:
        os.remove(vision_path)

    response = jsonify({'result': 1, 'goal_image_list': goal_image_list})

    response.headers.set('Content-Type', 'application/json')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25547)

