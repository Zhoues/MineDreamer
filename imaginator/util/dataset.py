from datasets import load_from_disk
import io
import numpy as np
from PIL import Image
import random
import copy
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
from torchvision import transforms
from imaginator.conversation_v01 import get_conv_template

def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)

# 1. InstructPix2Pix_Dataset
class InstructPix2Pix_Dataset(Dataset):
    '''
    according to InstructPix2Pix, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'edit_prompt'. 'original_image' can be used with the 'edit_prompt' and 'edited_image' denotes the image after applying the 'edit_prompt' on the 'original_image'.
    "original_image" + "edited_image" + "edit_prompt"
    '''
    def __init__(self,
                 InstructPix2PixDataset_path,
                 InstructPix2PixDataset_resolution_for_SD,
                 CLIP_tokenizer):

        # InstructPix2Pix Dataset path
        self.InstructPix2PixDataset_path = load_from_disk(InstructPix2PixDataset_path)
        # 256
        self.InstructPix2PixDataset_resolution_for_SD = InstructPix2PixDataset_resolution_for_SD
        # SD transformation
        self.InstructPix2PixDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                                    transforms.CenterCrop(self.InstructPix2PixDataset_resolution_for_SD)])
        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer

        # Dino-v2 transformation
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.dinov2_resolution = 224
        self.dinov2_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self,):
        return len(self.InstructPix2PixDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        InstructPix2PixDataset_sample = self.InstructPix2PixDataset_path[index]
        # {'original_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E4C0>, 'edited_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E460>, 'edit_prompt': 'make the leaves yellow'}

        # convert into torch style
        instructpix2pix_original_img = InstructPix2PixDataset_sample['original_image']
        instructpix2pix_edited_img = InstructPix2PixDataset_sample['edited_image']
        instructpix2pix_original_img = Image.open(io.BytesIO(instructpix2pix_original_img['bytes'])).convert('RGB')
        instructpix2pix_edited_img = Image.open(io.BytesIO(instructpix2pix_edited_img['bytes'])).convert('RGB')
        dino_image = copy.deepcopy(instructpix2pix_original_img)

        # convert into numpy array first, then to torch tensor
        # 1. Original Image & 2. Edited Image for SD input
        instructpix2pix_original_img = convert_to_np(instructpix2pix_original_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_edited_img = convert_to_np(instructpix2pix_edited_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_SD_input = np.concatenate([instructpix2pix_original_img, instructpix2pix_edited_img])
        instructpix2pix_SD_input = torch.tensor(instructpix2pix_SD_input)
        instructpix2pix_SD_input = 2 * (instructpix2pix_SD_input / 255) - 1
        # instructpix2pix_SD_input = self.InstructPix2PixDataset_transform(instructpix2pix_SD_input)
        instructpix2pix_original_img, instructpix2pix_edited_img = instructpix2pix_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = InstructPix2PixDataset_sample['edit_prompt']
        instructpix2pix_edited_prompt = self.CLIP_tokenizer(edited_prompt,
                                                            max_length=self.CLIP_tokenizer.model_max_length,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")
        input_ids = instructpix2pix_edited_prompt.input_ids[0]

        # 4. Dino-v2 image
        dino_image = dino_image.resize((self.dinov2_resolution, self.dinov2_resolution), resample=Image.Resampling.BICUBIC)
        dino_image = self.dinov2_transform(dino_image)

        # InstructPix2Pix dataloader -> 3 parts -> [bs, 3, 256, 256], [bs, 3, 256, 256], ['make the leaves yellow']
        return {'original_img': instructpix2pix_original_img,
                'edited_img': instructpix2pix_edited_img,
                'input_ids': input_ids,
                'dino_image': dino_image}

# 2. MagicBrush_Dataset
class MagicBrush_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 MagicBrushDataset_path,
                 MagicBrushDataset_resolution_for_SD,
                 CLIP_tokenizer):

        # MagicBrush Dataset path
        self.MagicBrushDataset_path = load_from_disk(MagicBrushDataset_path)
        # 256
        self.MagicBrushDataset_resolution_for_SD = MagicBrushDataset_resolution_for_SD
        # SD transformation
        self.MagicBrushDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                               transforms.CenterCrop(self.MagicBrushDataset_resolution_for_SD)])
        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer

        # Dino-v2 transformation
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.dinov2_resolution = 224
        self.dinov2_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self,):
        return len(self.MagicBrushDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        MagicBrushDataset_sample = self.MagicBrushDataset_path[index]
        # {'source_img': <PIL.Image.Image image mode=RGB size=500x500 at 0x7F327BE01100>, 'target_img': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F327BE010D0>, 'instruction': 'let the asparagus be replaced with sausages'}

        # convert into torch style
        MagicBrushDataset_source_img = MagicBrushDataset_sample['source_img']
        MagicBrushDataset_target_img = MagicBrushDataset_sample['target_img']
        MagicBrushDataset_source_img = Image.open(io.BytesIO(MagicBrushDataset_source_img['bytes'])).convert('RGB')
        MagicBrushDataset_target_img = Image.open(io.BytesIO(MagicBrushDataset_target_img['bytes'])).convert('RGB')
        dino_image = copy.deepcopy(MagicBrushDataset_source_img)

        # 1. Original Image & 2. Edited Image for SD input
        MagicBrushDataset_source_img = convert_to_np(MagicBrushDataset_source_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_target_img = convert_to_np(MagicBrushDataset_target_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_SD_input = np.concatenate([MagicBrushDataset_source_img, MagicBrushDataset_target_img])
        MagicBrushDataset_SD_input = torch.tensor(MagicBrushDataset_SD_input)
        MagicBrushDataset_SD_input = 2 * (MagicBrushDataset_SD_input / 255) - 1
        # MagicBrushDataset_SD_input = self.MagicBrushDataset_transform(MagicBrushDataset_SD_input)
        MagicBrushDataset_source_img, MagicBrushDataset_target_img = MagicBrushDataset_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = MagicBrushDataset_sample['instruction']
        MagicBrushDataset_instruction = self.CLIP_tokenizer(edited_prompt,
                                                            max_length=self.CLIP_tokenizer.model_max_length,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")
        input_ids = MagicBrushDataset_instruction.input_ids[0]

        # 4. Dino-v2 image
        dino_image = dino_image.resize((self.dinov2_resolution, self.dinov2_resolution), resample=Image.Resampling.BICUBIC)
        dino_image = self.dinov2_transform(dino_image)

        # MagicBrushDataset dataloader -> 3 parts -> [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': MagicBrushDataset_source_img,
                'edited_img': MagicBrushDataset_target_img,
                'input_ids': input_ids,
                'dino_image': dino_image}

# 3. Minecraft_Dataset
class Minecraft_Dataset(Dataset):
    '''
    according to Minecraft, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 Minecraft_Dataset_path,
                 Minecraft_Dataset_resolution_for_SD,
                 CLIP_tokenizer):

        # Minecraft Dataset path
        self.Minecraft_Dataset_path = Minecraft_Dataset_path
        js_path = self.Minecraft_Dataset_path + '/index.json'
        self.Minecraft_Data_List = []

        with open(js_path, 'r') as data_file:
            self.Minecraft_Data_List = json.load(data_file)
        #data_list keys: 'pair_id', 'instruction', 'start_image_path', 'end_image_path'
        # 256
        self.Minecraft_Dataset_resolution_for_SD = Minecraft_Dataset_resolution_for_SD
        # SD transformation
        self.Minecraft_Dataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                               transforms.CenterCrop(self.Minecraft_Dataset_resolution_for_SD)])
        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer


    def __len__(self,):
        return len(self.Minecraft_Data_List)

    def __getitem__(self, index):
        # Loading Path...
        Minecraft_Dataset_sample = self.Minecraft_Data_List[index]
        # {'source_img': <PIL.Image.Image image mode=RGB size=500x500 at 0x7F327BE01100>, 'target_img': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F327BE010D0>, 'instruction': 'let the asparagus be replaced with sausages'}

        # convert into torch style
        Minecraft_Dataset_source_img = Image.open(self.Minecraft_Dataset_path + '/' + Minecraft_Dataset_sample['start_image_path'])   
        Minecraft_Dataset_source_img = np.transpose(Minecraft_Dataset_source_img, (2, 0, 1)) # (3, 256, 256)
        Minecraft_Dataset_target_img = Image.open(self.Minecraft_Dataset_path + '/' + Minecraft_Dataset_sample['end_image_path'])   
        # Minecraft_Dataset_target_img = Image.open(self.Minecraft_Dataset_path + '/' + Minecraft_Dataset_sample['end_image_path'].replace('/start/', '/end/'))   
        Minecraft_Dataset_target_img = np.transpose(Minecraft_Dataset_target_img, (2, 0, 1)) # (3, 256, 256)

        # # 1. Original Image & 2. Edited Image for SD input
        Minecraft_Dataset_SD_input = np.concatenate([Minecraft_Dataset_source_img, Minecraft_Dataset_target_img])
        Minecraft_Dataset_SD_input = torch.tensor(Minecraft_Dataset_SD_input)
        Minecraft_Dataset_SD_input = 2 * (Minecraft_Dataset_SD_input / 255) - 1
        # Minecraft_Dataset_SD_input = self.Minecraft_Dataset_transform(Minecraft_Dataset_SD_input)
        Minecraft_Dataset_source_img, Minecraft_Dataset_target_img = Minecraft_Dataset_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = Minecraft_Dataset_sample['instruction']
        Minecraft_Dataset_instruction = self.CLIP_tokenizer(edited_prompt,
                                                            max_length=self.CLIP_tokenizer.model_max_length,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")
        input_ids = Minecraft_Dataset_instruction.input_ids[0]

        # Minecraft_Dataset dataloader -> 3 parts -> [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']

        return {'original_img': Minecraft_Dataset_source_img,
                'edited_img': Minecraft_Dataset_target_img,
                'input_ids': input_ids}



# 4. MLLM_Minecraft_Dataset
from transformers.trainer_pt_utils import LabelSmoother
DEFAULT_IMAGE_TOKEN = '<image>'
IGNORE_INDEX = -100
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
class MLLM_Minecraft_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 Minecraft_Dataset_path,
                 Minecraft_Dataset_resolution_ViT,
                 Minecraft_Dataset_resolution_for_SD,
                 CLIPImageProcessor,
                 mm_projection_length,
                 editing_template,
                 editing_max_length,
                 llm_tokenizer=None
                 ):

        self.Minecraft_Dataset_path = Minecraft_Dataset_path
        js_path = self.Minecraft_Dataset_path + '/index.json'
        self.Minecraft_Data_List = []

        with open(js_path, 'r') as data_file:
            self.Minecraft_Data_List = json.load(data_file)

        # 224, 256
        self.Minecraft_Dataset_resolution_ViT = Minecraft_Dataset_resolution_ViT
        self.Minecraft_Dataset_resolution_for_SD = Minecraft_Dataset_resolution_for_SD

        # CLIPImageProcessor
        self.CLIPImageProcessor = CLIPImageProcessor

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'

        # Vicuna conversation system for editing
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length
        self.mm_projection_length = mm_projection_length

    def __len__(self,):
        return len(self.Minecraft_Data_List)

    def __getitem__(self, index):

        Minecraft_Dataset_sample = self.Minecraft_Data_List[index]

        # convert into torch style
        original_image = Image.open(self.Minecraft_Dataset_path + '/' + Minecraft_Dataset_sample['start_image_path']).convert('RGB')
        target_image = Image.open(self.Minecraft_Dataset_path + '/' + Minecraft_Dataset_sample['end_image_path']).convert('RGB')   

        instruction = Minecraft_Dataset_sample['instruction']

        # 1. Original Image for ViT input
        RE_original_image = copy.deepcopy(original_image)
        RE_original_image = RE_original_image.resize((self.Minecraft_Dataset_resolution_ViT, self.Minecraft_Dataset_resolution_ViT),
                                                     resample=Image.Resampling.BICUBIC)
        RE_original_image = self.CLIPImageProcessor.preprocess(RE_original_image, return_tensors='pt')['pixel_values']
        RE_original_image = RE_original_image[0]

        # 2. Original Image & 3. Edited Image for SD input
        RE_original_image_2 = convert_to_np(original_image, self.Minecraft_Dataset_resolution_for_SD)
        RE_target_image = convert_to_np(target_image, self.Minecraft_Dataset_resolution_for_SD)
        RE_SD_input = np.concatenate([RE_original_image_2, RE_target_image])
        RE_SD_input = torch.tensor(RE_SD_input)
        RE_SD_input = 2 * (RE_SD_input / 255) - 1
        RE_original_image_2, RE_target_image = RE_SD_input.chunk(2)

        # Vicuna conversation system construction for image editing task...
        # Step 1. Choose Human-GPT templates
        conversation_templates = []
        with open(self.editing_template, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('Human: '):
                    d = dict()
                    d['Human'] = line[len("Human: "):]
                    conversation_templates.append(d)
                elif line.startswith('GPT: '):
                    conversation_templates[-1]['GPT'] = line[len("GPT: "):]

        # Step 2. Choose Vicuna_v1.3 system message
        conv = get_conv_template("vicuna_v1.3")
        roles = {"Human": conv.roles[0], "GPT": conv.roles[1]}

        # <img_i> tokens -> num_new_tokens=35: "<img>"(system message) + " <img_0> ... <img_31>"
        num_new_tokens = len(self.llm_tokenizer) - self.llm_tokenizer.vocab_size
        append_str = ""
        for i in range(num_new_tokens - 3):
            append_str += f" <img_{i}>"

        # Step 3. Vicuna conversation system construction
        edited_prompt = instruction
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'
        edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
        conversation_template = random.choice(conversation_templates)
        conv.messages = []
        conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
        conv.append_message(roles["GPT"], conversation_template["GPT"].replace(' [img].', append_str))
        conversation = conv.get_prompt()
        conversation = conversation.replace("\n", "")
        """
        * A chat between a curious user and an artificial intelligence assistant. 
        The assistant can generate <img>. 
        The assistant needs to conduct the image to image translation task based on given image and text information. 
        The assistant gives helpful, detailed, and polite answers to the user's questions. 
        * USER: I need an image that represents the prior image and caption "<im_start> <img_0> <im_end> turn them into woodcuts". Can you create it?
        * ASSISTANT: Certainly, I have created the image that represents the prior image and caption <img_0> ... <img_31></s>
        """

        # 4. Edited Prompt input_ids -> Tokenize conversations
        input_ids_max_len = self.editing_max_length - self.mm_projection_length
        input_ids = self.llm_tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            max_length=input_ids_max_len,
            truncation=True,
        ).input_ids[0]
        # [(editing_max_length-mm_projection_length)=256]

        # Step 4. Only show up tokens after 'ASSISTANT:'
        # IGNORE_TOKEN_ID=-100
        # retain 'All right, the image that demonstrates the concept is <img_0> ... <img_31></s>'，mask others
        # self.llm_tokenizer('All right, the image that demonstrates the concept is <img_0> ... <img_31></s>').input_ids
        generated_caption_targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        generated_caption_targets[:1] = IGNORE_TOKEN_ID
        total_padding_len = int(generated_caption_targets.ne(self.llm_tokenizer.pad_token_id).sum())

        parts = conversation.split(sep)
        parts[0] += sep
        """
        parts[0] -> LM loss for token after 'ASSISTANT:'
        * A chat between a curious user and an artificial intelligence assistant. 
        The assistant can generate <img>. 
        The assistant needs to conduct the image to image translation task based on given image and text information. 
        The assistant gives helpful, detailed, and polite answers to the user's questions. 
        * USER: I need an image that represents the prior image and caption "<im_start> <img_0> <im_end> turn them into woodcuts". Can you create it?
        * ASSISTANT: 
        """

        # 5. Generated caption targets for Language Model loss
        instruction_len = len(
            self.llm_tokenizer(
                parts[0],
                max_length=input_ids_max_len,
                truncation=True,
            ).input_ids) - 2
        generated_caption_targets[1:(1 + instruction_len)] = IGNORE_TOKEN_ID
        generated_caption_targets[total_padding_len:] = IGNORE_TOKEN_ID
        # [(editing_max_length-mm_projection_length)=256]

        # 6. Edited Prompt attention_mask
        RE_instruction_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # Reasoning-Editing dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': RE_original_image,
                'original_img_for_vae': RE_original_image_2,
                'edited_img': RE_target_image,
                'input_ids': input_ids,
                'input_attention_mask': RE_instruction_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
    

from dataclasses import dataclass, field
import transformers
from typing import Dict, Sequence
@dataclass
class DataCollatorForLLaVADataset(object):
    """ Collate examples for supervised fine-tuning. """

    LLM_tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # LLaVA: len(instances)=batch_size & instances[i].keys() -> dict_keys(['input_ids', 'labels', 'image'])
        original_img = [instance['original_img'] for instance in instances]
        original_img = torch.stack(original_img)
        original_img_for_vae = [instance['original_img_for_vae'] for instance in instances]
        original_img_for_vae = torch.stack(original_img_for_vae)
        edited_img = [instance['edited_img'] for instance in instances]
        edited_img = torch.stack(edited_img)
        is_editing_task = [instance['is_editing_task'] for instance in instances]
        is_editing_task = torch.stack(is_editing_task)

        # for LLaVA processing
        # 1. LLM tokenizer
        input_ids = tuple([instance['input_ids'] for instance in instances])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.LLM_tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.LLM_tokenizer.model_max_length]
        input_attention_mask = [input_id.ne(self.LLM_tokenizer.pad_token_id) for input_id in input_ids]
        input_attention_mask = torch.stack(input_attention_mask)
        input_attention_mask = input_attention_mask[:, :self.LLM_tokenizer.model_max_length]
        generated_caption_targets = tuple([instance['generated_caption_targets'] for instance in instances])
        generated_caption_targets = torch.nn.utils.rnn.pad_sequence(generated_caption_targets, batch_first=True, padding_value=IGNORE_INDEX)
        generated_caption_targets = generated_caption_targets[:, :self.LLM_tokenizer.model_max_length]
        generated_caption_encoder_attention_mask = tuple([instance['generated_caption_encoder_attention_mask'] for instance in instances])
        generated_caption_encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(generated_caption_encoder_attention_mask, batch_first=True, padding_value=False)
        generated_caption_encoder_attention_mask = generated_caption_encoder_attention_mask[:, :self.LLM_tokenizer.model_max_length]

        return {'original_img': original_img,
                'original_img_for_vae': original_img_for_vae,
                'edited_img': edited_img,
                'input_ids': input_ids,
                'input_attention_mask': input_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}


def get_prompt_embedding_LLM(img_embeddings, text_prompt, conversation_template, LLM_tokenizer, model_, temperature, condition_prompt='positive'):
    # Step 2. Choose Vicuna_v1.3 system message
    conv = get_conv_template("vicuna_v1.3")
    roles = {"Human": conv.roles[0], "GPT": conv.roles[1]}

    # Step 3. Vicuna conversation system construction
    edited_prompt = text_prompt
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
    conv.messages = []
    conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
    conv.append_message(roles["GPT"], None)
    conversation = conv.get_prompt()
    conversation = conversation.replace("\n", "")
    """
    'A chat between a curious user and an artificial intelligence assistant.
    The assistant can generate <img>.
    The assistant needs to conduct the image to image translation task based on given image and text information.
    The assistant gives helpful, detailed, and polite answers to the user\'s questions.
    USER: I\'m looking for an image that captures the former image and concept "<im_start> <img_0> <im_end> Apply face paint". Can you make it? ASSISTANT:'
    """

    # Step 4. Save conversation into discrete index and list
    text_prompt_input_ids = LLM_tokenizer(conversation).input_ids
    text_prompt_input_ids = torch.as_tensor(text_prompt_input_ids, device="cuda")
    output_text_ids = text_prompt_input_ids.tolist()

    # decode loop
    llm_img_token_states = []
    token_id = None
    max_new_tokens = 512
    for i in range(max_new_tokens):
        if i == 0:
            if condition_prompt == 'positive':
                images_llm_input = img_embeddings
                # [1, mm_projection_length=256, LLM_hidden_size=4096]

                ####################################################################################
                # 2.3. Remove the first placeholder "<img_0>"
                original_input_ids = text_prompt_input_ids
                LLM_img_start_token_id = LLM_tokenizer.img_start_token_id
                LLM_img_start_token_id_pos = (torch.where(text_prompt_input_ids == LLM_img_start_token_id)[0])[0].item()
                new_input_ids = torch.cat([original_input_ids[:LLM_img_start_token_id_pos], original_input_ids[(LLM_img_start_token_id_pos + 1):]], dim=0)

                # 2.4. prepare input embedding rather than input_ids
                inputs_embeds = model_.model.get_input_embeddings()(new_input_ids.unsqueeze(0))
                LLM_embedding_BeforeStart = inputs_embeds[0][:LLM_img_start_token_id_pos]
                insert_SPE = images_llm_input[0]
                LLM_embedding_AfterStart = inputs_embeds[0][LLM_img_start_token_id_pos:]
                inputs_embeds = torch.cat([LLM_embedding_BeforeStart.unsqueeze(0), insert_SPE.unsqueeze(0), LLM_embedding_AfterStart.unsqueeze(0)], dim=1)
                # [Before Padding, Insert Subject Prompt Embedding, After Padding] -> [1, editing_max_length=512, LLM_hidden_size=4096]

                # 2.4. next token prediction
                out = model_.inference_llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True)
                # out.keys() -> odict_keys(['logits', 'last_hidden_state', 'past_key_values', 'hidden_states', 'attentions'])

            elif condition_prompt == 'negative':
                inputs_embeds = model_.model.get_input_embeddings()(text_prompt_input_ids.unsqueeze(0))
                out = model_.inference_llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
            # [1, (ViT_qformer_query_length + len(input_ids)), LLM_new_vocab_size=32008], len(past_key_values)=32
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = model_.inference_llm(
                input_ids=torch.as_tensor([[token_id]], device="cuda"),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            # [1, 1, LLM_new_vocab_size=32008], len(past_key_values)=32 -> attention_mask从[1, (ViT_qformer_query_length + len(input_ids) + 1)]逐渐增长

        if token_id is not None and token_id >= LLM_tokenizer.img_start_token_id:
            print('Saving LLM embeddings...', token_id)
            llm_img_token_states.append(out.last_hidden_state)

        # mapping to vocabulary
        # temperature = args.temperature
        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token_id = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1))
            # [LLM_new_vocab_size=32008], [LLM_new_vocab_size=32008], e.g.token=13

        output_text_ids.append(token_id)

        if token_id == LLM_tokenizer.eos_token_id or len(llm_img_token_states) == model_.config.num_new_tokens:
            break

    print(output_text_ids)
    print(LLM_tokenizer.decode(output_text_ids))
    llm_img_token_states = torch.cat(llm_img_token_states, dim=1)
    num_new_tokens = 32
    assert llm_img_token_states.shape[1] == num_new_tokens
    print('llm_img_token_states:', llm_img_token_states, llm_img_token_states.shape)
    # [1, query_length(ViT q-former query)=8, LLM_hidden_size=4096]

    return model_.inference_sd_qformer(llm_img_token_states)


# LLaVA inference
@torch.inference_mode()
def generate_LLaVA_image(CLIP_image_features_llm_input, text_prompt, LLM_tokenizer, model_, temperature, editing_template):
    # Step 1. Choose Human-GPT templates
    conversation_templates = []
    with open(editing_template, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Human: '):
                d = dict()
                d['Human'] = line[len("Human: "):]
                conversation_templates.append(d)
            elif line.startswith('GPT: '):
                conversation_templates[-1]['GPT'] = line[len("GPT: "):]
    conversation_template = random.choice(conversation_templates)

    # image: 001.png -> [1, 3, 224, 224], text_prompt: 'Mark the pixels of the cat in the mirror to blue and leave the rest unchanged.'
    # 1. image embeddings + text embeddings for edited prompt
    original_image_embeddings = CLIP_image_features_llm_input
    edited_prompt = text_prompt
    both_condition_embeddings = get_prompt_embedding_LLM(img_embeddings=original_image_embeddings,
                                                         text_prompt=edited_prompt,
                                                         conversation_template=conversation_template,
                                                         LLM_tokenizer=LLM_tokenizer,
                                                         model_=model_,
                                                         temperature = temperature,
                                                         condition_prompt='positive')
    both_condition_embeddings = both_condition_embeddings.to(torch.float16)
    print('both conditional prompt_embedding:', both_condition_embeddings, both_condition_embeddings.shape)
    # [1, query_length(SD q-former query)=CLIP_model_max_length=77, SD_qformer_hidden_size=CLIP_test_dim=768]
    return both_condition_embeddings