import json
import os
import pickle
import random
import re
import yaml
import numpy as np
import cv2
from minedreamer.config import DATASET_DIRS
from minedreamer.utils.video_utils import load_frame_from_video


def read_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_text(filepath, by_lines=False):
    with open(filepath, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


def load_prompt_txt(prompt_dir, prompt_filename):
    return load_text(os.path.join(prompt_dir, f"{prompt_filename}.txt"))


def random_choose_sub_file_path(folder_path, isfile=False):
    if not os.path.isdir(folder_path):
        raise ValueError("The specified folder path does not exist.")

    if isfile:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    else:
        files = [f for f in os.listdir(folder_path)]
    
    if not files:
        raise ValueError("The specified folder contains no files.")

    random_file = random.choice(files)

    return os.path.join(folder_path, random_file)


def match_contractor_version_video_id(contractor_video_dir_name):
    pattern = r"contractor_(?P<version>\d+_x)_(?P<video_id>\d+)"
    match = re.search(pattern, contractor_video_dir_name)
    if match:
        return match.group("version"), match.group("video_id")
    else:
        return None, None
    
def match_meta_actions_version_video_id(meta_actions_dir_name):
    pattern = r"contractor_(?P<version>\d+_x)_(?P<video_id>\d+)_metadata_actions"
    match = re.search(pattern, meta_actions_dir_name)
    if match:
        return match.group("version"), match.group("video_id")
    else:
        return None, None


def save_frame_as_image(frame, savefile_path, savefile_name, target_size=None, to_bgr=True):
    if not os.path.exists(savefile_path):
        os.makedirs(savefile_path)

    save_path = os.path.join(savefile_path, savefile_name)

    _, file_extension = os.path.splitext(savefile_name)
    if not file_extension:
        raise ValueError("No file extension found in savefile_name. Please include an extension like '.jpg' or '.png'.")
    
    if target_size is not None:
        if not (isinstance(target_size, tuple) or isinstance(target_size, list)) or len(target_size) != 2:
            raise ValueError("target_size must be a tuple or list with 2 elements (width, height).")
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    if to_bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    success = cv2.imwrite(save_path, frame)
    if not success:
        raise IOError(f"Could not write image to {save_path}. Check file extension and OpenCV installation.")



def save_image_to_dir(idx, frame_tuple, start_frame_dir, end_frame_dir, target_size):

    assert len(frame_tuple) == 5, "Each element of frame_tuple must contain exactly 5 elements."
    _, _, video_dir_name, start_frame_id, end_frame_id = frame_tuple

    if video_dir_name.startswith("contractor_"):

        version, video_id = match_contractor_version_video_id(video_dir_name)

        if version is not None and video_id is not None:

            start_frame = load_frame_from_video(os.path.join(DATASET_DIRS['contractor_org'], version, f"{video_id}.mp4"), start_frame_id)
            end_frame = load_frame_from_video(os.path.join(DATASET_DIRS['contractor_org'], version, f"{video_id}.mp4"), end_frame_id)

            save_frame_as_image(start_frame, start_frame_dir, f"start_frame_{idx}.png", target_size)
            save_frame_as_image(end_frame, end_frame_dir, f"end_frame_{idx}.png", target_size)