import argparse
import json
import os
import requests
import torch
from tqdm import tqdm
from minedreamer.config import DEVICE, DREAMER_IMAGE_SIZE, FPS, LOAD_CVAE_KEY_INFO
from minedreamer.results.record import Record
from minedreamer.text_alignment.manager import CVAEManager
from minedreamer.utils.cmd_utils import custom_init_env
from minedreamer.utils.embed_utils import embed_one_frame, get_cvae_embed_from_text_instruction_with_current_and_goal_image_embed
from minedreamer.utils.file_utils import read_config, save_frame_as_image
from minedreamer.utils.mineclip_agent_env_utils import make_dreamer, load_mineclip_agent_env, setup_env_seed
from minedreamer.utils.text_overlay_utils import create_video_frame_w_prompt_on_right_side, create_video_frame_w_prompt_on_frame
from minedreamer.utils.video_utils import save_frames_as_video


block_filter_types = ["dirt", "granite", "redstone", "cobblestone", "andesite", "diamond", "diorite", "iron_ore", "gold_ore", "coal", "lapis_lazuli"]

block_names = ["dirt", "granite", "redstone", "cobblestone", "andesite", "diamond", "diorite", "iron_ore", "gold_ore", "coal", "lapis_lazuli"]

def play_interactive(args, cvae_manager, metadata):

    prompt_id = 0
    prompt_list = ["dig down", "mine horizontally"]
    
    agent, mineclip, env = load_mineclip_agent_env(args.in_model, args.in_weights, args.env_seed, args.cond_scale, args.inventory, args.preferred_spawn_biome)
    goal_dreamer = make_dreamer(args.dreamer_url)
    current_image_save_dir = os.path.join(args.save_video_dir, "current")
    goal_image_save_dir = os.path.join(args.save_video_dir, "goal")
    os.makedirs(args.save_video_dir, exist_ok=True)
    os.makedirs(current_image_save_dir, exist_ok=True)
    os.makedirs(goal_image_save_dir, exist_ok=True)

    is_new_prompt = True
    state = {'obs': None}
    record = Record(block_filter_types, block_names)
    video_frames = []

    def reset_env():
        print('\nResetting environment...')
        env.reset()
        
        if args.env_seed is not None:
            print(f'Setting seed to {args.env_seed}...')
            env.seed(args.env_seed)

        state['obs'], _, _, _ = custom_init_env(env, args, load_long_horizon_cmds=True)
 
    reset_env()
    frame = state['obs']['pov']

    for i in tqdm(range(12000)):

        if is_new_prompt:
            prompt = prompt_list[prompt_id]

            print(f"Change prompt to {prompt}")
            prompt_id += 1
            agent.reset(args.cond_scale)
            text_prompt, mapping_key = cvae_manager.seach_user_prompt(prompt)
            freq = LOAD_CVAE_KEY_INFO[mapping_key]

        if is_new_prompt or (i % freq == 0 and i != 0):
            agent.reset(args.cond_scale)
            is_new_prompt = False
            current_image_embed = embed_one_frame(frame, mineclip, DEVICE, batch_size=1, show_prog=False)
            save_frame_as_image(frame, current_image_save_dir, f"{i}.png", target_size=DREAMER_IMAGE_SIZE, to_bgr=True)
            try:
                goal_image_list = goal_dreamer.generate_goal_image(
                    suffix="goal_image",
                    text_prompt=text_prompt,
                    current_image_path=os.path.join(current_image_save_dir, f"{i}.png"),
                    is_del=1
                )
            except requests.exceptions.JSONDecodeError as e:
                continue

            goal_image_embed = embed_one_frame(goal_image_list[0], mineclip, DEVICE, batch_size=1, show_prog=False)
            save_frame_as_image(goal_image_list[0], goal_image_save_dir, f"{i}.png", target_size=DREAMER_IMAGE_SIZE, to_bgr=False)

            with torch.cuda.amp.autocast():
                prompt_embed = get_cvae_embed_from_text_instruction_with_current_and_goal_image_embed(text_prompt, current_image_embed, goal_image_embed, mineclip, cvae_manager.models[mapping_key], DEVICE)

        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(state['obs'], prompt_embed)

        # execute action
        state['obs'], _, _, _ = env.step(minerl_action)
        frame = state['obs']['pov']

        # Save Current Image
        if i % 20 == 0:
            frame_w_prompt_on_frame = create_video_frame_w_prompt_on_frame(frame, prompt, i, state['obs']['location_stats'])
            save_frame_as_image(frame_w_prompt_on_frame, os.path.join(args.save_video_dir, "obs"), f"{i}.png", target_size=DREAMER_IMAGE_SIZE, to_bgr=False)

        frame_with_prompt = create_video_frame_w_prompt_on_right_side(frame, prompt, i, state['obs']['location_stats'])
        video_frames.append(frame_with_prompt)

        record.update_obs(prompt, state['obs'])

        if state['obs']['location_stats']['ypos'].item() <= 14:
            if prompt_id < len(prompt_list):
                print("Change prompt...")
                is_new_prompt = True

        if not state['obs']['life_stats']['is_alive'].item() \
            or state['obs']['life_stats']['life'].item() < 1 \
            or (prompt_id == 2 and state['obs']['location_stats']['ypos'].item() > 60) \
            or state['obs']['location_stats']['ypos'].item() <= 4:
            print("Agent is Dead! We can try again!")
            break

        if state['obs']['inventory']['diamond'].item() > 0:
            print("Successfully mine diamond ore!") 

            video_name = f"diamond"
            # Save both the video and the prompts for each frame
            output_video_filepath = os.path.join(args.save_video_dir, f'{video_name}.mp4')
            print(f'Saving video to {output_video_filepath}...')
            save_frames_as_video(video_frames, output_video_filepath, fps=FPS)
            break
    else:
        print("Fail to mine diamond ore!")   

    video_name = f"diamond"
    prompts_for_frames_filepath = os.path.join(args.save_video_dir, f'{video_name}.json')
    metadata["record"] = record.record_list
    print(f'Saving prompts for frames to {prompts_for_frames_filepath}...')
    with open(prompts_for_frames_filepath, 'w') as f:
        json.dump(metadata, f) 
    

def update_args_with_config(args, config):
    args.in_model = config['model'].get('in_model', 'data/weights/vpt/2x.model')
    args.in_weights = config['model'].get('in_weights', 'data/weights/steve1/steve1.weights')
    args.cond_scale = float(config['conditioning'].get('cond_scale', 6.0))
    args.env_seed = config['environment'].get('env_seed', -1)
    args.dreamer_url = config['dreamer'].get('dreamer_url', "http://127.0.0.1:8000/")
    args.preferred_spawn_biome = config['environment'].get('preferred_spawn_biome', None)
    args.inventory = config['environment'].get('inventory', None)
    args.custom_init_commands = config['environment'].get('custom_init_commands', [])
    args.summon_mobs = config['environment'].get('summon_mobs', [])
    return args

def main(args):

    args.env_seed = setup_env_seed(args.env_seed)
    os.makedirs(args.save_video_dir, exist_ok=True)

    metadata = vars(args)

    cvae_manager = CVAEManager(prior_weights_dir=args.prior_weights_dir)
    if args.update:
        cvae_manager.del_all_mapping_db()
        cvae_manager.update_mapping_db()

    save_video_dir = args.save_video_dir
    for i in range(1, args.times + 1):
        args.save_video_dir = os.path.join(save_video_dir, f"diamond_{i}")
        play_interactive(args, cvae_manager, metadata)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--prior_weights_dir', type=str, help='Path to the prior model name.')
    parser.add_argument('--save_video_dir', type=str, default='data/play/chaining/diamond')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()

    config = read_config(args.config)
    args = update_args_with_config(args, config)

    args_dict = vars(args)
    print("Args Value:")
    for arg_name in args_dict:
        print(f">>> '{arg_name}': {args_dict[arg_name]}")

    main(args)