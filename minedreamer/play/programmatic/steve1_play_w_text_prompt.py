import argparse
import os
import torch
from tqdm import tqdm
from minedreamer.config import DEVICE, DREAMER_IMAGE_SIZE, FPS, PRIOR_INFO
from minedreamer.results.programmatic_eval import ProgrammaticEvaluator
from minedreamer.text_alignment.vae import load_vae_model
from minedreamer.utils.cmd_utils import custom_init_env
from minedreamer.utils.embed_utils import get_prior_embed_from_text_instruction
from minedreamer.utils.file_utils import read_config, save_frame_as_image, save_json
from minedreamer.utils.mineclip_agent_env_utils import load_mineclip_agent_env, setup_env_seed
from minedreamer.utils.text_overlay_utils import create_video_frame_w_prompt_on_frame
from minedreamer.utils.video_utils import save_frames_as_video


def play(args, metadata):
    text_prompt = args.text_prompt

    agent, mineclip, env = load_mineclip_agent_env(args.in_model, args.in_weights, args.env_seed, args.cond_scale, args.inventory, args.preferred_spawn_biome)
    PRIOR_INFO['model_path'] = args.prior_weights_path
    prior = load_vae_model(PRIOR_INFO)

    obs = env.reset()
    env.seed(args.env_seed)
    obs, _, _, _ = custom_init_env(env, args)

    # Setup
    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)
    frame = obs['pov']

    obs_image_save_dir = os.path.join(args.save_video_dir, "obs")
    os.makedirs(obs_image_save_dir, exist_ok=True)

    with torch.cuda.amp.autocast():
        prompt_embed = get_prior_embed_from_text_instruction(text_prompt, mineclip, prior, DEVICE)

    for i in tqdm(range(args.gameplay_length)):

        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)
        # execute action
        obs, _, _, _ = env.step(minerl_action)
        frame = obs['pov']

        if i % 20 == 0:
            save_frame_as_image(frame, obs_image_save_dir, f"{i}.png", target_size=DREAMER_IMAGE_SIZE, to_bgr=True) 

        # frame = cv2.resize(frame, (128, 128))
        frame_w_prompt_on_frame = create_video_frame_w_prompt_on_frame(frame, text_prompt, i, obs['location_stats'])
        gameplay_frames.append(frame_w_prompt_on_frame)

        prog_evaluator.update(obs)

    # Make the eval episode dir and save it
    save_frames_as_video(gameplay_frames, os.path.join(args.save_video_dir, f'{text_prompt}.mp4'), FPS, to_bgr=False)

    # Print the programmatic eval task results at the end of the gameplay
    metadata["results"] = prog_evaluator.get_results()
    prog_evaluator.print_results()


def update_args_with_config(args, config):
    args.in_model = config['model'].get('in_model', 'data/weights/vpt/2x.model')
    args.in_weights = config['model'].get('in_weights', 'data/weights/steve1/steve1.weights')
    args.cond_scale = float(config['conditioning'].get('cond_scale', 6.0))
    args.env_seed = config['environment'].get('env_seed', -1)
    args.gameplay_length = int(config['gameplay'].get('gameplay_length', 3000))
    args.text_prompt = config['prompt']['text_prompt']
    args.preferred_spawn_biome = config['environment'].get('preferred_spawn_biome', None)
    args.inventory = config['environment'].get('inventory', None)
    args.custom_init_commands = config['environment'].get('custom_init_commands', [])
    args.summon_mobs = config['environment'].get('summon_mobs', [])
    return args

def main(args):

    text_prompt = args.text_prompt
    if isinstance(args.env_seed, int):
        args.env_seed = setup_env_seed(args.env_seed)
        for i in range(args.times):
            args.env_seed += 1

            args.save_dirpath = 'data/play/programmatic/steve1/steve1_' + str(text_prompt.replace(' ', '_')) + '_cond_scale_' + str(args.cond_scale) + '_length_' + str(args.gameplay_length) + '_seed_' + str(args.env_seed)
            os.makedirs(args.save_dirpath, exist_ok=True)

            print(f'\nGenerating video for text prompt with name: {text_prompt}')
            args.save_video_dir = os.path.join(args.save_dirpath, text_prompt.replace(' ', '_'))
            metadata = vars(args)

            play(args, metadata)

            metadata_name = text_prompt.replace(' ', '_')
            save_json(metadata, os.path.join(args.save_video_dir, f'{metadata_name}.json'))
    elif isinstance(args.env_seed, list):
        env_seed_list = args.env_seed
        for env_seed in env_seed_list:
            args.env_seed = setup_env_seed(env_seed)

            args.save_dirpath = 'data/play/programmatic/steve1/steve1_' + str(text_prompt.replace(' ', '_')) + '_cond_scale_' + str(args.cond_scale) + '_length_' + str(args.gameplay_length) + '_seed_' + str(args.env_seed)
            os.makedirs(args.save_dirpath, exist_ok=True)

            print(f'\nGenerating video for text prompt with name: {text_prompt}')
            args.save_video_dir = os.path.join(args.save_dirpath, text_prompt.replace(' ', '_'))
            metadata = vars(args)

            play(args, metadata)

            metadata_name = text_prompt.replace(' ', '_')
            save_json(metadata, os.path.join(args.save_video_dir, f'{metadata_name}.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--prior_weights_path', type=str, help='Path to the prior weights path.')
    parser.add_argument('--times', type=int, default=1)

    args = parser.parse_args()

    config = read_config(args.config)
    args = update_args_with_config(args, config)

    args_dict = vars(args)
    print("Args Value:")
    for arg_name in args_dict:
        print(f">>> '{arg_name}': {args_dict[arg_name]}")

    main(args)

    