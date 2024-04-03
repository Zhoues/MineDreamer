import argparse
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

from minedreamer.config import MINECLIP_CONFIG
from minedreamer.data.utils.contractor import ContractorData
from minedreamer.data.EpisodeStorage import EpisodeStorage
from minedreamer.mineclip_code.load_mineclip import load
from minedreamer.data.generation.FrameBuffer import QueueFrameBuffer
from minedreamer.utils.embed_utils import embed_videos_mineclip_batched, embed_one_frame_batched


def convert_episode(contractor_data, idx, mineclip, ep_dirpath, batch_size, min_timesteps):
    print(f'Downloading and converting episode {idx}...')
    try:
        frames, frames_mineclip, actions = contractor_data.download(idx)
        if frames is None:
            print(f'Episode {idx} not valid. Skipping...')
            return

        num_timesteps = len(frames)
        if num_timesteps < min_timesteps:
            print(f'Episode has {num_timesteps} timesteps, less than {min_timesteps}. Skipping...')
            return
        
        frame_buffer = QueueFrameBuffer()
        for frame in frames_mineclip:
            frame_buffer.add_frame(frame)

        print(f'Embedding frames...')

        mineclip_embeds = embed_videos_mineclip_batched(frame_buffer, mineclip, 'cuda', batch_size)
        one_frame_mineclip_embeds = embed_one_frame_batched(frames_mineclip, mineclip, 'cuda', batch_size)

        # NOTE: 0~14 frames do not need mineclip embeddings
        mineclip_embeds = [None] * 15 + mineclip_embeds

        # Save Data
        ep = EpisodeStorage(ep_dirpath)
        for frame, action, embed, one_frame_embed in zip(frames, actions, mineclip_embeds, one_frame_mineclip_embeds):
            ep.append(frame, action, embed, one_frame_embed)

        metadata = {
            'contractor_version': contractor_data.version,
            'contractor_index': idx,
        }

        # outdir/VPT n.x + idx
        labeled_episode_dirpath = ep_dirpath
        ep.update_episode_dirpath(labeled_episode_dirpath)
        
        ep.save_episode()
        ep.save_metadata(metadata)
        print(f'Episode saved to {labeled_episode_dirpath}')
    except Exception as e:
        print(f'Error processing episode {idx}: {e}')
        return


def episode_exists(ep_dirpath, existing_episodes):
    ep_dirpath = ep_dirpath.split('/')[-1]
    if ep_dirpath in existing_episodes:
        return True
    else:
        return False

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading MineCLIP...')
    mineclip = load(MINECLIP_CONFIG, device='cuda')

    print(f'Loading contractor data index {args.index}...')

    contractor_data = ContractorData(version=args.index, cache_dir=args.cache_dir, video_downloaded=args.video_downloaded, action_downloaded=args.action_downloaded)
    print(f'Index loaded.')

    print(f'This dataset has {len(contractor_data)} episodes.')

    index_name = args.index.replace('.', '_')

    start_idx = args.worker_id * args.num_episodes
    
    end_idx = start_idx + args.num_episodes

    existing_episodes = os.listdir(args.output_dir)

    for idx in range(start_idx, end_idx):

        print(f'Converting episode {idx}...')

        episode_name = f'contractor_{index_name}_{idx}'

        ep_dirpath = os.path.join(args.output_dir, episode_name)
        if episode_exists(ep_dirpath, existing_episodes):
            print(f'Episode {idx} already exists at {ep_dirpath}. Skipping...')
            continue

        convert_episode(contractor_data, 
                        idx, 
                        mineclip, 
                        ep_dirpath, 
                        args.batch_size,
                        args.min_timesteps)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='10.x', choices=['6.x',
                                                                      '7.x',
                                                                      '8.x',
                                                                      '9.x',
                                                                      '10.x',
                                                                      '11.x'])
    parser.add_argument('--output_dir', type=str, default='data/dataset_contractor/')
    parser.add_argument('--cache_dir', type=str, default='data/contractor_cache')
    parser.add_argument('--batch_size', type=int, default=8)        
    parser.add_argument('--worker_id', type=int, default=0)        
    parser.add_argument('--num_episodes', type=int, default=200)   
    parser.add_argument('--min_timesteps', type=int, default=1000)  
    parser.add_argument('--video_downloaded', type=int, default=0)       
    parser.add_argument('--action_downloaded', type=int, default=0)      

    args = parser.parse_args()
    main(args)