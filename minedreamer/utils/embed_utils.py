import torch
import numpy as np
from tqdm import tqdm
from minedreamer.data.EpisodeStorage import EpisodeStorage
from minedreamer.data.generation.FrameBuffer import QueueFrameBuffer
from minedreamer.data.utils.contractor import process_frame_mineclip
from minedreamer.utils.video_utils import load_video_to_lst

@torch.no_grad()
def get_prior_embed_from_text_instruction(text, mineclip, prior, device):
    """Get the embed from text instruction processed by the mineclip and prior."""
    with torch.cuda.amp.autocast():
        text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_prompt_embed = prior(torch.tensor(text_embed).float().to(device)).cpu().detach().numpy()
    return text_prompt_embed

@torch.no_grad()
def get_cvae_embed_from_text_instruction_with_current_and_goal_image_embed(text, current_image_embed, goal_image_embed, mineclip, cvae, device):
    """Get the embed from text instruction processed by the mineclip and prior."""
    with torch.cuda.amp.autocast():
        text_embed = mineclip.encode_text(text).detach().cpu().numpy()

    with torch.no_grad(), torch.cuda.amp.autocast():
        combined_embed = np.concatenate([text_embed, current_image_embed, goal_image_embed], axis=1)
        text_prompt_embed = cvae.generate(torch.tensor(combined_embed).float().to(device)).cpu().detach().numpy()
    return text_prompt_embed

@torch.no_grad()
def get_text_embed_from_text_instruction(text, mineclip):
    """Get the embed from text instruction processed by the mineclip."""
    with torch.cuda.amp.autocast():
        text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    return text_embed

@torch.no_grad()
def get_prior_embed_from_text_embed(text_embed, prior, device):
    """Get the embed from text embedding processed by the prior."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_prompt_embed = prior(torch.tensor(text_embed).float().to(device)).cpu().detach().numpy()
    return text_prompt_embed

@torch.no_grad()
def get_visual_embed_from_video(video_filepath, mineclip, start_frame_id=0):
    end_frame_id = start_frame_id + 16
    video_frames = load_video_to_lst(filepath=video_filepath, to_rgb=True, 
                               only_range=[start_frame_id, end_frame_id], 
                               length=end_frame_id)

    frame_buffer = QueueFrameBuffer()

    for frame_id in range(start_frame_id, end_frame_id):
        frame_mineclip = process_frame_mineclip(video_frames[frame_id])
        frame_buffer.add_frame(frame_mineclip)

    video_embed = embed_videos_mineclip_batched(frame_buffer, mineclip, 'cuda', 1)[0]

    assert video_embed.shape == (1, 512)

    return video_embed

@torch.no_grad()
def get_visual_embed_from_episode(episode_dirpath, timestep):
    """Get the visual embed at the given timestep from the given episode in the dataset. Episode must have been
    saved with EpisodeStorage format (this is how the dataset generation code saves episodes).
    """
    episode = EpisodeStorage(episode_dirpath)
    visual_embeds = episode.load_embeds_attn()
    visual_embed = visual_embeds[timestep]
    return visual_embed

@torch.no_grad()
def embed_one_frame_openclip(frame_urls: list, openclip):
    return openclip.embed_image(frame_urls)

@torch.no_grad()
def embed_one_frame(frame, mineclip, device, batch_size=1, show_prog=True):
    frame_mineclip = process_frame_mineclip(frame)
    frame_buffer = QueueFrameBuffer()

    for _ in range(16): 
        frame_buffer.add_frame(frame_mineclip)

    return embed_videos_mineclip_batched(frame_buffer, mineclip, device, batch_size, show_prog)[0]


@torch.no_grad()
def embed_one_frame_batched(frames: list, mineclip, device, batch_size=32):
    """For each frame in frames_mineclip, replicate it 16 times, process it,
    and get the embedding from mineclip in batches of specified size."""
    
    stack_frames, video_embeds_all = [], []

    for frame in frames:
        stack_frame = [frame for _ in range(16)]
        stack_frames.append(torch.Tensor(np.array(stack_frame)).unsqueeze(0))

    stack_frames_len = len(stack_frames)

    # Process frames in batches
    for batch_start in tqdm(range(0, stack_frames_len, batch_size), desc='Processing one frames in batches'):

        # bsz * (1, 16, 3, 160, 256)
        batch_stack_frames = stack_frames[batch_start: batch_start+batch_size]
        # Compute embeddings in batched form
        video_batch = torch.cat(batch_stack_frames).to(device)
        bsz = video_batch.shape[0]
        # Autocast so that we can use fp16
        with torch.cuda.amp.autocast():
            video_embeds = mineclip.encode_video(video_batch)
        video_embeds = video_embeds.detach().cpu().numpy()

        assert video_embeds.shape == (bsz, 512)  # batch of 512-vectors

        # Add to list (each embed is its own element)
        for video_embed in video_embeds:
            video_embed = video_embed.reshape(1, 512)
            assert video_embed.shape == (1, 512)

            video_embeds_all.append(video_embed)
    return video_embeds_all



@torch.no_grad()
def embed_videos_mineclip_batched(frame_buffer: QueueFrameBuffer, mineclip, device, batch_size=32, show_prog=True):
    """Compute mineclip_code video embedding for an entire QueueFrameBuffer. Returns a listr of 512 vectors
    with shape (1, 512).
    """
    # print(f'Embedding {len(frame_buffer)} frames with batch size {batch_size}...')

    frame_iter = iter(frame_buffer)
    if show_prog:
        prog = tqdm(total=len(frame_buffer))
    video_embeds_all = []
    done = False

    while not done:

        # Get batch of videos
        videos = []
        for _ in range(batch_size):
            try:
                frame = next(frame_iter)
                if show_prog:
                    prog.update(1)
            except StopIteration:
                done = True
                break
            videos.append(frame)
        if len(videos) == 0:
            break

        # Compute embeddings in batched form
        video_batch = torch.cat(videos).to(device)
        bsz = video_batch.shape[0]
        # Autocast so that we can use fp16
        with torch.cuda.amp.autocast():
            video_embeds = mineclip.encode_video(video_batch)
        video_embeds = video_embeds.detach().cpu().numpy()

        assert video_embeds.shape == (bsz, 512)  # batch of 512-vectors

        # Add to list (each embed is its own element)
        for video_embed in video_embeds:
            video_embed = video_embed.reshape(1, 512)
            assert video_embed.shape == (1, 512)

            video_embeds_all.append(video_embed)
    return video_embeds_all