import torch
import cv2

# DEMO FRAME CONFIG
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 2
TEXT_POSITION = (10, 20)
FONT_SCALE = 0.7
LIGHT_ORANGE_COLOR = (0, 165, 255)
DARK_ORANGE_COLOR = (0, 100, 255)

MINECLIP_CONFIG = {
    'arch': "vit_base_p16_fz.v2.t2",
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': "attn.d2.nh8.glusw",
    'resolution': [160, 256],
    'ckpt': {
        'path': "data/weights/mineclip/attn.pth",
        'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
    }
}

OPENCLIP_CONFIG = {
    "model_name": "ViT-g-14", 
    "checkpoint": "laion2b_s34b_b88k"
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRIOR_INFO = {
    'mineclip_dim': 512,
    'latent_dim': 512,
    'hidden_dim': 512,
    'model_path': 'data/weights/steve1/steve1_prior.pt',
}

FPS = 20

DREAMER_IMAGE_SIZE = (256, 256)

DATASET_DIRS = {
    'contractor': 'data/dataset_contractor',
    'contractor_org': 'data/vpt_video',
    'mixed_agents': 'data/dataset_mixed_agents'
}

LONG_HORIZON_INIT_GAME_CMDS = [
    "/difficulty peaceful",
    "/gamerule doDaylightCycle false",
    "/gamerule keepInventory true"
]

CVAE_INFO = {
    'visual_goal_dim': 512,
    'text_dim': 512,
    'current_img_dim': 512,
    'goal_img_dim': 512,
    'hidden_dim': 512,
    'latent_dim': 512,
    'model_path': '',
}

LOAD_CVAE_KEY_INFO = {
    "craft_item_planks": 25,
    "easy_action_dig": 25,
    "easy_action_explore": 25,
    "easy_action_sky": 25,
    "easy_action_swim": 250,
    "easy_action_underwater": 250,
    "easy_action_build_a_tower": 25,
    "easy_action_mine_horizontally": 25,
    "mine_block_dirt": 25,
    "mine_block_grass": 25,
    "mine_block_wood": 25, 
}