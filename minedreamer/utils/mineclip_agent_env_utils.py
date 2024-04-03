import pickle

import numpy as np
import gym
from minedreamer.text_alignment.cvae import load_cvae_model
from minedreamer.dreamer.dreamer import Dreamer
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minedreamer.steve1_agent.MineRLConditionalAgent import MineRLConditionalAgent
from minedreamer.VPT.agent import ENV_KWARGS, MineRLAgent
from minedreamer.config import CVAE_INFO, MINECLIP_CONFIG, DEVICE, OPENCLIP_CONFIG
from minedreamer.mineclip_code.load_mineclip import load

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def load_mineclip_wconfig():
    print('Loading MineClip...')
    return load(MINECLIP_CONFIG, device=DEVICE)

def load_openclip_wconfig():
    print('Loading OpenClip...')
    return OpenCLIPEmbeddings(**OPENCLIP_CONFIG)

def setup_env_seed(env_seed):
    if env_seed == -1:
        env_seed = int(np.random.randint(0, 2 ** 25))
    return env_seed

def make_env(seed, inventory=None, preferred_spawn_biome=None):
    print('Loading MineRL...')
    if inventory is not None:
        print("Initializing Agent's Inventory...")
        for item_dict in inventory:
            print(f"Add {item_dict['quantity']} {item_dict['type']} to inventory...")
    else:
        print("Initializing Agent's Inventory As Empty!")

    if preferred_spawn_biome is not None:
        print("Initializing Agent's Biome...")
        print(f"Move agent to {preferred_spawn_biome}")
    else:
        print("Initializing Agent's Biome Randomly!")

    env = HumanSurvival(**ENV_KWARGS, inventory=inventory, preferred_spawn_biome=preferred_spawn_biome).make()
    print('Starting new env...')
    env.reset()
    if seed is not None:
        print(f'Setting seed to {seed}...')
        env.seed(seed)
    return env

# create org vpt agent
def make_vpt_agent(in_model, in_weights):
    # get vpt model architecture
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    # create agent for specific task (before doing it, we must create minerl env)
    env = gym.make("MineRLBasaltFindCave-v0")
    # create vpt agent
    agent = MineRLAgent(env, device='cuda', policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()
    return agent

def make_agent(in_model, in_weights, cond_scale):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device='cuda', policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent

def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale, inventory=None, preferred_spawn_biome=None):
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed, inventory, preferred_spawn_biome)
    return agent, mineclip, env

def make_dreamer(dreamer_url):
    dreamer = Dreamer(dreamer_url)
    return dreamer
