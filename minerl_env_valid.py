import argparse
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minedreamer.utils.cmd_utils import custom_init_env
from minedreamer.utils.file_utils import save_frame_as_image

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.summon_mobs = [
    {
        "mob_name": "sheep",
        "range_x": [0, 1],
        "range_z": [5, 15],
        "number": 5
    }
]

env = HumanSurvival(inventory=[{"type": "diamond_axe", "quantity": 1}], preferred_spawn_biome="plains").make()
env.reset()

obs, _, _, _ = custom_init_env(env, args)

save_frame_as_image(obs['pov'], ".", "env_valid.png")