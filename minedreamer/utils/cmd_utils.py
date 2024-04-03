import random
from minedreamer.config import LONG_HORIZON_INIT_GAME_CMDS


def custom_init_env(env, args, load_long_horizon_cmds=False):
    cmds = []
    custom_init_commands = getattr(args, 'custom_init_commands', []) or []
    summon_mobs = getattr(args, 'summon_mobs', []) or []

    if load_long_horizon_cmds:
        cmds += LONG_HORIZON_INIT_GAME_CMDS

    if len(custom_init_commands) > 0:
        cmds += custom_init_commands

    if len(summon_mobs) > 0:
        cmds += create_summon_mobs_cmds(summon_mobs)

    if len(cmds) == 0:
        print("No need executing command")
        return env.step(env.action_space.no_op())

    for cmd in cmds:
        print(f"Executing {cmd} ...")
        env.execute_cmd(cmd)

    print("Finish executing all commands!")
    return [env.step(env.action_space.no_op()) for _ in range(5)][-1]

def create_summon_mobs_cmds(summon_mobs):
    cmds = []
    for mob_conf in summon_mobs:
        mob_name = mob_conf['mob_name']
        range_x = mob_conf['range_x']
        range_z = mob_conf['range_z']
        number = mob_conf['number']
        for _ in range(number):
            cmd = '/execute as @p at @p run summon minecraft:{} ~{} ~ ~{} {{Age:0}}'.format(mob_name, str(random.randint(range_x[0], range_x[1])), str(random.randint(range_z[0], range_z[1])))
            cmds.append(cmd)
    return cmds