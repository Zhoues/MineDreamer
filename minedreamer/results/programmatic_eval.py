import numpy as np


class ProgrammaticEvaluator:
    """Class for keeping track of: travel distance, seed count, dirt count, and log count."""
    def __init__(self, initial_obs) -> None:
        # Store the max inventory counts for each block type and travel distance (these are lower bound measures)
        # 存储指定 block 收集的最大数量
        self.inventory_dis_values = {}
        self.mine_block_values = {}
        self.craft_item_values = {}
        # 初始位置
        self.initial_obs = initial_obs

    def update(self, obs):
        self.update_inventory_dist(obs)
        self.update_mine_block(obs)
        self.update_craft_item(obs)

    def update_inventory_dist(self, obs):
        """Update the programmatic evaluation metrics."""
        self.inventory_dis_values = compute_programmatic_rewards(self.initial_obs, obs, self.inventory_dis_values)

    def update_mine_block(self, obs):
        """Update the mine block evaluation metrics."""
        self.mine_block_values = compute_mine_block_rewards(obs, self.mine_block_values)
    
    def update_craft_item(self, obs):
        """Update the craft item evaluation metrics."""
        self.craft_item_values = compute_craft_item_rewards(obs, self.craft_item_values)

    def get_results(self):
        return {
            "inventory_dis_values": self.inventory_dis_values,
            "mine_block_values": self.mine_block_values,
            "craft_item_values": self.craft_item_values,
        }

    def print_results(self):
        """Print the results of the programmatic evaluation."""
        print("Programmatic Evaluation Results:")
        for inventory_item in self.inventory_dis_values.keys():
            print(f"{inventory_item}: {self.inventory_dis_values[inventory_item]}")
        print()

        print("Mine Block Evaluation Results:")
        for mine_block_item in self.mine_block_values.keys():
            print(f"{mine_block_item}: {self.mine_block_values[mine_block_item]}")
        print()

        print("Craft Item Evaluation Results:")
        for craft_item_item in self.craft_item_values.keys():
            print(f"{craft_item_item}: {self.craft_item_values[craft_item_item]}")
        print()


# 维护背包中包含 block_type(filter) 字样的 block_key 在 inventory 中的最大数量
def update_max_inventory_counts(current_inventory, inventory_counts, block_type, block_key):
    """ Update the inventory counts for the block type

    Args:
        current_inventory (dict): Dictionary containing the agent's current inventory counts for each block type
        inventory_counts (dict): Dictionary containing the max inventory counts for each block type
        block_type (str): The string filter for the block type to update the inventory count for
        block_key (str): The key for the block type in the inventory dictionary
    """
    block_names = [x for x in current_inventory.keys() if block_type in x]
    block_count = 0
    for block_name in block_names:
        block_count += int(current_inventory.get(block_name, 0))

    # Update the dirt count in inventory_counts
    if block_count > inventory_counts.get(block_key, 0):
        print(f"Updating inventory count for {block_key} from {inventory_counts.get(block_key, 0)} to {block_count}")
        inventory_counts[block_key] = block_count

    return inventory_counts


def update_max_mine_block_counts(current_mine_block_values, mine_block_counts, block_type_list, block_key):

    block_names = []

    for x in current_mine_block_values.keys():
        for block_type in block_type_list:
            if block_type in x:
                block_names.append(x)

    block_count = 0
    for block_name in block_names:
        block_count += int(current_mine_block_values.get(block_name, 0))

    if block_count > mine_block_counts.get(block_key, 0):
        print(f"Updating mine block count for {block_key} from {mine_block_counts.get(block_key, 0)} to {block_count}")
        mine_block_counts[block_key] = block_count

    return mine_block_counts


def update_max_craft_item_counts(current_craft_item_values, craft_item_counts, block_type_list, block_key):

    block_names = []

    for x in current_craft_item_values.keys():
        for block_type in block_type_list:
            if block_type in x:
                block_names.append(x)

    block_count = 0
    for block_name in block_names:
        block_count += int(current_craft_item_values.get(block_name, 0))

    if block_count > craft_item_counts.get(block_key, 0):
        print(f"Updating craft item count for {block_key} from {craft_item_counts.get(block_key, 0)} to {block_count}")
        craft_item_counts[block_key] = block_count

    return craft_item_counts


def compute_programmatic_rewards(obs_init, obs_current, inventory_dis_values):
    """Compute the inventory count across various types of blocks."""
    current_inventory = obs_current['inventory']

    # inventory 中需要 filter 出来的 block
    block_filter_types = ["_log", "dirt", "seed"]
    # 最后统计的 block 名称
    block_names = ["log", "dirt", "seed"]

    # Update the inventory counts for the block types
    for block_name in block_names:
        if block_name not in inventory_dis_values:
            inventory_dis_values[block_name] = 0

    for block_filter_type, block_name in zip(block_filter_types, block_names):
        inventory_dis_values = update_max_inventory_counts(current_inventory, inventory_dis_values, block_filter_type, block_name)

    # update travel distance

    # Keep track of the travel distance. The travel distance is the Euclidean distance from the spawn point to the
    # farthest point the agent reached during the episode on the horizontal (x-z) plane
    curr_x, curr_y, curr_z = obs_current['location_stats']['xpos'], obs_current['location_stats']['ypos'], obs_current['location_stats']['zpos']

    # Compute the Euclidean distance from the spawn point to the current location
    dist = np.sqrt((curr_x - obs_init['location_stats']['xpos']) ** 2 + (curr_z - obs_init['location_stats']['zpos']) ** 2)

    if dist > inventory_dis_values.get("travel_dist", 0):
        inventory_dis_values["travel_dist"] = dist

    height = depth = curr_y - obs_init['location_stats']['ypos']

    if height > inventory_dis_values.get("height", 0):
        inventory_dis_values["height"] = height
    if depth < inventory_dis_values.get("depth", 0):
        inventory_dis_values["depth"] = depth

    return inventory_dis_values


def compute_mine_block_rewards(obs_current, mine_block_values):
    current_mine_block_values = obs_current['mine_block']

    # mine_block 中需要 filter 出来的 block
    block_filter_types = [["_log"], ["dirt", "grass_block"], ["grass"]]
    # 最后统计的 block 名称
    block_names = ["log", "dirt", "grass"]

    # Update the inventory counts for the block types
    for block_name in block_names:
        if block_name not in mine_block_values:
            mine_block_values[block_name] = 0
    
    for block_filter_type_list, block_name in zip(block_filter_types, block_names):
        mine_block_values = update_max_mine_block_counts(current_mine_block_values, mine_block_values, block_filter_type_list, block_name)

    return mine_block_values

def compute_craft_item_rewards(obs_current, craft_item_values):
    current_craft_item_values = obs_current['craft_item']

    # craft_item 中需要 filter 出来的 block
    block_filter_types = [["_planks"]]
    # 最后统计的 block 名称
    block_names = ["planks"]

    # Update the inventory counts for the block types
    for block_name in block_names:
        if block_name not in craft_item_values:
            craft_item_values[block_name] = 0
    
    for block_filter_type_list, block_name in zip(block_filter_types, block_names):
        craft_item_values = update_max_craft_item_counts(current_craft_item_values, craft_item_values, block_filter_type_list, block_name)

    return craft_item_values