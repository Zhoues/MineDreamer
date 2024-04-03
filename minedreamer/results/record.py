

class Record:
    def __init__(self, block_filter_types, block_names):
        self.block_filter_types = block_filter_types
        self.block_names = block_names
        self.record_list = []

    def update_max_inventory_counts(self, current_inventory, inventory_counts, block_type, block_key):
        block_names = [x for x in current_inventory.keys() if block_type in x]
        block_count = 0
        for block_name in block_names:
            block_count += int(current_inventory.get(block_name, 0))

        inventory_counts[block_key] = block_count

        return inventory_counts

    def update_obs(self, prompt, obs_current):
        inventory = {}
        for block_name in self.block_names:
            if block_name not in inventory:
                inventory[block_name] = 0

        current_inventory = obs_current['inventory']
        
        for block_filter_type, block_name in zip(self.block_filter_types, self.block_names):
            inventory =  self.update_max_inventory_counts(current_inventory, inventory, block_filter_type, block_name)

        pos = {
            "x": abs(int(obs_current['location_stats']['xpos'])),
            "y": abs(int(obs_current['location_stats']['ypos'])),
            "z": abs(int(obs_current['location_stats']['zpos']))
        }

        self.record_list.append(
            {
                "inventory": inventory,
                "pos": pos,
                "prompt": prompt
            }
        )
