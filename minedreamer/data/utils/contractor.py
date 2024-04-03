import requests
import os
import jsonlines
import shutil
import cv2
import numpy as np

from minedreamer.VPT.agent import resize_image, AGENT_RESOLUTION

# 鼠标image
CURSOR_FILE = 'minedreamer/data/generation/assets/mouse_cursor_white_16x16.png'


# 归一化之后的鼠标image
cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
cursor_image = cursor_image[:16, :16, :]
cursor_alpha = cursor_image[:, :, 3:] / 255.0
cursor_image = cursor_image[:, :, :3]

# 键盘映射
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

CAMERA_SCALER = 360.0 / 2400.0

MINEREC_ORIGINAL_HEIGHT_PX = 720


# MineCLIP 需要 (160x256)的(H, W) 和 (C, H, W) 的图片
def process_frame_mineclip(frame: np.ndarray, height: int = 160, width: int = 256):
    """Processes frame to format that mineclip expects (160x256) and (C, H, W)."""
    assert frame.shape[2] == 3, f'Expected channel dim to be at axis 2, got shape {frame.shape}'

    if frame.shape != (160, 256, 3):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.moveaxis(frame, -1, 0)

# metadata action prefix
EVENT_PREFIXES = {
    "minecraft.mine_block:minecraft.": "mine_block",
    "minecraft.craft_item:minecraft.": "craft_item",
    "minecraft.use_item:minecraft.": "use_item",
    "minecraft.kill_entity:minecraft.": "kill_entity",
    "minecraft.custom:minecraft.interact_with_": "interact_with"
}

# VPT Version JSON 路径
index_files = {
    '6.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json',
    '7.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json',
    '8.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json',
    '9.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json',
    '10.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json',
    '11.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/find-cave-Jul-28.json'
}


# 得到 VPT Index JSON (baseurl + url.mp4/jsonl)
def get_index(version):
    index = requests.get(index_files[version]).json()
    return index


class ContractorData:

    def __init__(self, version, cache_dir, video_downloaded=0, action_downloaded=0):
        # 获取到 baseurl + index(.mp4/.jsonl)
        self.index = get_index(version)
        # VPT Version
        self.version = version
        # 缓存路径
        self.cache_dir = cache_dir

        self.video_downloaded = video_downloaded != 0
        self.action_downloaded = action_downloaded != 0

    @property
    # 获取 baseurl
    def basedir(self):
        return self.index['basedir'][:-1]

    # 根据 baseurl 获得 video url(.mp4)
    def get_video_url(self, idx):
        relpath = self.index['relpaths'][idx]
        relpath = relpath.replace('.mp4', '')
        path = f'{self.basedir}/{relpath}.mp4'
        return path

    # 根据 baseurl 获得 action url(.jsonl)
    def get_action_url(self, idx):
        relpath = self.index['relpaths'][idx]
        relpath = relpath.replace('.mp4', '')
        path = f'{self.basedir}/{relpath}.jsonl'
        return path
    
    def get_video_downloaded_video_path(self, idx):
        relpath = "data/vpt_video"
        version = self.version.replace('.', '_')
        path = f'{relpath}/{version}/{idx}.mp4'
        return path
    
    def get_action_downloaded_action_path(self, idx):
        relpath = "data/vpt_action"
        version = self.version.replace('.', '_')
        path = f'{relpath}/{version}/{idx}.jsonl'
        return path

    # 获得 idx 对应的 frame, mineclip 对应的 frame 和 action
    def download(self, idx):
        """Returns the location of the locally video_downloaded video and also the
        action object."""
        # get video id url
        if self.video_downloaded:
            video_url = self.get_video_downloaded_video_path(idx)
        else:
            video_url = self.get_video_url(idx)

        # get action id url
        # get video id url
        if self.action_downloaded:
            action_url = self.get_action_downloaded_action_path(idx)
        else:
            action_url = self.get_action_url(idx)
        # cache_dir(idx) += VPT Version and idx
        cache_dir = os.path.join(self.cache_dir, f'{self.version}_{idx}')

        # Download video
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        try:
            # Download certain idx video to cache dir idx
            video_path = os.path.join(cache_dir, 'video.mp4')

            if self.video_downloaded:
                if not os.path.exists(video_path):
                    print(f'Copying {video_url} to {video_path}...')
                    shutil.copy(video_url, video_path)
            else:
                if not os.path.exists(video_path):
                    print(f'Downloading {video_url} to {video_path}...')
                    r = requests.get(video_url)
                    with open(video_path, 'wb') as f:
                        f.write(r.content)

            # Download action
            action_path = os.path.join(cache_dir, 'action.jsonl')
            if self.action_downloaded:
                if not os.path.exists(action_path):
                    print(f'Copying {action_url} to {action_path}...')
                    shutil.copy(action_url, action_path)
            else:
                if not os.path.exists(action_path):
                    print(f'Downloading {action_url} to {action_path}...')
                    r = requests.get(action_url)
                    with open(action_path, 'wb') as f:
                        f.write(r.content)

            # Open the action with jsonlines, json_data -> [ Dict(key&mouse), ...]
            with jsonlines.open(action_path) as reader:
                json_data = [action for action in reader]

            print(f'Converting data to frames and actions...')

            # 获得每一帧image，每一帧的mineclip需求的格式image，这一帧对应的action
            frames, frames_mineclip, actions = load_episode(video_path, json_data)

            # 删除 cache dir idx
            self.clean_cache(cache_dir)
        except Exception as e:
            print(f'Failed to download {video_url} or {action_url}: {e}')
            self.clean_cache(cache_dir)
            return None, None, None

        return frames, frames_mineclip, actions

    def process_action_meta(self, idx):
        # get action id url
        if self.action_downloaded:
            action_url = self.get_action_downloaded_action_path(idx)
        else:
            action_url = self.get_action_url(idx)
        # cache_dir(idx) += VPT Version and idx
        cache_dir = os.path.join(self.cache_dir, f'{self.version}_{idx}')

        # Download video
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        try:
            # Download action
            action_path = os.path.join(cache_dir, 'action.jsonl')
            if self.action_downloaded:
                if not os.path.exists(action_path):
                    print(f'Copying {action_url} to {action_path}...')
                    shutil.copy(action_url, action_path)
            else:
                if not os.path.exists(action_path):
                    print(f'Downloading {action_url} to {action_path}...')
                    r = requests.get(action_url)
                    with open(action_path, 'wb') as f:
                        f.write(r.content)

            with jsonlines.open(action_path) as reader:
                json_data = [action for action in reader]

            print(f'Processing metadata actions...')

            meta_actions = {
                'mine_block': {},
                'craft_item': {},
                'use_item': {},
                'kill_entity':{},
                'interact_with': {}
            }

            # 上一步的状态，用于对比是否发生变化
            previous_stats = {}
            # 读取文件
            for i, json_data_idx in enumerate(range(len(json_data))):
                entry = json_data[json_data_idx]
                stats = entry['stats']
                tick = entry['tick'] + 1
                
                if i != 0:
                    # 遍历stats中的键值对
                    for key, value in stats.items():
                        for prefix, event_type in EVENT_PREFIXES.items():
                            if key.startswith(prefix):
                                # 提取事件类型和名称
                                item_name = key[len(prefix):]

                                # 如果物品名称不在数据结构中，则初始化
                                if item_name not in meta_actions[event_type]:
                                    meta_actions[event_type][item_name] = {'total_num': 0, 'timesteps': []}

                                # 如果统计数增加，则记录时间步长和总数
                                if key not in previous_stats or previous_stats[key] != value:
                                    meta_actions[event_type][item_name]['timesteps'].append(tick)
                                    meta_actions[event_type][item_name]['total_num'] += 1
                # 更新前一步的状态
                previous_stats = stats.copy()
            # 删除 cache dir idx
            self.clean_cache(cache_dir)

            if any( len(meta_actions[event_type]) > 0 for event_type in EVENT_PREFIXES.values()):
                return meta_actions
            else:
                print(f"Since no required event occurred, the episode {idx} metadata action is not valid...")
                return None
                
        except Exception as e:
            print(f'Failed to download {action_url}: {e}')
            self.clean_cache(cache_dir)
            return None

    def __len__(self):
        # VPT Version 中有多少个视频
        return len(self.index['relpaths'])

    def clean_cache(self, cache_dir):
        # 删除 cache dir idx 中的 视频和对应的动作
        """Removes all cached videos and actions."""
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

# 将鼠标叠加到视频帧上
def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


# 将 VPT Format 变成 MineRL Format (按键盘，转动鼠标，按鼠标这三类不算是空动作)
def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

# Given video and action json idx path
def load_episode(video_path, json_data):
    # load video from idx path
    video = cv2.VideoCapture(video_path)

    # attack_is_stuck用来追踪攻击按钮是否一直按着（对于数据错误的处理），last_hotbar用来记录上一次选中的快捷栏（hotbar）索引
    attack_is_stuck = False
    last_hotbar = 0

    frames, frames_mineclip, actions = [], [], []

    # 处理每一帧对应的动作
    for i in range(len(json_data)):
        # 获取当前帧对应的动作
        step_data = json_data[i]
        if i == 0:
            # Check if attack will be stuck down
            # 判断鼠标左键是否一直按着（第一个应该是数据的错误）
            if step_data['mouse']['newButtons'] == [0]:
                attack_is_stuck = True
        # 如果鼠标左键一直按着（错误状态）
        elif attack_is_stuck:
            # 此时是下一个鼠标左键持续按着的状态（此时已经不是错误状态了）
            if 0 in step_data['mouse']['newButtons']:
                # 之后恢复正常
                attack_is_stuck = False
        # If still stuck, remove the action; 此时处于鼠标错误状态，对于鼠标的状态无需记录（就是一开始的鼠标状态是有问题的）
        if attack_is_stuck:
            step_data['mouse']['buttons'] = [button for button in step_data['mouse']['buttons'] if button != 0]

        # 将当前帧的动作从 VPT Format 变成 MineRL 格式是数据 
        action, is_null_action = json_action_to_env_action(step_data)

        # Update hotbar selection; 更新快捷栏索引
        current_hotbar = step_data['hotbar']
        if current_hotbar != last_hotbar:
            action['hotbar.{}'.format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar

        # Read frame even if this is null so we progress forward
        ret, frame = video.read()
        if ret:
            # Skip null actions as done in the VPT paper
            # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
            #       We do this here as well to reduce amount of data sent over.
            if is_null_action:
                continue
            # 检测 Craft GUI 是否打开
            if step_data["isGuiOpen"]:
                # 计算相机缩放因子，这可能是为了调整鼠标位置。
                camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                # 根据缩放因子计算游戏中鼠标的x和y位置。
                cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                # 将一个带有alpha通道的图像（如游戏内的鼠标光标）合成到视频帧上
                composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
            # 将帧从BGR颜色空间转换为RGB颜色空间, 视频的image是 BGR 类型
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            # 使用np.clip函数将帧的所有像素值限制在0到255的范围内，然后将其转换为np.uint8类型的NumPy数组（因为图像数据的像素值通常表示为0到255的整数
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            # 处理成 mineclip 需要的 image 格式
            mineclip_frame = process_frame_mineclip(frame)
            # 调用resize_image函数将帧缩放到某个预设的分辨率(AGENT_RESOLUTION)
            frame = resize_image(frame, AGENT_RESOLUTION)
            frames.append(frame)
            frames_mineclip.append(mineclip_frame)
            actions.append(action)

    video.release()
    return frames, frames_mineclip, actions
