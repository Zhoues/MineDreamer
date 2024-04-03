import base64
import numpy as np
import requests
import cv2


class Dreamer:
    def __init__(self, dreamer_url):
        self.dreamer_url = dreamer_url
    
    # Support Active Perception And Caption
    def generate_goal_image(self, suffix, text_prompt, current_image_path, is_del=0):
        # Ask MLLM and Get Answer
        with open(current_image_path, 'rb') as f:
            file = {'current_image': f}
            data = {'text_prompt': text_prompt, 'is_del': is_del}
            response = requests.post(self.dreamer_url + suffix, files=file, data=data)
            base64_goal_image_list = response.json()['goal_image_list']

            goal_image_list = []

            for base64_goal_image in base64_goal_image_list:

                img_data = base64.b64decode(base64_goal_image)

                nparr = np.frombuffer(img_data, np.uint8)

                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                goal_image_list.append(img)

            return goal_image_list