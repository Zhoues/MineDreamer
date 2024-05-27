import base64
import numpy as np
import requests
import cv2


class Dreamer:
    def __init__(self, dreamer_url):
        self.dreamer_url = dreamer_url
    
    def generate_goal_image(self, suffix, text_prompt, current_image_path, is_del=0):
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

    def generate_t2v_video(self, suffix, text_prompt, video_save_path):
        data = {'text_prompt': text_prompt}
        response = requests.post(self.dreamer_url + suffix, data=data)

        with open(video_save_path, 'wb') as f:
            f.write(response.content)

        return video_save_path

    def generate_ti2v_video(self, suffix, text_prompt, video_save_path, current_image_path, is_del=0):
        with open(current_image_path, 'rb') as f:
            file = {'current_image': f}
            data = {'text_prompt': text_prompt, 'is_del': is_del}
            response = requests.post(self.dreamer_url + suffix, files=file, data=data)
            
        with open(video_save_path, 'wb') as f:
            f.write(response.content)

        return video_save_path