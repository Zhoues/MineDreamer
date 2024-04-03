import os
import torch
import numpy as np
import torch.nn.functional as F
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from minedreamer.utils.embed_utils import embed_one_frame_openclip
from minedreamer.utils.file_utils import load_json, load_pickle
from minedreamer.utils.mineclip_agent_env_utils import load_openclip_wconfig

class MultiModalMemory:
    def __init__(self, memory_dir="data/memory"):
        self.memory_dir = memory_dir
        self.memory_list_path = os.path.join(self.memory_dir, "memory_list.json")
        self.memory_db_dir = os.path.join(self.memory_dir, "memory_db")
        
        os.makedirs(self.memory_db_dir, exist_ok=True)

        self.openclip = load_openclip_wconfig()
        self.memory_list = load_json(self.memory_list_path)
        self.memory = {}

        for memory_key in self.memory_list:
            self.load_memory(memory_key)

        
    def load_memory(self, memory_key):
        memory_db_path = os.path.join(self.memory_db_dir, f"{memory_key}.pkl")
        self.memory[memory_key] = load_pickle(memory_db_path)

    def search_memory(self, frame_path, memory_key):
        if memory_key not in self.memory.keys():
            print(f"WARNING: There is no memory for {memory_key}")
            return None

        memory = self.memory[memory_key]
        frame_embed = embed_one_frame_openclip([frame_path], self.openclip)
        frame_embed = torch.from_numpy(np.array(frame_embed)).float().cuda()

        scores = []

        for memory_embeds in memory:
            """start_frame_openclip_embed, end_frame_img, goal_visual_embed"""
            memory_embed = memory_embeds[0].cuda()
            score = F.cosine_similarity(frame_embed, memory_embed).item()
            scores.append(score)

            sorted_scores_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            sorted_scores = [scores[idx] for idx in sorted_scores_indices]

        print(f"[Top 1 score: {sorted_scores[0]}] Finish searching mm memory")
        best_frame_embeds = memory[sorted_scores_indices[0]]

        start_frame_openclip_embed, end_frame_img, goal_visual_embed = best_frame_embeds[0], best_frame_embeds[1], best_frame_embeds[2]
        return start_frame_openclip_embed, end_frame_img, goal_visual_embed