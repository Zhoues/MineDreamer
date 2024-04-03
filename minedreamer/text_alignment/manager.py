import os
import random
import ast
import copy
from minedreamer.config import CVAE_INFO, LOAD_CVAE_KEY_INFO
from minedreamer.text_alignment.cvae import load_cvae_model
from minedreamer.utils.file_utils import load_json
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings


class CVAEManager:

    def __init__(self, prior_weights_dir="data/weights/cvae", mapping_dir="data/prompt_mapping", retrieval_top_k=1):
        self.mapping_db_dir = os.path.join(mapping_dir, "mapping_db")

        os.makedirs(self.mapping_db_dir, exist_ok=True)
        
        self.retrieval_top_k = retrieval_top_k
        assert retrieval_top_k == 1, "Only select the closest prior model"

        self.mapping_db = Chroma(
            collection_name=f"mapping_db",
            embedding_function=OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k"),
            persist_directory=self.mapping_db_dir
        )
        
        self.models = {}

        for model_key in LOAD_CVAE_KEY_INFO.keys():
            model_path = os.path.join(prior_weights_dir, f"{model_key}.pt")
            if os.path.exists(model_path):
                model_info = copy.deepcopy(CVAE_INFO)
                model_info["model_path"] = model_path
                cvae = load_cvae_model(model_info)
                self.add_model(model_key, cvae)

    def seach_user_prompt(self, user_prompt):
        k = min(self.mapping_db._collection.count(), self.retrieval_top_k)
        assert k == 1, "Mapping DB Num is 0!"

        print(f"Searching the closest prior model for {user_prompt}...")
        mapping_doc, mapping_score = self.mapping_db.similarity_search_with_score(user_prompt, k=k)[0]
        
        prompt, mapping_keys = mapping_doc.metadata["prompt"], ast.literal_eval(mapping_doc.metadata["mapping_key"])
        mapping_key = random.choice(mapping_keys)
        print(f"[Top 1 score: {1 - mapping_score}] Mapping '{user_prompt}' to '{prompt}', prior is '{mapping_key}'")

        return prompt, mapping_key

    def add_model(self, model_key, model):
        if model_key in self.models:
            raise ValueError(f"Model Key {model_key} already exists.")
        self.models[model_key] = model
    
    def remove_model(self, model_key):
        if model_key in self.models:
            del self.models[model_key]
        else:
            raise KeyError(f"Model Key {model_key} does not exist.")

    def update_mapping_db(self, mapping_dict_path="data/prompt_mapping/prompt_to_key.json"):
        mapping_dict = load_json(mapping_dict_path)
        mapping_prompt = list(mapping_dict.keys())

        for idx, prompt in enumerate(mapping_prompt):
            mapping_key = str(mapping_dict[prompt])
            self.mapping_db.add_texts(
                texts=[prompt],
                ids=[str(idx)],
                metadatas=[{"prompt": prompt, "mapping_key": mapping_key}],
            )
            self.mapping_db.persist()
            print(f"Succeed Mapping {prompt} to {mapping_key}")

    def get_mapping_db_len(self):
        return self.mapping_db._collection.count()

    def get_all_mapping_db(self):
        return self.mapping_db.get()
    
    def get_mapping_db_by_id(self, idx):
        return self.mapping_db.get(ids=[str(idx)])
    
    def del_mapping_db_by_id(self, idx):
        mapping_dict = self.get_mapping_db_by_id(idx)
        if len(mapping_dict["metadatas"]) > 0:
            prompt, mapping_key = mapping_dict["metadatas"][0]["prompt"], mapping_dict["metadatas"][0]["mapping_key"]
            self.mapping_db._collection.delete(ids=[str(idx)])
            self.mapping_db.persist()
            print(f"Succeed Deleting {prompt} to {mapping_key}")
    
    def del_all_mapping_db(self):
        mapping_db_len = self.mapping_db._collection.count()
        for idx in range(mapping_db_len):
            self.del_mapping_db_by_id(idx)