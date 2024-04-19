import os
import json

import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl

class POPEDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco/"):
        super(POPEDataset, self).__init__()
        self.ann_path = f"./data/pope/coco_{split}"
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.split = split
         
    def get_data(self):
        data = []
        cats = ["adversarial", "popular", "random"]
        for category in cats:
            ann = read_jsonl(os.path.join(self.ann_path, f"coco_{self.split}_pope_{category}.json"))
            data_cat = [
                {
                    "img_path": os.path.join(self.img_root, ins['image']),
                    "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                    "label": 0 if ins['label'] == 'no' else 1,
                    "question_id": ins["question_id"],
                    "category": category
                }
                for ins in ann
            ]
            if len(data_cat) >= 6000: # Subsample
                data_cat = data_cat[:6000]
#                 idx = np.random.choice(range(len(data_cat)), 500, replace=False)
#                 data_cat = [data_cat[i] for i in idx]
    
            data += data_cat
            
        return data, ["question_id", "category"]