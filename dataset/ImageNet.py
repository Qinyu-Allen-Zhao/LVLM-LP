import os
import json

from glob import glob

from dataset.base import BaseDataset
from utils.func import read_jsonl

class ImageNetDataset(BaseDataset):
    def __init__(self, split="val"):
        super(ImageNetDataset, self).__init__()
        self.wnid_to_idx = json.load(open("./data/ImageNet2012_wnid_to_idx.json"))
        self.img_root = f"/data/ImageNet2012/{split}"
        self.split = split
         
    def get_data(self):
        data = []
        for wnid, idx in self.wnid_to_idx.items():
            images = glob(os.path.join(self.img_root, wnid, "*.JPEG"))
            data += [
                {
                    "img_path": img,
                    "question": "Give me a one-word label for the foreground object in this image.",
                    "label": idx
                }
                for img in images
            ]
        return data, []