import os
import json

from dataset.base import BaseDataset

class MADBench(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data/coco/"):
        super(MADBench, self).__init__()
        self.ann_root = "./data/MADBench/"
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
         
    def get_data(self):
        ann = json.load(open(os.path.join(self.ann_root, "Normal.json"), 'r'))
        normal_data = [
            {
                "img_path": ins["file"],
                "question": self.prompter.build_prompt(ins["Question(GPT)"]),
                "label": 1,
                "scenario": "Normal"
            }
            for ins in ann
        ]
        if self.split == "train":
            data = normal_data[:100]
        else:
            data = normal_data[100:]
        
        val_phrases = []
        for sc in ["CountOfObject", "NonexistentObject", "ObjectAttribute", "SceneUnderstanding", "SpatialRelationship"]:
            ann = json.load(open(os.path.join(self.ann_root, f"{sc}.json"), 'r'))
            print(sc)
            sc_data = [
                {
                    "img_path": ins["file"],
                    "question": self.prompter.build_prompt(ins["Question(GPT)"]),
                    "label": 0,
                    "scenario": sc
                }
                for ins in ann
            ]
            if self.split == "train":
                sc_data = sc_data[:20]
            else:
                sc_data = sc_data[20:]
            data += sc_data

        return data, ["scenario"]