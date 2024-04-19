import os
import json

from dataset.base import BaseDataset

class VizWizDataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data/VizWiz/"):
        super(VizWizDataset, self).__init__()
        self.ann = json.load(open(os.path.join(data_root, f"{split}.json"), 'r'))
        self.img_root = os.path.join(data_root, f"{split}/") 
        self.prompter = prompter
        self.split = split
         
    def get_data(self):
        data = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": self.prompter.build_prompt(ins['question']),
                "label": ins['answerable'] if self.split != "test" else "IDK"
            }
            for ins in self.ann
        ]
        return data, []