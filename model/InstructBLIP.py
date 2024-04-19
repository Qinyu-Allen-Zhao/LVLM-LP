import sys
###################################################
####### Set the path to the repository here #######
sys.path.append("../LAVIS")
###################################################

import numpy as np
from PIL import Image
import torch
from torch import nn

from lavis.models import load_model_and_preprocess
from model.base import LargeMultimodalModel

class InstructBLIP(LargeMultimodalModel):
    def __init__(self, args):
        super(InstructBLIP, self).__init__()
        # Load InstructBLIP model
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", 
            model_type="vicuna7b", 
            is_eval=True, 
            device="cuda")
        
        self.temperature = 1.0 #args.temperature
        self.top_p = 0.9 #args.top_p
        self.num_beams = 1 # args.num_beams
    
    def forward(self, image, prompt):
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # prepare the image
        processed_image = self.vis_processors["eval"](pil_image).unsqueeze(0).to("cuda")

        llm_message, _ = self.model.generate({"image": processed_image, 
                                           "prompt": prompt},
                                            temperature=self.temperature,
                                            top_p=self.top_p,
                                            num_beams=self.num_beams,
                                            max_length=20000,
                         )
        
        return llm_message[0]
    
    def forward_with_probs(self, image, prompt):
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # prepare the image
        processed_image = self.vis_processors["eval"](pil_image).unsqueeze(0).to("cuda")

        llm_message, outputs = self.model.generate({"image": processed_image, 
                                           "prompt": prompt},
                                           temperature=self.temperature,
                                            top_p=self.top_p,
                                            num_beams=self.num_beams,
                                            max_length=20000
                               )
        
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
        output_ids = output_ids.cpu().numpy()
        
        return llm_message[0], output_ids, logits, probs
    