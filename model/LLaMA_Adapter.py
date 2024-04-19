import sys
###################################################
####### Set the path to the repository here #######
sys.path.append("../LLaMA-Adapter/llama_adapter_v2_multimodal7b")
###################################################

import torch
from PIL import Image

import llama
from model.base import LargeMultimodalModel

class LLaMA_Adapter(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaMA_Adapter, self).__init__()
        llama_dir = "../model/llama-v1/"

        # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
        self.model, self.preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device="cuda")
        self.model.eval()
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        
    def forward_with_probs(self, image, prompt):
        prompt = llama.format_prompt(prompt)
        image = Image.fromarray(image).convert('RGB')
        img = self.preprocess(image).unsqueeze(0).to("cuda")

        response, output_ids, logits = self.model.generate(img, [prompt],
                temperature=self.temperature,
                top_p=self.top_p)
        
        return response[0], output_ids, logits, None
    