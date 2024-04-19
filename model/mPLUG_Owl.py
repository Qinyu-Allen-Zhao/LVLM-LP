import sys
###################################################
####### Set the path to the repository here #######
sys.path.append("../mPLUG-Owl/mPLUG-Owl2")
###################################################

import torch
from torch import nn
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from model.base import LargeMultimodalModel

class mPLUG_Owl(LargeMultimodalModel):
    def __init__(self, args):
        super(mPLUG_Owl, self).__init__()
        model_path = 'MAGAer13/mplug-owl2-llama2-7b'
        model_name = get_model_name_from_path(model_path)
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
        self.temperature = 0.0 # args.temperature
        self.top_p = 0.9 # args.top_p
        self.num_beams = 1 # args.num_beams
        
    def _basic_forward(self, image, prompt, return_dict=False):
        image = Image.fromarray(image).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)


        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                streamer=streamer,
                use_cache=True,
                max_new_tokens = 512,
                stopping_criteria=[stopping_criteria],
                
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                
                return_dict_in_generate=return_dict,
                output_attentions=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict
            )

        return outputs
    
    
    def forward_with_probs(self, image, prompt):
        outputs = self._basic_forward(image, prompt, return_dict=True)
        
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
            
        response = self.tokenizer.decode(output_ids).strip()[:-4]

        output_ids = output_ids.cpu().numpy()
        
        return response, output_ids, logits, probs