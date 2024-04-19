import sys
###################################################
####### Set the path to the repository here #######
sys.path.append("../Multimodal-GPT/")
###################################################

import torch
from torch import nn
from pathlib import Path
from PIL import Image

from mmgpt.models.builder import create_model_and_transforms
from model.base import LargeMultimodalModel

mmgpt_path = Path("../Multimodal-GPT/")

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
response_split = "### Response:"

class Inferencer:

    def __init__(self, finetune_path, llama_path, open_flamingo_path):
        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)
        model, image_processor, tokenizer = create_model_and_transforms(
            model_name="open_flamingo",
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            pretrained_model_path=open_flamingo_path,
            tuning_config=tuning_config,
        )
        model.load_state_dict(state_dict, strict=False)
        model.half()
        model = model.to("cuda")
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, prompt, images, max_new_token, num_beams, temperature,
                 top_k, top_p, do_sample):
        if len(images) > 1:
            raise gr.Error(
                "Current only support one image, please clear gallery and upload one image"
            )
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        if len(images) == 0 or images is None:
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(True)
            output_ids = self.model.lang_encoder.generate(
                input_ids=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )[0]
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(False)
        else:
            vision_x = [self.image_processor(im).unsqueeze(0) for im in images]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).half()
            
            outputs = self.model.generate(
                vision_x=vision_x.cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            output_ids = outputs["sequences"][0]

        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
#         print("generated_text:", generated_text)
        result = generated_text.split(response_split)[-1].strip()
        return result, outputs


class PromptGenerator:

    def __init__(
        self,
        prompt_template=TEMPLATE,
        ai_prefix="Response",
        user_prefix="Instruction",
        sep: str = "\n\n### ",
        buffer_size=-1,
    ):
        self.all_history = list()
        self.ai_prefix = ai_prefix
        self.user_prefix = user_prefix
        self.buffer_size = buffer_size
        self.prompt_template = prompt_template
        self.sep = sep

    def add_message(self, role, message):
        self.all_history.append([role, message])

    def get_images(self):
        img_list = list()
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        for his in all_history:
            if type(his[-1]) == tuple:
                img_list.append(his[-1][-1])
        return img_list

    def get_prompt(self):
        format_dict = dict()
        if "{user_prefix}" in self.prompt_template:
            format_dict["user_prefix"] = self.user_prefix
        if "{ai_prefix}" in self.prompt_template:
            format_dict["ai_prefix"] = self.ai_prefix
        prompt_template = self.prompt_template.format(**format_dict)
        ret = prompt_template
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        context = []
        have_image = False
        for role, message in all_history[::-1]:
            if message:
                if type(message) is tuple and message[
                        1] is not None and not have_image:
                    message, _ = message
                    context.append(self.sep + "Image:\n<image>" + self.sep +
                                   role + ":\n" + message)
                else:
                    context.append(self.sep + role + ":\n" + message)
            else:
                context.append(self.sep + role + ":\n")

        ret += "".join(context[::-1])
        return ret


class MMGPT(LargeMultimodalModel):
    def __init__(self, args):
        super(MMGPT, self).__init__()
        llama_path = str(mmgpt_path / "checkpoints/llama-7b_hf")
        open_flamingo_path = str(mmgpt_path / "checkpoints/OpenFlamingo-9B/checkpoint.pt")
        finetune_path = str(mmgpt_path / "checkpoints/mmgpt-lora-v0-release.pt")

        self.inferencer = Inferencer(
            llama_path=llama_path,
            open_flamingo_path=open_flamingo_path,
            finetune_path=finetune_path)
        
        self.max_new_token = 200
        self.num_beams = 1
        self.temperature = 1.0
        self.top_k = 20
        self.top_p = 1.0
        self.do_sample = False
        
        self.conv = PromptGenerator()
        
    def refresh_chat(self):
        self.conv.all_history = []
        
    def forward_with_probs(self, image, prompt):
        self.refresh_chat()
        image = Image.fromarray(image).convert("RGB")
        
        if image:
            self.conv.add_message(self.conv.user_prefix, (prompt, image))
        else:
            self.conv.add_message(self.conv.user_prefix, prompt)
        self.conv.add_message(self.conv.ai_prefix, None)
        inputs = self.conv.get_prompt()
        images = self.conv.get_images()[-1:]
#         print(inputs)
        
        inference_results, outputs = self.inferencer(inputs, images, self.max_new_token,
                               self.num_beams, self.temperature, self.top_k, self.top_p,
                               self.do_sample)
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]

        return inference_results, output_ids, logits, probs
    