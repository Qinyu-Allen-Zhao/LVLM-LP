import os
import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk
from utils.prompt import Prompter


def get_model_output(args, data, model, extra_keys, answers_file):
    ans_file = open(answers_file, 'w')
    
    for ins in tqdm(data):
        img_id = ins['img_path'].split("/")[-1]
        if args.model_name == "GPT4V":
            image = [ins['img_path']]
            prompt = ins['question']
            response = model.forward(image, prompt)
            out = {
                "image": img_id,
                "question": prompt,
                "label": ins["label"],
                "response": response,
            }
            print(response)
            for key in extra_keys:
                out[key] = ins[key]
            ans_file.write(json.dumps(out) + "\n")
            
        else:
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            prompt = ins['question']
            response, output_ids, logits, probs = model.forward_with_probs(image, prompt)
            if len(logits) <= args.token_id:
                continue
            out = {
                "image": img_id,
                "model_name": args.model_name,
                "question": prompt,
                "label": ins["label"],
                "response": response,
                "output_ids": output_ids.tolist(),
                "logits": logits.tolist()[args.token_id],
#                 "probs": probs.tolist()[args.token_id],
            }
            for key in extra_keys:
                out[key] = ins[key]
            ans_file.write(json.dumps(out) + "\n")            
        
        ans_file.flush()

    ans_file.close()


def main(args):
    model = build_model(args)
    
    prompter = Prompter(args.prompt, args.theme)
    
    data, extra_keys = build_dataset(args.dataset, args.split, prompter)
    if args.num_samples is not None:
        if args.sampling == 'first':
            data = data[:args.num_samples]
        elif args.sampling == "random":
            np.random.shuffle(data)
            data = data[:args.num_samples]
        else:
            labels = np.array([ins['label'] for ins in data])
            classes = np.unique(labels)
            data = np.array(data)
            final_data = []
            for cls in classes:
                cls_data = data[labels == cls]
                idx = np.random.choice(range(len(cls_data)), args.num_samples, replace=False)
                final_data.append(cls_data[idx])
            data = list(np.concatenate(final_data))

    data = get_chunk(data, args.num_chunks, args.chunk_idx)
        
    if not os.path.exists(f"./output/{args.model_name}/"):
        os.makedirs(f"./output/{args.model_name}/")
    
    results = get_model_output(args, data, model, extra_keys, args.answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", default="LLaVA-7B")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sampling", choices=['first', 'random', 'class'], default='first')
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", default="VizWiz")
    parser.add_argument("--prompt", default='oeh')
    parser.add_argument("--theme", default='unanswerable')
    parser.add_argument("--answers_file", type=str, default="./output/LLaVA/LLaVA_VizWiz_val_oeh.jsonl")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--token_id", type=int, default=0)
    
    main(parser.parse_args())