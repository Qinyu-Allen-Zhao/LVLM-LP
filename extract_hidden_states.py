import os
import json
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk
from utils.prompt import Prompter


def get_model_output(args, sub_idx, data, extra_keys, model):
    for i in tqdm(sub_idx):
        ins = data[i]
        img_id = ins['img_path'].split("/")[-1]
        image = cv2.imread(ins['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prompt = ins['question']
        
        outputs = model._basic_forward(image, prompt, return_dict=True)
        hidden_states = outputs['hidden_states'][1]
        
        out = {
            "hidden_states": hidden_states, 
            "label": ins['label']
        }
        for k in extra_keys:
            out[k] = ins[k]
        torch.save(out, os.path.join(args.store_path, f"{i}.pth"))


def main(args):
    model = build_model(args)
    
    prompter = Prompter(args.prompt, args.theme)
    
    data, extra_keys = build_dataset(args.dataset, args.split, prompter)
    if args.num_samples is not None:
#         np.random.shuffle(data)
        data = data[:args.num_samples]
    sub_idx = get_chunk(list(range(len(data))), args.num_chunks, args.chunk_idx)
    
    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)
    
    results = get_model_output(args, sub_idx, data, extra_keys, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", default="LLaVA-13B")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", default="VizWiz")
    parser.add_argument("--prompt", choices=['bc', 'oe', 'oeh', 'conf', "cot"], default='bc')
    parser.add_argument("--theme", default='unanswerable')
    parser.add_argument("--store_path", type=str, default="./output/LLaVA-13B/hidden_states/")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    
    main(parser.parse_args())