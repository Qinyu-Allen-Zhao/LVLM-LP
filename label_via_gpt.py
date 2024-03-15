import os
import json

from tqdm.auto import trange

from utils.func import read_jsonl
from utils.annotate import perform_eval_safe, perform_eval_ans, perform_eval_dec

prompt = 'oe'
dataset = "MAD_val"

for model_name in [
#     "LLaVA-13B", "LLaVA-7B", 
#     "MiniGPT4", "LLaMA_Adapter", 
    "InstructBLIP", 
#     "mPLUG-Owl","MMGPT", 
]:
    print(f"========={model_name}=========")
    
    data = read_jsonl(f"./output/{model_name}/{dataset}_{prompt}.jsonl")
    output_file = f"./output/{model_name}/{dataset}_{prompt}_labeled.json"

    if os.path.exists(output_file):
        label = json.load(open(output_file))
    else:
        label = []
    
    if dataset == "Safety":
        label_data = [ins for ins in data if ins['scenario'] != 'coco']


        for i in trange(len(label), len(label_data)):
            ins = label_data[i]

            is_safe = perform_eval(ins)
            label.append({
                'image': ins['image'],
                 'model_name': ins['model_name'],
                 'question': ins['question'],
                 'label': ins['label'],
                 'response': ins['response'],
                "is_safe": is_safe,
                "scenario": ins['scenario']
            })

            if i % 10 == 0:
                json.dump(label, open(output_file, 'w'), indent=4)

        json.dump(label, open(output_file, 'w'), indent=4)
    
    else:
        for i in trange(len(label), len(data)):
            ins = data[i]

            is_answer = perform_eval_ans(ins) if dataset == "VizWiz_val" else perform_eval_dec(ins)
            label.append({
                'image': ins['image'],
                 'model_name': ins['model_name'],
                 'question': ins['question'],
                 'label': ins['label'],
                 'response': ins['response'],
                "is_answer": is_answer,
            })

            if i % 10 == 0:
                json.dump(label, open(output_file, 'w'), indent=4)

        json.dump(label, open(output_file, 'w'), indent=4)