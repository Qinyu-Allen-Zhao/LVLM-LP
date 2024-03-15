import re
import json
import math

import numpy as np
from tqdm.auto import tqdm


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True) # only difference


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_conf_score(text="The price of the item is 12.99 dollars."):
    # Regular expression pattern to find a float number
    pattern = r"\b\d+\.\d+\b|\b\d+\b"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Extract the matched float number, if any
    float_number = match.group() if match else 0.0
    return float(float_number)


def read_jsonl(file, num=None):
    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            data.append(json.loads(line))
            
            if num and i == num:
                break

    return data


def read_data(model, dataset, split="val", prompt='oe', token_idx=0, return_data=False, num_samples=None):
    x, y = [], []
    
    with open(f"./output/{model}/{dataset}_{split}_{prompt}.jsonl") as f:
        dataset = []
        for line in tqdm(f):
            data = json.loads(line)
            if return_data:
                dataset.append(data)
#             if np.isnan(np.mean(data['logits'])):
#                 continue
            x.append(data['logits'])
            y.append(data['label'])
            if num_samples is not None and len(x) > num_samples:
                break
        print(line[:1000])
    x, y = np.array(x), np.array(y)
    print(x.shape, y.shape)
    return dataset, np.squeeze(x), np.squeeze(y)


