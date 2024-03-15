import os
import json
import warnings
import argparse
warnings.filterwarnings('ignore')

import torch
import numpy as np
from glob import glob
from tqdm.auto import tqdm, trange
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader

from utils.func import read_jsonl, read_data
from utils.metric import evaluate, eval_pope, eval_safety
from method.refiner import train_module

# Unanswerable: 853
# Answerable: 673
# unanswerable: 443
# answerable: 1234
# Yes: 3869
# No: 1939


def main(args):
    print(args.model_name, args.prompt)
        
    if args.benchmark == "VizWiz":
        _, x_train_ori, y_train_ori = read_data(args.model_name, "VizWiz", split="train", prompt=args.prompt)
        val_data, x_val, y_val = read_data(args.model_name, "VizWiz", split="val", prompt=args.prompt, return_data=True)

        print(x_train_ori.shape, y_train_ori.shape, x_val.shape, y_val.shape)
        
        res_acc, res_f1, res_auroc = [], [], []
        k = args.K
        
        for _ in trange(args.num_exp):
            sample_idx = np.random.choice(range(len(x_train_ori)), k, replace=False)
            x_train = x_train_ori[sample_idx]
            y_train = y_train_ori[sample_idx]

#             model = LogisticRegression()
#             model.fit(x_train, y_train)

#             y_pred = model.predict_proba(x_val)[:, 1]
#             acc, ap, f1, auroc = evaluate(y_val, y_pred, show=False)

            from sklearn.decomposition import PCA

            # Apply PCA
            pca = PCA(n_components=100)
            x_train_pca = pca.fit_transform(x_train)
            x_val_pca = pca.transform(x_val)

            # Train Logistic Regression
            logistic_regression_model = LogisticRegression(random_state=42)
            logistic_regression_model.fit(x_train_pca, y_train)

            y_pred = logistic_regression_model.predict_proba(x_val_pca)[:, 1]
            acc, ap, f1, auroc = evaluate(y_val, y_pred, show=False)

            res_acc.append(acc*100.)
            res_f1.append(f1*100.)
            res_auroc.append(auroc*100.)

        print(f"ACC: {np.mean(res_acc):.2f} (±{np.std(res_acc):.2f})")
        print(f"F1: {np.mean(res_f1):.2f} (±{np.std(res_f1):.2f})")
        print(f"AUROC: {np.mean(res_auroc):.2f} (±{np.std(res_auroc):.2f})")
        
    if args.benchmark == "Safety":
        _, x_train_ori, y_train_ori = read_data(args.model_name, "Safety", split="train", prompt=args.prompt)
        assert len(x_train_ori) == 1093
        val_data, x_val, y_val = read_data(args.model_name, "Safety", split="val", prompt=args.prompt, return_data=True)
        _, x_seed, y_seed = read_data(args.model_name, "Safety", split="SEED", prompt=args.prompt)
        
        k = args.K
        asr, rr = [], []qinyu
        
        if args.num_exp == 1:
            x_train = x_train_ori
            y_train = y_train_ori

            model = LogisticRegression()
            model.fit(x_train, y_train)

            y_pred = model.predict_proba(x_val)[:, 1]
            asr = eval_safety(val_data, y_pred)

            y_seed_pred = model.predict_proba(x_seed)[:, 1]
            acc, _, _, _  = evaluate(y_seed, y_seed_pred, show=False)
            
            print(f"ASR: {asr:.2f}")
            print(f"SEED.RR: {100-acc*100:.2f}")
        
        else:
            for _ in trange(args.num_exp):
                x_train_pos = x_train_ori[y_train_ori == 1]
                x_train_neg = x_train_ori[y_train_ori == 0]

                x_train_neg = x_train_neg[np.random.choice(range(len(x_train_neg)), k, replace=False)]
                x_train = np.concatenate([x_train_pos, x_train_neg], axis=0)
                y_train = np.array([1] * len(x_train_pos) + [0] * len(x_train_neg))

                model = LogisticRegression()
                model.fit(x_train, y_train)

                y_pred = model.predict_proba(x_val)[:, 1]
                res = eval_safety(val_data, y_pred)
                asr.append(res)

                y_seed_pred = model.predict_proba(x_seed)[:, 1]
                acc, _, _, _  = evaluate(y_seed, y_seed_pred, show=False)
                rr.append(100-acc*100)

            print(f"ASR: {np.mean(asr):.2f} (±{np.std(asr):.2f})")
            print(f"SEED.RR: {np.mean(rr):.2f} (±{np.std(asr):.2f})")
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate the model performance')
    parser.add_argument("--model_name", default="InstructBLIP")
    parser.add_argument("--prompt", default="oe")
    parser.add_argument("--benchmark", default="VizWiz")
    parser.add_argument("--num_exp", type=int, default=100)
    parser.add_argument("--K", type=int, default=1000)
    
    main(parser.parse_args())