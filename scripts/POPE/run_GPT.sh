#!/bin/bash

python -m run_model \
    --model_name GPT4V \
    --split val \
    --dataset POPE \
    --answers_file ./output/GPT4V/POPE_val_oe.jsonl