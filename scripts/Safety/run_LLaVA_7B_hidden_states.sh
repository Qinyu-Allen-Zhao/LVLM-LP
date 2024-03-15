#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m extract_hidden_states \
        --model_name LLaVA-7B \
        --model_path liuhaotian/llava-v1.5-7b \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt oe \
        --theme safety \
        --store_path ./output/LLaVA-7B/hidden_states/Safety \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait
