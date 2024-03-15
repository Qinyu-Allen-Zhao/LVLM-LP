#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path liuhaotian/llava-v1.5-7b \
        --split val \
        --dataset ImageNet \
        --prompt oe \
        --answers_file ./output/LLaVA-7B/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B/ImageNet_val.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python -m run_model \
        --num_samples 16 \
        --sampling class \
        --model_name LLaVA-7B \
        --model_path liuhaotian/llava-v1.5-7b \
        --split train \
        --dataset ImageNet \
        --prompt oe \
        --answers_file ./output/LLaVA-7B/ImageNet_Train/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

