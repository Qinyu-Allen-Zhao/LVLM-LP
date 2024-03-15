#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name InstructBLIP \
        --split InstructBLIP \
        --dataset MathVista \
        --prompt oe \
        --theme uncertainty \
        --answers_file ./output/InstructBLIP/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/InstructBLIP/MathV_self_eval.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/InstructBLIP/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/InstructBLIP/tmp/${CHUNKS}_${IDX}.jsonl
done