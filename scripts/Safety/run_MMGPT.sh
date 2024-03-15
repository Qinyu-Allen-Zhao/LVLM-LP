#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt mq \
        --theme safety \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/Safety_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt oe \
        --theme safety \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/Safety_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt oeh \
        --theme safety \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/Safety_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}.jsonl
done