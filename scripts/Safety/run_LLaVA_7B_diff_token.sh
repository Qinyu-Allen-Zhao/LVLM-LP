#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for TOKEN in $(seq 3 9); do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
            --model_name LLaVA-7B \
            --model_path liuhaotian/llava-v1.5-7b \
            --split SD_TYPO \
            --dataset MMSafety \
            --prompt oe \
            --theme safety \
            --answers_file ./output/tmp/${CHUNKS}_${IDX}.jsonl \
            --num_chunks $CHUNKS \
            --chunk_idx $IDX \
            --temperature 0.0 \
            --top_p 0.9 \
            --num_beams 1 \
            --token_id $TOKEN &
    done

    wait

    output_file=./output/LLaVA-7B/Safety_oe_${TOKEN}.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./output/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
        rm ./output/tmp/${CHUNKS}_${IDX}.jsonl
    done
done

# Use the last token
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path liuhaotian/llava-v1.5-7b \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt oe \
        --theme safety \
        --answers_file ./output/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 \
        --token_id -1 &
done

wait

output_file=./output/LLaVA-7B/Safety_val_oe_neg_1.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/tmp/${CHUNKS}_${IDX}.jsonl
done
