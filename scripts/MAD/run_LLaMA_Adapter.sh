#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split val \
        --dataset MAD \
        --prompt mq \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split train \
        --dataset MAD \
        --prompt mq \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_train_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split val \
        --dataset MAD \
        --prompt oe \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_val_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split train \
        --dataset MAD \
        --prompt oe \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_train_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split val \
        --dataset MAD \
        --prompt oeh \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_val_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split train \
        --dataset MAD \
        --prompt oeh \
        --theme mad \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/MAD_train_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}.jsonl
done