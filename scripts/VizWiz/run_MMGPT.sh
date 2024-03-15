#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split val \
        --dataset VizWiz \
        --prompt oe \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_val_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split train \
        --dataset VizWiz \
        --prompt oe \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_train_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split val \
        --dataset VizWiz \
        --prompt oeh \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_val_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split train \
        --dataset VizWiz \
        --prompt oeh \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_train_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split val \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name MMGPT \
        --split train \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/MMGPT/VizWiz_train_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl >> "$output_file"
    rm ./output/MMGPT/tmp/${CHUNKS}_${IDX}_vw.jsonl
done


