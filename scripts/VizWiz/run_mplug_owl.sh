#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# VizWiz val
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split val \
        --dataset VizWiz \
        --prompt oe \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_val_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split train \
        --dataset VizWiz \
        --prompt oe \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_train_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done



# VizWiz val
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split val \
        --dataset VizWiz \
        --prompt oeh \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_val_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split train \
        --dataset VizWiz \
        --prompt oeh \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_train_oeh.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done


# VizWiz val
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split val \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path 'MAGAer13/mplug-owl2-llama2-7b' \
        --split train \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/mPLUG-Owl/VizWiz_train_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/mPLUG-Owl/tmp/${CHUNKS}_${IDX}.jsonl
done