#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split val \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/VizWiz_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done

# VizWiz train
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split train \
        --dataset VizWiz \
        --prompt mq \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/VizWiz_train_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split SD_TYPO \
        --dataset MMSafety \
        --prompt mq \
        --theme safety \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/Safety_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split val \
        --dataset MAD \
        --prompt mq \
        --theme mad \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/MAD_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split train \
        --dataset MAD \
        --prompt mq \
        --theme mad \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/MAD_train_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split val \
        --dataset POPE \
        --prompt oe \
        --theme general \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/POPE_val_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split train \
        --dataset POPE \
        --prompt oe \
        --theme general \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/POPE_train_oe.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done



for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split LLaVA-7B \
        --dataset MathVista \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/MathV_self_eval.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done



for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split val \
        --dataset ImageNet \
        --prompt oe \
        --answers_file ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B-FT/ImageNet_val.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm ./output/LLaVA-7B-FT/tmp/${CHUNKS}_${IDX}.jsonl
done


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python -m run_model \
        --num_samples 16 \
        --sampling class \
        --model_name LLaVA-7B \
        --model_path ../LLaVA/checkpoints/llava-v1.5-7b-lora \
        --split train \
        --dataset ImageNet \
        --prompt oe \
        --answers_file ./output/LLaVA-7B-FT/ImageNet_Train/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait