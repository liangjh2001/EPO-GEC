#!/bin/bash


model_name_or_path="your_path_to_sft_stage1_full_model"
dataset="FCGEC_train_single"
val_dataset="FCGEC_valid"
lr=3e-5

output_dir="qwen2-sft-stage2"
if [ ! -d "../saves/${output_dir}" ]; then
  mkdir -p "../saves/${output_dir}"
fi

cp $0 "../saves/${output_dir}"

CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 ../src/train_bash.py \
    --stage sft \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --dataset_dir ../data \
    --max_samples 100 \
    --dataset ${dataset} \
    --val_dataset ${val_dataset} \
    --preprocessing_num_workers 60 \
    --cutoff_len 200 \
    --template cgec \
    --finetuning_type lora \
    --lora_rank 32 \
    --lora_target all \
    --output_dir "../saves/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy="steps" \
    --eval_steps 0.05 \
    --save_strategy="steps" \
    --save_steps 0.05 \
    --save_only_model True \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --plot_loss \
    --bf16 \
    --log_level info \
    2>&1 | tee "../saves/${output_dir}/${output_dir}.log" &

