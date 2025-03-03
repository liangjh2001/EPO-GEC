#!/bin/bash


model_name_or_path="your_path_to_sft_stage2_full_model"
dataset="your_path_to_FCGEC_train_single_pairwise_dataset"
val_dataset="your_path_to_FCGEC_valid_pairwise_dataset"
pref_loss=sigmoid
use_length_normalized=false
beta=0.5
ftx=0.0
use_edit_mask=true
pivot_token_weight=10
edit_token_weight=5
normal_token_weight=1
pref_margin=1.0
warmup_ratio=0.1
lr=5e-7


output_dir="qwen2-epo"

if [ ! -d "../saves/${output_dir}" ]; then
  mkdir -p "../saves/${output_dir}"
fi

cp $0 "../saves/${output_dir}"


CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 ../src/train_bash.py \
    --stage custom_dpo_pair \
    --pref_loss ${pref_loss} \
    --pref_beta ${beta} \
    --pref_ftx ${ftx} \
    --use_edit_mask ${use_edit_mask} \
    --use_length_normalized ${use_length_normalized} \
    --pivot_token_weight ${pivot_token_weight} \
    --edit_token_weight ${edit_token_weight} \
    --normal_token_weight ${normal_token_weight} \
    --pref_margin ${pref_margin} \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy="steps" \
    --eval_steps 0.125 \
    --save_strategy="steps" \
    --save_steps 0.125 \
    --save_only_model True \
    --num_train_epochs 3 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --plot_loss \
    --bf16 \
    --warmup_ratio ${warmup_ratio} \
    --log_level info \
    2>&1 | tee "../saves/${output_dir}/${output_dir}.log" &
