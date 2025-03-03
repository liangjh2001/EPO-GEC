#!/bin/bash
export CUDA_VISIBLE_DEVICES=3


lora_model="qwen2-sft-stage2"
base_model="your_path_to_sft_stage1_full_model"
checkpoint="1"


echo ----------------------------merge lora weight----------------------------
python ../src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path "../saves/${lora_model}/checkpoint-${checkpoint}" \
    --template cgec \
    --finetuning_type lora \
    --export_dir "../saves/${lora_model}/checkpoint-${checkpoint}/full-model" \
    --export_size 2 \
    --export_legacy_format False


echo --------------using vllm to infer train for output multi seqs----------------------
python ../utils/cgec_vllm_output_multi_seqs.py \
        --data_dir "../data/FCGEC_train_single.json" \
        --output_dir "../data/${lora_model}-checkpoint-${checkpoint}-FCGEC_train_single_multi_predict.json" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo --------------using vllm to infer dev for output multi seqs----------------------
python ../utils/cgec_vllm_output_multi_seqs.py \
        --data_dir "../data/FCGEC_valid.json" \
        --output_dir "../data/${lora_model}-checkpoint-${checkpoint}-FCGEC_valid_multi_predict.json" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"





