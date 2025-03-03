#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

lora_model="mistral-sft-stage2"
base_model="your_path_to_sft_stage1_full_model"
checkpoint="1"


echo ----------------------------merge lora weight----------------------------
python ../src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path "../saves/${lora_model}/checkpoint-${checkpoint}" \
    --template gec \
    --finetuning_type lora \
    --export_dir "../saves/${lora_model}/checkpoint-${checkpoint}/full-model" \
    --export_size 2 \
    --export_legacy_format False


echo --------------using vllm to infer train for output multi seqs----------------------
python ../utils/vllm_output_multi_seqs.py \
        --data_dir "../data/wi_locness.json" \
        --output_dir "../data/${lora_model}-checkpoint-${checkpoint}-wi_locness_multi_predict.json" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo --------------using vllm to infer dev for output multi seqs----------------------
python ../utils/vllm_output_multi_seqs.py \
        --data_dir "../data/bea_dev.json" \
        --output_dir "../data/${lora_model}-checkpoint-${checkpoint}-bea_dev_multi_predict.json" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"





