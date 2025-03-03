#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

base_model="your_base_model_path"
lora_model="your_lora_model_path"
export_dir="your_export_dir"
template="gec" # gec or cgec

echo "----------------------------merge lora weight----------------------------"
python ../src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path ${lora_model} \
    --template ${template} \
    --finetuning_type lora \
    --export_dir ${export_dir} \
    --export_size 2 \
    --export_legacy_format False