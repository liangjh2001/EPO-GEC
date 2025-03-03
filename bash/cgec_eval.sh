#!/bin/bash
export CUDA_VISIBLE_DEVICES=3



lora_model="qwen2-epo"
base_model="your_path_to_sft_stage2_full_model"
checkpoint="1"

mkdir -p "../saves/eval/${lora_model}"


echo ----------------------------merge lora weight----------------------------
python ../src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path "../saves/${lora_model}/checkpoint-${checkpoint}" \
    --template cgec \
    --finetuning_type lora \
    --export_dir "../saves/${lora_model}/checkpoint-${checkpoint}/full-model" \
    --export_size 2 \
    --export_legacy_format False


echo "----------------------------using vllm to infer fcgec dev----------------------------"
python ../utils/vllm_chat_template.py \
        --data_dir "../data/FCGEC_valid_extract.txt" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-fcgec_dev_vllm.txt" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo "----------------------------using vllm to infer fcgec test----------------------------"
python ../utils/fcgec_test_vllm.py \
        --data_dir "../data/FCGEC_test.json" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-fcgec_test_vllm.json" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"


echo "----------------------------using vllm to infer nacgec test----------------------------"
python ../utils/vllm_chat_template.py \
        --data_dir "../data/NACGEC_test_extract.txt" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-nacgec_test_vllm.txt" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo "----------------------------eval fcgec dev----------------------------"
python ../utils/ChERRANT/parallel_to_m2.py \
        -f "../saves/eval/${lora_model}/checkpoint-${checkpoint}-fcgec_dev_vllm.txt" \
        -o "../saves/eval/${lora_model}/checkpoint-${checkpoint}-FCGEC_valid_10_beam_vllm.m2"


nohup python -u ../utils/ChERRANT/compare_m2_for_evaluation.py \
        -hyp "../saves/eval/${lora_model}/checkpoint-${checkpoint}-FCGEC_valid_10_beam_vllm.m2" \
        -ref "../data/FCGEC_valid.m2" \
        >"../saves/eval/${lora_model}/checkpoint-${checkpoint}-FCGEC_valid.log" 2>&1 &


echo "----------------------------eval nacgec test----------------------------"
python ../utils/ChERRANT/parallel_to_m2.py \
        -f "../saves/eval/${lora_model}/checkpoint-${checkpoint}-nacgec_test_vllm.txt" \
        -o "../saves/eval/${lora_model}/checkpoint-${checkpoint}-NACGEC_test_10_beam_vllm.m2"

nohup python -u ../utils/ChERRANT/compare_m2_for_evaluation.py \
        -hyp "../saves/eval/${lora_model}/checkpoint-${checkpoint}-NACGEC_test_10_beam_vllm.m2" \
        -ref "../data/NACGEC_test.m2" \
        >"../saves/eval/${lora_model}/checkpoint-${checkpoint}-NACGEC_test.log" 2>&1 &

