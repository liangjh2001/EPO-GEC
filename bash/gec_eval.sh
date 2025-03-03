#!/bin/bash
export CUDA_VISIBLE_DEVICES=3


lora_model="mistral-epo"
base_model="your_path_to_sft_stage2_full_model"
checkpoint="1"

mkdir -p "../saves/eval/${lora_model}"


echo ----------------------------merge lora weight----------------------------
python ../src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path "../saves/${lora_model}/checkpoint-${checkpoint}" \
    --template gec \
    --finetuning_type lora \
    --export_dir "../saves/${lora_model}/checkpoint-${checkpoint}/full-model" \
    --export_size 2 \
    --export_legacy_format False


echo ----------------------------using vllm to infer bea dev----------------------------
python ../utils/vllm_chat_template.py \
        --data_dir "../data/bea_dev_source.txt" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_dev_vllm.txt" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo ----------------------------using vllm to infer bea test----------------------------
python ../utils/vllm_chat_template.py \
        --data_dir "../data/ABCN.test.bea19.orig" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_test_vllm.txt" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"


echo ----------------------------using vllm to infer conll test----------------------------
python ../utils/vllm_chat_template.py \
        --data_dir "../data/conll_14_test_source.txt" \
        --output_dir "../saves/eval/${lora_model}/checkpoint-${checkpoint}-conll_test_vllm.txt" \
        --model_name  "../saves/${lora_model}/checkpoint-${checkpoint}/full-model"



echo ----------------------------eval bea dev----------------------------
errant_parallel -orig "../data/bea_dev_source.txt" \
                -cor "../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_dev_vllm.txt" \
                -out "../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_dev_vllm.m2"


nohup errant_compare -hyp "../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_dev_vllm.m2"\
                      -ref "../data/ABCN.dev.gold.bea19.m2" \
                      >"../saves/eval/${lora_model}/checkpoint-${checkpoint}-bea_dev.log" 2>&1 &

echo ----------------------------eval conll test----------------------------
nohup python -u ../utils/m2scorer/scripts/m2scorer.py \
         "../saves/eval/${lora_model}/checkpoint-${checkpoint}-conll_test_vllm.txt" \
          "../data/conll14st-test-data-noalt-official-2014.combined.m2" \
        >"../saves/eval/${lora_model}/checkpoint-${checkpoint}-conll_test_no_alt.log" 2>&1 &

