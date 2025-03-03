import argparse
import json
import os

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main(args):
    model_dir = args.model_name
    is_json_file = args.data_dir.lower().endswith('.json')
    is_nlpcc = "nlpcc2018" in args.data_dir or "bea" in args.data_dir or "conll" in args.data_dir
    if not is_json_file:
        with open(args.data_dir, 'r', encoding='utf-8') as f:
            original_data_list = f.readlines()
    else:
        with open(args.data_dir, "r", encoding="utf-8") as file:
            original_data_list = json.load(file)

    print(f"test mode: only use 5 data")
    original_data_list = original_data_list[:5]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    questions = []
    for line in original_data_list:
        if not is_json_file:
            if args.is_align_model:
                original_prompt = line.split('\t')[1].strip() + tokenizer.eos_token + line.split('\t')[2].strip()
            else:
                if not is_nlpcc:
                    original_prompt = line.split('\t')[1].strip()
                else:
                    original_prompt = line.strip()
        else:
            original_prompt = line.get("source")
        ins = [{"role": "user", "content": original_prompt}]
        chat_ins = tokenizer.apply_chat_template(ins, tokenize=False)
        questions.append(chat_ins)
    print(f"length{len(questions)}")
    print(f"example{questions[0]}")

    tokens = tokenizer(questions, add_special_tokens=False).input_ids
    eot_tokens_ = [tokenizer.eos_token_id]

    if is_json_file:
        max_tokens = 512
        sampling_params = SamplingParams(max_tokens=max_tokens, stop_token_ids=eot_tokens_, temperature=0)
    else:
        data_base_name = os.path.basename(args.data_dir)
        if data_base_name == 'conll_14_test_source.txt':
            max_tokens= 400
        else:
            max_tokens = 200
        sampling_params = SamplingParams(best_of=10, use_beam_search=True,
                                         max_tokens=max_tokens, stop_token_ids=eot_tokens_,
                                         temperature=0, stop=None if not args.is_aux_model else '###')
    print('sampleing =====', sampling_params)
    llm = LLM(model=model_dir, tensor_parallel_size=torch.cuda.device_count(), swap_space=12,gpu_memory_utilization=0.9,
              dtype="auto", trust_remote_code=True, max_model_len=4096, enable_lora=True if args.adapter else False)
    res_completions = []
    # completions = llm.generate(prompts=questions, sampling_params=sampling_params)
    completions = llm.generate(prompt_token_ids=tokens, sampling_params=sampling_params,
                               lora_request=LoRARequest("lora_adapter", 1, args.adapter) if args.adapter else None)
    length_limit = []
    stop_string_limit = 0
    stop_eos_limit = 0
    stop_eos_limit_list = []
    for (idx, line), output in zip(enumerate(original_data_list), completions):
        generated_text = output.outputs[0].text
        if output.outputs[0].finish_reason == 'length':
            if not is_nlpcc:
                generated_text = line.split('\t')[1].strip() if not is_json_file else line.get("source").strip()
            else:
                generated_text = line.strip()
            length_limit.append(idx + 1)
        elif output.outputs[0].finish_reason == 'stop':
            if output.outputs[0].stop_reason == '###':
                stop_string_limit += 1
            elif output.outputs[0].stop_reason == tokenizer.eos_token_id:
                stop_eos_limit += 1
                stop_eos_limit_list.append(idx + 1)
        res_completions.append(generated_text)
    print(f'length count {len(length_limit)} \n {length_limit}')
    print(f'stop string count {stop_string_limit}')
    print(f'stop eos count {stop_eos_limit}')

    if not is_json_file:
        error_dir = args.output_dir.replace('.txt', '_error.txt')
    else:
        error_dir = args.output_dir.replace('.json', '_error.txt')
    with open(error_dir, 'w', encoding='utf-8') as file1:
        file1.write(f'length count  {len(length_limit)}\n length count idx {length_limit} \n')
        file1.write(f'stop string count {stop_string_limit} \n')
        file1.write(f'stop eos count {stop_eos_limit} \n')
        file1.write(f'stop_eos_limit_list={stop_eos_limit_list}')

    assert len(original_data_list) == len(res_completions)
    if not is_json_file:
        with open(args.output_dir, 'w', encoding='utf-8') as file:
            for line, (idx, policy_output) in zip(original_data_list, enumerate(res_completions)):
                if args.is_align_model:
                    line = line.split('\t')[0].strip() + '\t' + line.split('\t')[1].strip()
                else:
                    line = line.strip()
                policy_output = policy_output.strip()
                if not is_nlpcc:
                    file.write(f"{line}\t{policy_output}\n")
                else:
                    file.write(f"{policy_output}\n")
    else:
        for line, policy_output in zip(original_data_list, res_completions):
            line['predict'] = policy_output.strip()
        with open(args.output_dir, 'w', encoding='utf-8') as output_file:
            json.dump(original_data_list, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str,
                        default="your_data_dir")
    parser.add_argument("--output_dir", "-s", type=str,
                        default="your_output_dir")
    parser.add_argument("--model_name", "-n", type=str,
                        default="your_model_name")
    parser.add_argument("--adapter", "-a", type=str,
                        default=None)
    parser.add_argument("--is_align_model", action='store_true')
    parser.add_argument("--is_aux_model", action='store_true')
    parser.add_argument("--is_reverse_model", action='store_true')
    args = parser.parse_args()
    main(args)
