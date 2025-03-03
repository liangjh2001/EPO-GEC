import argparse
import json
from multiprocessing import Pool

import errant
import spacy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def process_item(data):
    nlp = spacy.load('en_core_web_sm')
    annotator = errant.load('en', nlp)
    source = data['source']
    target = data['target']
    predict = data['predict']
    distinct = False

    errant_edit_distances = []
    tokens_ids2 = tokenizer(target, add_special_tokens=False).input_ids
    sentence_with_spaces2 = " ".join(tokenizer.convert_ids_to_tokens(tokens_ids2))
    cor = annotator.parse(sentence_with_spaces2, tokenise=False)

    for p in predict:
        tokens_ids1 = tokenizer(p, add_special_tokens=False).input_ids
        sentence_with_spaces1 = " ".join(tokenizer.convert_ids_to_tokens(tokens_ids1))
        orig = annotator.parse(sentence_with_spaces1, tokenise=False)
        edits = annotator.annotate(orig, cor, merging='rules')
        errant_edit_dis = len(edits)
        errant_edit_distances.append(errant_edit_dis)

    min_distance_index = errant_edit_distances.index(min(errant_edit_distances))
    max_distance_index = errant_edit_distances.index(max(errant_edit_distances))

    if max(errant_edit_distances) - min(errant_edit_distances) > 0 and predict[min_distance_index] != predict[max_distance_index]:
        distinct = True

    entry = {
        'source': source,
        'target': target,
        'pos_neg_label': [predict[min_distance_index], predict[max_distance_index], target],
        'pos_neg': [predict[min_distance_index], predict[max_distance_index]],
        'label_neg': [target, predict[max_distance_index]],
        'distinct': distinct,
        'max_edit_dis': max(errant_edit_distances),
        'pos_neg_edit_dis': max(errant_edit_distances) - min(errant_edit_distances),
        'predict': predict
    }

    return entry, distinct, predict[min_distance_index] != predict[max_distance_index], predict[min_distance_index] != predict[max_distance_index] and min(errant_edit_distances) == max(errant_edit_distances),predict[max_distance_index]!=target


def process_data(data_list):
    num_processes = 1
    output_data_list = []
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with Pool(processes=num_processes) as pool:
        for entry, distinct, cond2, cond1,cond3 in tqdm(pool.imap_unordered(process_item, data_list), total=len(data_list)):
            output_data_list.append(entry)
            if distinct:
                count += 1
            if cond2:
                count2 += 1
            if cond1:
                count1 += 1
            if cond3:
                count3 += 1

    print(
        f"Difference between max and min edit distances greater than 0 && predictions corresponding to max and min edit distances are not the same: {count}")
    print(
        f"Difference between max and min edit distances equals 0 && predictions corresponding to max and min edit distances are not the same: {count1}")
    print(f"Predictions corresponding to max and min edit distances are not the same: {count2}")
    print(f"Prediction with max edit distance differs from label: {count3}")
    print(f"Length of input data: {len(data_list)}")
    print(f"Length of output data: {len(output_data_list)}")

    total_pos_neg_edit_dis = sum([entry['pos_neg_edit_dis'] for entry in output_data_list])
    print(f"Average pos_neg edit distance: {total_pos_neg_edit_dis / len(output_data_list)}")
    total_max_edit_dis = sum([entry['max_edit_dis'] for entry in output_data_list])
    print(f"Average max edit distance: {total_max_edit_dis / len(output_data_list)}")
    return output_data_list


def main(args):
    print(f"args: {args}")
    model_dir = args.model_name
    with open(args.data_dir, "r", encoding="utf-8") as file:
        original_data_list = json.load(file)

    print(f"test mode: only use 50 data")
    original_data_list = original_data_list[:50] # for test

    tokenizer_new = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    questions = []
    for line in original_data_list:
        original_prompt = line.get("source")
        ins = [{"role": "user", "content": original_prompt}]
        chat_ins = tokenizer_new.apply_chat_template(ins, tokenize=False)
        questions.append(chat_ins)
    print(f"length{len(questions)}")
    print(f"example{questions[0]}")

    tokens = tokenizer_new(questions, add_special_tokens=False).input_ids
    eot_tokens_ = [tokenizer_new.eos_token_id]

    max_tokens = 512
    sampling_params = SamplingParams(max_tokens=max_tokens, stop_token_ids=eot_tokens_, temperature=1.1, n=10)

    print('sampleing =====', sampling_params)
    llm = LLM(model=model_dir, tensor_parallel_size=torch.cuda.device_count(), swap_space=12,
              dtype="auto", trust_remote_code=True, max_model_len=4096, enable_lora=True if args.adapter else False)
    res_completions = []
    completions = llm.generate(prompt_token_ids=tokens, sampling_params=sampling_params,
                               lora_request=LoRARequest("lora_adapter", 1, args.adapter) if args.adapter else None)
    length_limit = []
    for (idx, line), completion in zip(enumerate(original_data_list), completions):
        temp_seqs = []
        for output in completion.outputs:
            generated_text = output.text.strip()
            if output.finish_reason == 'length':
                generated_text = line.get("source").strip()
                length_limit.append(idx + 1)
            temp_seqs.append(generated_text)
        res_completions.append(temp_seqs)
    print(f'length limit {len(length_limit)} \n {length_limit}')

    error_dir = args.output_dir.replace('.json', '_error.txt')
    with open(error_dir, 'w', encoding='utf-8') as file1:
        file1.write(f'length limit {len(length_limit)}\n idx {length_limit} \n')

    assert len(original_data_list) == len(res_completions)
    for line, policy_output in zip(original_data_list, res_completions):
        line['predict'] = policy_output
    output_data_list = process_data(original_data_list)
    with open(args.output_dir, 'w', encoding='utf-8') as output_file:
        json.dump(output_data_list, output_file, indent=4, ensure_ascii=False)


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    main(args)
