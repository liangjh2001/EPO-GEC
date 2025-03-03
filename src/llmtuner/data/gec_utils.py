import sys
import os
sys.path.append('../utils/ChERRANT/modules')  # Add the ChERRANT modules to the path
sys.path.append('../utils/ChERRANT')

from modules.alignment import read_cilin, read_confusion, AlignmentNoPos
from modules.merger import Merger
from pypinyin import lazy_pinyin


def generate_gec_edit_masks(rejected_ids, chosen_ids, tokenizer, annotator):
    rejected_with_spaces = " ".join(tokenizer.convert_ids_to_tokens(rejected_ids))
    chosen_with_spaces = " ".join(tokenizer.convert_ids_to_tokens(chosen_ids))

    orig = annotator.parse(rejected_with_spaces, tokenise=False)
    cor = annotator.parse(chosen_with_spaces, tokenise=False)
    edits = annotator.annotate(orig, cor, merging='all-merge')

    rejected_edit_mask = [0] * len(rejected_ids)
    chosen_edit_mask = [0] * len(chosen_ids)

    for e in edits:
        if e.o_start >= len(rejected_ids) or e.c_start >= len(chosen_ids):
            continue
        rejected_edit_mask[e.o_start] = 1  # pivot token
        chosen_edit_mask[e.c_start] = 1  # pivot token
        for i in range(e.o_start+1, e.o_end):
            if i < len(rejected_ids):
                rejected_edit_mask[i] = 2
        for i in range(e.c_start+1, e.c_end):
            if i < len(chosen_ids):
                chosen_edit_mask[i] = 2

    return rejected_edit_mask, chosen_edit_mask


def create_cherrant_obj():
    semantic_dict, semantic_class = read_cilin()
    confusion_dict = read_confusion()
    alignment = AlignmentNoPos(semantic_dict, confusion_dict)
    m = Merger(granularity='char')

    return alignment, m


def filter_edits(edits):
    filtered_edits = []

    for edit in edits:
        start_position1 = edit[1]
        start_position2 = edit[3]

        # Flag to determine if the current edit should be added
        should_add = True

        for i, filtered_edit in enumerate(filtered_edits):
            if filtered_edit[1] == start_position1:
                if filtered_edit[3] > start_position2:
                    filtered_edits[i] = edit  # Replace the existing edit
                should_add = False
                break
            elif filtered_edit[3] == start_position2:
                if filtered_edit[1] > start_position1:
                    filtered_edits[i] = edit  # Replace the existing edit
                should_add = False
                break

        if should_add:
            filtered_edits.append(edit)

    return filtered_edits


def generate_cgec_edit_masks(rejected_ids, chosen_ids, tokenizer, alignment, m):

    if 'Qwen2' in tokenizer.name_or_path:
        rejected_converted_sentence1 = []
        chosen_converted_sentence2 = []
        for token_id in rejected_ids:
            rejected_converted_sentence1.append(tokenizer.decode(token_id))
        for token_id in chosen_ids:
            chosen_converted_sentence2.append(tokenizer.decode(token_id))
    else:
        rejected_converted_sentence1 = tokenizer.convert_ids_to_tokens(rejected_ids)
        chosen_converted_sentence2 = tokenizer.convert_ids_to_tokens(chosen_ids)

    result = []
    for converted_sentence in [rejected_converted_sentence1, chosen_converted_sentence2]:
        pinyin = [lazy_pinyin(word) for word in converted_sentence]
        result.append(list(zip(converted_sentence, pinyin)))

    align_objs = alignment(result[0], result[1], verbose=False)

    align_obj = align_objs[0]
    edits = m(align_obj, result[0], result[1], verbose=False)

    rejected_edit_mask = [0] * len(rejected_ids)
    chosen_edit_mask = [0] * len(chosen_ids)

    for e in edits:
        if e[1] == len(rejected_ids) or e[3] == len(chosen_ids):
            continue
        for i in range(e[1], e[1] + 1):
            rejected_edit_mask[i] = 1
        for i in range(e[3], e[3] + 1):
            chosen_edit_mask[i] = 1

    if sum(rejected_edit_mask) != sum(chosen_edit_mask):
        edits = filter_edits(edits)
        rejected_edit_mask = [0] * len(rejected_ids)
        chosen_edit_mask = [0] * len(chosen_ids)
        for e in edits:
            if e[1] == len(rejected_ids) or e[3] == len(chosen_ids):
                continue
            for i in range(e[1], e[1] + 1):
                rejected_edit_mask[i] = 1
            for i in range(e[3], e[3] + 1):
                chosen_edit_mask[i] = 1

    return rejected_edit_mask, chosen_edit_mask
