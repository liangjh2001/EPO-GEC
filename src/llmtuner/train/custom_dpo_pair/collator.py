from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class CustomDPOPairDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]

            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        edit_masks = []
        for key in ("chosen", "rejected"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[f"{key}_ids"])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[f"{key}_ids"],
                        "attention_mask": [1] * (prompt_len + answer_len)
                    }
                )
                edit_masks.append([0] * prompt_len + feature[f"{key}_edit_mask"])
                label_positions.append((prompt_len, answer_len))
        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)

        # Pad edit_mask to the same length as input_ids
        max_length = batch["input_ids"].shape[1]
        padded_edit_masks = []
        for edit_mask in edit_masks:
            edit_mask = edit_mask + [0] * (max_length - len(edit_mask))
            padded_edit_masks.append(edit_mask)

        batch["edit_mask"] = torch.tensor(padded_edit_masks, dtype=torch.long)
        return batch
