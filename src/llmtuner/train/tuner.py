import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model_and_tokenizer
from .dpo import run_dpo
from .sft import run_sft
from .custom_dpo_pair import run_custom_dpo_pair


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks


    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "custom_dpo_pair":
        run_custom_dpo_pair(model_args, data_args, training_args, finetuning_args)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    get_template_and_fix_tokenizer(tokenizer, data_args.template)

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is None:  # cannot convert dtype of a quantized model
        output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        setattr(model.config, "torch_dtype", output_dtype)
        for param in model.parameters():
            param.data = param.data.to(output_dtype)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if "chatglm3" in model_args.model_name_or_path:
            with open(model_args.export_dir + '/tokenizer_config.json', 'r', encoding='utf-8') as file:
                config = json.load(file)
            del config['eos_token']
            del config['model_max_length']
            del config['pad_token']
            del config['unk_token']
            with open(model_args.export_dir + '/tokenizer_config.json', "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
