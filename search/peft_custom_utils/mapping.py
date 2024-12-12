from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
from .config import PeftConfig
from .peft_model import (PeftModel, PeftModelForCausalLM)
from .tuners import (LoraConfig, LoraModel)
from .tuners.tuners_utils import BaseTuner
from .utils import _prepare_prompt_learning_config


if TYPE_CHECKING:
    from transformers import PreTrainedModel


MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[PeftModel]] = {"CAUSAL_LM": PeftModelForCausalLM}
PEFT_TYPE_TO_CONFIG_MAPPING: dict[str, type[PeftConfig]] = {"LORA": LoraConfig}
PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[BaseTuner]] = {"LORA": LoraModel}

def get_peft_config(config_dict: dict[str, Any]) -> PeftConfig:
    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", mixed: bool = False, autocast_adapter_dtype: bool = True,
                   revision: Optional[str] = None, low_cpu_mem_usage: bool = False, rank:int = 0, search_space: list = []):
    model_config = BaseTuner.get_model_config(model)
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model = model, peft_config = peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype, rank=rank, search_space = search_space)


def inject_adapter_in_model(peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default", low_cpu_mem_usage: bool = False):
    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
    return peft_model.model