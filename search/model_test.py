# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import dataclasses
import os
import random
from collections import Counter
from warnings import warn
import time
import fire
import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available

from configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
from data.concatenator import ConcatDataset
from policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)

from utils.fsdp_utils import hsdp_device_mesh
from utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    freeze_LLM_only,
    get_policies,
    print_model_size,
    print_frozen_model_status,
    setup,
    setup_environ_flags,
    train,
)
from peft import PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    MllamaForConditionalGeneration,
)


from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def main(**kwargs):
    
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        
        model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
        
        processor = AutoProcessor.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        processor.tokenizer.padding_side = "right"
        model.supports_gradient_checkpointing = True
        model.language_model.supports_gradient_checkpointing = True

    elif config.model_type == "llama":
        is_vision = False
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama or mllama model."
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name
        if train_config.tokenizer_name is None
        else train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    
    if (
        train_config.enable_fsdp
        and fsdp_config.pure_bf16
        and not train_config.quantization
    ):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
            
        else:
            from peft_utils import get_peft_model
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
            model.merge_and_unload()
            print("model", model)
            exit()
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()
        trainable_params, all_param = model.get_nb_trainable_parameters()


    hsdp_device_mesh_plan = None
    if (
        fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")

    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)
            
        if not train_config.use_peft and train_config.freeze_LLM_only and config.model_type == "mllama":
            freeze_LLM_only(model)
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)
        
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        if is_vision:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(
                model,
                [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                ],
            )
        else:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        
        if train_config.freeze_LLM_only:
            use_orig_params = True
        else:
            use_orig_params = False
        model = FSDP(
            model,
            auto_wrap_policy=(
                my_auto_wrapping_policy if train_config.use_peft else wrapping_policy
            ),
            cpu_offload=(
                CPUOffload(offload_params=True)
                if fsdp_config.fsdp_cpu_offload
                else None
            ),
            mixed_precision=(
                mixed_precision_policy if not fsdp_config.pure_bf16 else None
            ),
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if train_config.low_cpu_fsdp and rank != 0
                else None
            ),
            use_orig_params=use_orig_params,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")
    


if __name__ == "__main__":

    start_time = time.perf_counter()
    results, train_config, peft_config = fire.Fire(main)
    end_time = time.perf_counter()


