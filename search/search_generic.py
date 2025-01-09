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
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)

from data.concatenator import ConcatDataset
from policies import AnyPrecisionAdamW


from utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)

from utils.search_utils import search
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

from utils.flop_utils import get_nb_trainable_parameters

from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)
 

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)

from transformers import MllamaForConditionalGeneration

def main(**kwargs):
    
    train_config = TRAIN_CONFIG()
    update_config((train_config), **kwargs)
    
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # if train_config.enable_fsdp:
    #     setup()
    #     local_rank = int(os.environ["LOCAL_RANK"])
    #     world_size = int(os.environ["WORLD_SIZE"])

    rank = int(os.environ["RANK"])

    # if torch.distributed.is_initialized():
    #     if is_xpu_available():
    #         torch.xpu.set_device(local_rank)
    #     elif torch.cuda.is_available():
    #         torch.cuda.set_device(local_rank)
    #     clear_gpu_cache(local_rank)
    #     setup_environ_flags(rank)

    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    use_cache = None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map="auto",
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
            device_map="auto",
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

    print_model_size(model, train_config, 0)
    
    if train_config.use_peft:
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
            
        else:

            from peft_custom_utils import get_peft_model
            peft_config = generate_peft_config(train_config, kwargs)
            
            peft_config.lora_bias = None
            peft_config.search_space = [8, 16, 32, 64]
            peft_config.supernet = True

            model = get_peft_model(model, peft_config, search_space = peft_config.search_space)

            for name, param in model.named_parameters():
                if rank == 0:
                    print(name)

            exit()

            search(
                model,
                train_dataloader,
                train_config.gradient_accumulation_steps,
                train_config,
                None,
                None,
                None,
                wandb_run,
            )

        exit()

        trainable_params, all_param, trainable_params_percent = get_nb_trainable_parameters(model)

        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

    if not train_config.quantization:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    if is_vision:
        dataset_processer = processor
    else:
        dataset_processer = tokenizer
    
    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
    )
    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="test",
    )
    if rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
    print("length of dataset_train", len(dataset_train))
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)})"
            )
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=train_config.lr,
                            weight_decay=train_config.weight_decay
                            )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    alpha_params, lora_weight_params = [], []
    
    if rank == 0:
        print(model)

    exit()

    for name, param in model.named_parameters():
        
        if rank == 0:
            print(name)

        if 'alpha' in name:
            alpha_params += [param]
        
        if 'lora' in name:
            if param.requires_grad:
                lora_weight_params += [param]

    print("alpha_params", alpha_params)
    # print("lora_weight_params", lora_weight_params)
    exit()

    search(
        model,
        train_dataloader,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run,
    )

    for name, module in model.named_modules():
        if name.split('.')[-1] in peft_config.target_modules:
            if hasattr(module, "get_sampled_network") and callable(getattr(module, "get_sampled_network")):
                module.get_sampled_network()
    
    exit()

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run,
    )

    results["trainable_params"] = trainable_params
    results["all_param"] = all_param

    if rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

    model.get_sampled_network(peft_config)
    model.save_pretrained(train_config.finetune_model_dir)

    exit()

    del model

    model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map="auto",
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(model, train_config.output_dir)
    merged_model = lora_model.merge_and_unload()
    
    # merged_model.push_to_hub(train_config.HF_repo)

    return results, train_config, peft_config


if __name__ == "__main__":

    start_time = time.perf_counter()
    results, train_config, peft_config = fire.Fire(main)
    end_time = time.perf_counter()

    total_time = start_time - end_time
    rank = int(os.environ["RANK"]) 
    if rank == 0:
        print(results)
        
        import csv
        list_1 = ["Hardware", "Num of Hardware", "Model", "Trainable", "All params", "Dataset", "target_modules", "LoRA Rank", "Avg Epoch Time", "Eval PPL"]
        list_2 = ["Nvidia A100 GPU", 4, train_config.model_name, results["trainable_params"], results["all_param"], "ocrvqa", peft_config.target_modules, peft_config.r, results["avg_epoch_time"], results["best_eval"]] 
        assert len(list_1) == len(list_2)

        csv_file = "LoRA_Bench.csv"
        file_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(list_1)
            
            writer.writerow(list_2) 
            
        csvfile.close()





