
import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import math 
import contextlib
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json

import sys
sys.path.append("..")
from model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint
from policies import fpSixteen,bfSixteen, get_llama_wrapper
from utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from utils.flop_utils import FlopMeasure

from utils.config_utils import generate_dataset_config, get_dataloader_kwargs
from utils.dataset_utils import get_custom_data_collator, get_preprocessed_dataset
from data.concatenator import ConcatDataset


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None

def search(model, 
           gradient_accumulation_steps,
           train_config,
           fsdp_config = None,
           is_vision = True,
           dataset_processer = None,
           local_rank = None,
           rank = None,
           max_steps = None,
           kwargs = None):
    
    dataset_config    = generate_dataset_config(train_config, kwargs)
    dataset_train     = get_preprocessed_dataset(dataset_processer, dataset_config, split="train")
    
    if train_config.batching_strategy == "packing":
        if is_vision:
            raise ValueError("Packing is not supported for vision datasets")
        else:
            dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_config.batch_size_training = train_config.search_batch_size_training
    train_dl_kwargs      = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    
    if custom_data_collator:
        train_dl_kwargs["collate_fn"] = custom_data_collator
    
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   num_workers=train_config.num_workers_dataloader,
                                                   pin_memory=True,
                                                   **train_dl_kwargs)


    alpha_params, lora_weight_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        
        if 'lora' in name:
            if param.requires_grad:
                lora_weight_params += [param]
        
    lora_optimizer = optim.AdamW(lora_weight_params,
                            lr=train_config.lr,
                            weight_decay=train_config.weight_decay)

    alpha_optimizer = optim.AdamW(alpha_params,
                                lr=train_config.lr,
                                weight_decay=train_config.weight_decay)
    
    lr_scheduler_lora  = StepLR(lora_optimizer, step_size=1, gamma=train_config.gamma)
    lr_scheduler_alpha = StepLR(alpha_optimizer, step_size=1, gamma=train_config.gamma)

    enable_fsdp = False
    if train_config.use_fp16 and enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    total_train_steps = 0

    for epoch in range(train_config.num_search_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        
        with MemoryTrace() as memtrace:
            model.train()
            
            total_loss_lora = 0.0
            total_loss_alpha = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):

                    if total_train_steps > max_steps:
                        return  
                    
                    total_train_steps += 1
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        if not enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    
                    for key in batch.keys():
                        if enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')
                    


                    with autocast():
                        loss_lora = model(**batch).loss
                    total_loss_lora += loss_lora.detach().float()
                    loss_lora = loss_lora / gradient_accumulation_steps
                    
                    if train_config.use_fp16:
                        scaler.scale(loss_lora).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(lora_optimizer)
                                if enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(lora_weight_params, train_config.gradient_clipping_threshold)
                            scaler.step(lora_optimizer)
                            scaler.update()
                            lora_optimizer.zero_grad()


                    with autocast():
                        alpha_loss = model(**batch).loss
                    total_loss_alpha += alpha_loss.detach().float()
                    alpha_loss = alpha_loss / gradient_accumulation_steps
                    
                    if train_config.use_fp16:
                        scaler.scale(alpha_loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(alpha_optimizer)
                                if enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(alpha_params, train_config.gradient_clipping_threshold)
                            scaler.step(alpha_optimizer)
                            scaler.update()
                            alpha_optimizer.zero_grad()
                            pbar.update(1)

                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss_lora.detach().float()})")
                pbar.close()

        if is_xpu_available() and (torch.xpu.device_count() > 1 and enable_fsdp):
            dist.all_reduce(total_loss_lora, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_alpha, op=dist.ReduceOp.SUM)

        elif torch.cuda.device_count() > 1 and enable_fsdp:
            dist.all_reduce(total_loss_lora, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_alpha, op=dist.ReduceOp.SUM)
        
        search_epoch_loss = total_loss_lora / len(train_dataloader)
        
        lr_scheduler_lora.step()
        lr_scheduler_alpha.step() 
        