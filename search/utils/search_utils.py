
import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import math 
import contextlib


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from policies import AnyPrecisionAdamW

import sys
sys.path.append("..")
from model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint
from policies import fpSixteen,bfSixteen, get_llama_wrapper
from utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from utils.flop_utils import FlopMeasure

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

def update_weights(model, batch, train_config, local_rank,
                   autocast, gradient_accumulation_steps, scaler,
                   step, dataloader, lora_optimizer, pbar):
    
    for key in batch.keys():
        if train_config.enable_fsdp:
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
        loss = model(**batch).loss
    total_loss += loss.detach().float()
    loss = loss / gradient_accumulation_steps
    
    if train_config.use_fp16:
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                scaler.unscale_(lora_optimizer)
                if train_config.enable_fsdp:
                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
            scaler.step(lora_optimizer)
            scaler.update()
            lora_optimizer.zero_grad()
            
    else:
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                if train_config.enable_fsdp:
                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
            lora_optimizer.step()
            lora_optimizer.zero_grad()
            
    pbar.update(1)
    



def search(model, train_dataloader, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
        
    alpha_params, lora_weight_params = [], []
    
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        
        if 'lora' in name:
            if param.requires_grad:
                lora_weight_params += [param]
    
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        lora_optimizer = AnyPrecisionAdamW(lora_weight_params,
                                      lr=train_config.lr,
                                      momentum_dtype=torch.bfloat16,
                                      variance_dtype=torch.bfloat16,
                                      use_kahan_summation=False,
                                      weight_decay=train_config.weight_decay)
        
        alpha_optimizer = AnyPrecisionAdamW(alpha_params,
                                      lr=train_config.lr,
                                      momentum_dtype=torch.bfloat16,
                                      variance_dtype=torch.bfloat16,
                                      use_kahan_summation=False,
                                      weight_decay=train_config.weight_decay)
        
    else:
        alpha_optimizer = optim.AdamW(alpha_params,
                                lr=train_config.lr,
                                weight_decay=train_config.weight_decay)
        
        lora_optimizer = optim.AdamW(lora_weight_params,
                                lr=train_config.lr,
                                weight_decay=train_config.weight_decay)
    
    lr_scheduler = StepLR(lora_optimizer, step_size=1, gamma=train_config.gamma)
    
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    for epoch in range(train_config.num_search_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            
            for step, batch in enumerate(train_dataloader):
                
                split_index = int(len(batch) * 0.75)

                mini_batch_1 = batch[0:split_index] 
                mini_batch_2 = batch[split_index:]

                update_weights(model, mini_batch_1, train_config, local_rank,
                   autocast, gradient_accumulation_steps, scaler,
                   step, train_dataloader, lora_optimizer, pbar)
                
                update_weights(model, mini_batch_2, train_config, local_rank,
                   autocast, gradient_accumulation_steps, scaler,
                   step, train_dataloader, alpha_optimizer, pbar)
                
                pbar.update(1)
                
                # for key in batch.keys():
                #     if train_config.enable_fsdp:
                #         if is_xpu_available():
                #             batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                #         else:
                #             batch[key] = batch[key].to(local_rank)
                #     else:
                #         if is_xpu_available():
                #             batch[key] = batch[key].to('xpu:0')
                #         elif torch.cuda.is_available():
                #             batch[key] = batch[key].to('cuda:0')
                
                # with autocast():
                #     loss = model(**batch).loss
                # total_loss += loss.detach().float()
                # loss = loss / gradient_accumulation_steps
                
                # if train_config.use_fp16:
                #     scaler.scale(loss).backward()
                #     if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                #         if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                #             scaler.unscale_(lora_optimizer)
                #             if train_config.enable_fsdp:
                #                 model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                #             else:
                #                 torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                #         scaler.step(lora_optimizer)
                #         scaler.update()
                #         lora_optimizer.zero_grad()
                #         pbar.update(1)
                # else:
                #     loss.backward()
                #     if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                #         if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                #             if train_config.enable_fsdp:
                #                 model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                #             else:
                #                 torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                #         lora_optimizer.step()
                #         lora_optimizer.zero_grad()
                #         pbar.update(1)
            pbar.close()

        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        lr_scheduler.step()
        
