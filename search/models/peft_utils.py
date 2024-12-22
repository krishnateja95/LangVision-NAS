from __future__ import annotations

import math
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.utils.other import transpose

class Lora_Layer(nn.Module):
    def __init__(self, base_layer, peft_config, base_layer_name):
        super(Lora_Layer, self).__init__()

        self.base_layer = base_layer
    
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.peft_config = peft_config
        self.base_layer_name = base_layer_name

        if not base_layer_name in self.peft_config.target_modules:
            self.lora_adapters = False

        else:
            self.lora_adapters = True

            self.lora_alpha = self.peft_config.lora_alpha
            self.scaling = {}
            
            self.lora_dropout = self.peft_config.lora_dropout
            
            self.init_lora_weights = self.peft_config.init_lora_weights
            self.use_rslora = self.peft_config.use_rslora
            self.use_dora = self.peft_config.use_dora 
            
            self.merged = False
            self.merged_adapters = []
            
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.fan_in_fan_out = False
            self.lora_bias = peft_config.lora_bias

            if peft_config.supernet:
                self.if_supernet = True
                self.alpha_params = nn.Parameter(torch.rand(len(self.peft_config.search_space)), requires_grad=True).to(device = self.base_layer.weight.device, dtype = self.base_layer.weight.dtype)
                self.rank = max(self.peft_config.search_space)
            else:
                self.rank = self.peft_config.r

            self.lora_A = nn.Linear(self.in_features, self.rank, bias=False).to(device = self.base_layer.weight.device, dtype = self.base_layer.weight.dtype)
            self.lora_B = nn.Linear(self.rank, self.out_features, bias=self.lora_bias).to(device = self.base_layer.weight.device, dtype = self.base_layer.weight.dtype)

            if self.use_rslora:
                self.scaling = self.lora_alpha / math.sqrt(self.rank)
            else:
                self.scaling = self.lora_alpha / self.rank

            self.lora_A_bias = False
            self.lora_B_bias = self.lora_bias

            if self.lora_dropout > 0.0:
                self.lora_dropout_layer = nn.Dropout(p=self.lora_dropout)
            else:
                self.lora_dropout_layer = nn.Identity()
            
            self.reset_lora_parameters(peft_config.init_lora_weights)

    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
               
        if not self.merged and self.lora_adapters:
            if self.if_supernet:
                alphas = F.softmax(self.alpha_params, dim=0) 
            
                super_weight_A = torch.zeros_like(self.lora_A.weight)
                super_weight_B = torch.zeros_like(self.lora_B.weight)

                for i, (neurons, alpha_param) in enumerate(zip(self.search_space, alphas)):
                    start_idx = (self.max_r - neurons) // 2
                    end_idx = start_idx + neurons

                    partial_weight_A = self.lora_A.weight.clone()
                    partial_weight_A[:start_idx, :] = 0
                    partial_weight_A[end_idx:, :] = 0
                    partial_weight_A[start_idx:end_idx, :] *= alpha_param
                    super_weight_A += partial_weight_A

                    partial_weight_B = self.lora_B.weight.clone()
                    partial_weight_B[:, :start_idx] = 0
                    partial_weight_B[:, end_idx:] = 0
                    partial_weight_B[:, start_idx:end_idx] *= alpha_param
                    super_weight_B += partial_weight_B

                result = result + F.linear(F.linear(x, super_weight_A, self.lora_A_bias), super_weight_B, self.lora_B_bias)
            
            else:
                # x = x.to(self.lora_A.weight.dtype)
                result = result + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
        result = result.to(torch_result_dtype)

        return result

    def slice_lora_weights(self):
        
        best_alpha_index = torch.argmax(self.alpha_params)
        sampled_rank = self.search_space[best_alpha_index]

        start_idx = (self.max_r - sampled_rank) // 2
        end_idx = start_idx + sampled_rank

        sampled_A =  self.lora_A.weight.clone()[start_idx:end_idx, :]
        sampled_B =  self.lora_B.weight.clone()[:, start_idx:end_idx]
        
        self.lora_A.weight.data = sampled_A
        self.lora_B.weight.data = sampled_B 

        self.lora_A.out_features = sampled_rank
        self.lora_B.in_features = sampled_rank
    
        self.if_supernet = False
        self.merged = True
    
    def reset_lora_parameters(self, init_lora_weights):
        if init_lora_weights is False:
            return
        
        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.rank)
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights}")
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_bias:
            nn.init.zeros_(self.lora_B.bias)
   
    def merge_adapters(self):

        if not self.base_layer_name in self.peft_config.target_modules:
            self.lora_adapters = False

        else:
            weight_A = self.lora_A.weight
            weight_B = self.lora_B.weight

            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling

            new_weight = self.base_layer.weight.data + output_tensor
            self.base_layer.weight.data = new_weight

            if self.lora_bias:
                new_bias = self.base_layer.bias + self.lora_B.bias
                self.base_layer.bias.data = new_bias

            del self.lora_A
            del self.lora_B
            del self.lora_dropout_layer

            self.merged = True
            self.lora_adapters = False
            self.if_supernet = False


        

