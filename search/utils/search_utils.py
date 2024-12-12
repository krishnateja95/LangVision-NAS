import torch
import torch.nn as nn
import torch.nn.functional as F


def get_searched_network(model, peft_config):
    pass


def get_named_parameters(model):
    key_list = [key for key, _ in model.named_modules()]

    params_dict = {name: param for name, param in model.named_parameters()}

    for name, param in params_dict.items():
        print(f"Name: {name}")
    
    return 


def slice_lora_weights(layer, ):
    
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
    

