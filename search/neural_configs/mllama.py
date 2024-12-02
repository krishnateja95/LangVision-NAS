import csv

mllama_11B = {
    "vision_model": {"num_layers": 32, "Attention_heads": 16, "MLP_Intermedite_size": 5120, "hidden_size": 1280},
    "Language_model": {"num_layers": 40, "Q_Attention_heads": 32, "KV_Attention_heads": 8, "MLP_Intermedite_size": 14336, "hidden_size": 4096}
    }

mllama_11B_search_space = {
    "vision_model": {"num_layers": 32, "Attention_heads": [8, 16], "MLP_Intermedite_size": [2560,5120], "hidden_size": 1280},
    "Language_model": {"num_layers": 40, "Attention_head": [[32, 8], [16,4]], "MLP_Intermedite_size": 14336, "hidden_size": 4096}
    }
