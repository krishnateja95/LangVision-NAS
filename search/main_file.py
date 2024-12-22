# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from search import main

if __name__ == "__main__":

    import time, os
    start_time = time.perf_counter()
    results, train_config, peft_config = fire.Fire(main)
    end_time = time.perf_counter()

    total_time = start_time - end_time 
    rank = int(os.environ["RANK"]) 
    
    if rank == 0:
        print(results)
        
        import csv
        list_1 = ["Hardware", "Num of Hardware", "Model", "Trainable", "All params", "Dataset", "target_modules", "LoRA Rank", "Avg Epoch Time", "Eval PPL"]
        list_2 = ["Nvidia A100 GPU", 4, train_config.model_name, results["trainable_params"], results["all_param"], "ocrvqa", peft_config.target_modules, peft_config.r, results["avg_epoch_time"], float(results["best_eval"])] 
        assert len(list_1) == len(list_2)

        csv_file = "LoRA_Bench.csv"
        file_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(list_1)
            
            writer.writerow(list_2) 
            
        csvfile.close()
