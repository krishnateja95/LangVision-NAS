


torchrun --nnodes 1 --nproc_per_node 4  finetuning.py \
                                        --enable_fsdp \ 
                                        --model_name /path_of_model_folder/8B \
                                         --use_peft \
                                        --peft_method lora \ 
                                        --output_dir Path/to/save/PEFT/model \

                                        