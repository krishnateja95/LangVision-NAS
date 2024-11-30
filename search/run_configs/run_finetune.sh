
module use /soft/modulefiles/
module load conda

conda activate LLaMA_Finetune_bench

export HF_HOME="/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache"

cd ../ 

torchrun --nnodes 1 --nproc_per_node 4  main_file.py \
                                        --enable_fsdp \
                                        --lr 1e-5 \
                                        --num_epochs 3 \
                                        --batch_size_training 8 \
                                        --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
                                        --dist_checkpoint_root_folder ./finetuned_model \
                                        --dist_checkpoint_folder fine-tuned \
                                        --use_fast_kernels \
                                        --dataset "custom_dataset" \
                                        --custom_dataset.test_split "test" \
                                        --custom_dataset.file "custom_data/ocrvqa_dataset.py" \
                                        --run_validation True \
                                        --batching_strategy padding \
                                        --use_peft \
                                        --peft_method lora \