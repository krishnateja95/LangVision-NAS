
module use /soft/modulefiles/
module load conda

conda activate LLaMA_Finetune_bench

export HF_HOME="/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache"

cd ../ 

for rank in 8; do
    for lora_adapters in "qkvogudfc"; do 
        torchrun --nnodes 1 --nproc_per_node 4 main_file.py \
                                                --enable_fsdp \
                                                --lr 1e-5 \
                                                --num_epochs 1 \
                                                --batch_size_training 16 \
                                                --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
                                                --dist_checkpoint_root_folder "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/finetune_folder" \
                                                --dist_checkpoint_folder "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/finetuned_folder" \
                                                --use_fast_kernels \
                                                --dataset "custom_dataset" \
                                                --custom_dataset.test_split "test" \
                                                --custom_dataset.file "custom_data/ocrvqa_dataset.py" \
                                                --run_validation True \
                                                --batching_strategy padding \
                                                --use_peft \
                                                --peft_method lora \
                                                --output_dir "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/output_dir" \
                                                --profiler_dir "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/profile_dir" \
                                                --lora_rank $rank \
                                                --lora_adapters $lora_adapters \
                                                --finetune_model_dir "/lus/grand/projects/datascience/krishnat/krishnat_HF_weights/Llama-3.2-11B-Vision-finetuned" \
                                                --HF_repo "krishnateja95/Llama-3.2-11B-Vision-ocrvqa-finetuned" \
                                                --use_fp16
    done
done



