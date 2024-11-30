from transformers import AutoModelForCausalLM, AutoProcessor
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def get_model(model_name):

    if "meta-llama/Llama-3.2-11B-Vision" in model_name:

        model = MllamaForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_name)

        






