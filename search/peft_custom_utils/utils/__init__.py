from .integrations import map_cache_to_layer_device_map

from .peft_types import PeftType, TaskType
from .constants import INCLUDE_LINEAR_LAYERS_SHORTHAND
from .other import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING,
    CONFIG_NAME,
    WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    _set_trainable,
    bloom_model_postprocess_past_key_value,
    prepare_model_for_kbit_training,
    shift_tokens_right,
    transpose,
    _get_batch_size,
    _get_submodules,
    _set_adapter,
    _freeze_adapter,
    ModulesToSaveWrapper,
    _prepare_prompt_learning_config,
    _is_valid_match,
    infer_device,
    get_auto_gptq_quant_linear,
    get_quantization_config,
    id_tensor_storage,
    cast_mixed_precision_params,
)
from .save_and_load import get_peft_model_state_dict, set_peft_model_state_dict, load_peft_weights