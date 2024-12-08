from __future__ import annotations

import collections
import copy
import inspect
import os
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
# from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, named_module_tensors
from huggingface_hub import HfFileSystem, ModelCard, ModelCardData, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
from transformers.utils import PushToHubMixin

from .utils.constants import DUMMY_MODEL_CONFIG, PEFT_TYPE_TO_PREFIX_MAPPING

from . import __version__
from .config import PeftConfig
from .tuners import LoraModel

from .tuners.tuners_utils import BaseTuner, BaseTunerLayer

from .utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    id_tensor_storage,
    infer_device,
    load_peft_weights,
    map_cache_to_layer_device_map,
    set_peft_model_state_dict,
    shift_tokens_right,
)

PEFT_TYPE_TO_MODEL_MAPPING = {PeftType.LORA: LoraModel}


class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
            self.set_additional_trainable_modules(peft_config, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
            if not isinstance(adapters, list):
                adapters = self.active_adapter
                if isinstance(adapters, str):
                    adapters = [adapters]
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        convert_pissa_to_lora: Optional[str] = None,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )
        # TODO: remove deprecated parameter in PEFT v0.14.0
        if convert_pissa_to_lora is not None:
            warnings.warn(
                "`convert_pissa_to_lora` is deprecated and will be removed in a future version. "
                "Use `path_initial_model_for_weight_conversion` instead."
            )
            path_initial_model_for_weight_conversion = convert_pissa_to_lora

        def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
            if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
                msg = (
                    "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                    "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
                )
                raise ValueError(msg)

            if not any(
                str(peft_config.init_lora_weights).lower().startswith(prefix) for prefix in ["pissa", "olora", "true"]
            ):
                warnings.warn(
                    "`path_initial_model_for_weight_conversion` only works for converting a PiSSA or OLoRA adapter to "
                    "a LoRA adapter"
                )
            initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
            try:
                self.load_adapter(
                    os.path.dirname(path_initial_model_for_weight_conversion),
                    subfolder=initial_adapter_name,
                    adapter_name=initial_adapter_name,
                )
                is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
                is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
                if is_pissa or is_olora:
                    raise ValueError(
                        "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                        "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                        "residual model."
                    )
                output_state_dict = self.base_model.subtract_mutated_init(
                    output_state_dict, initial_adapter_name, kwargs
                )
            finally:
                self.delete_adapter(initial_adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        ptrs[id(tensor)].append(name)

                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    if not peft_config.use_rslora:
                        peft_config.lora_alpha *= 2
                    else:
                        peft_config.lora_alpha *= 2**0.5

                    if peft_config.rank_pattern:
                        peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                    if peft_config.alpha_pattern:
                        peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # Runtime configuration, if supported
        if hasattr(config, "runtime_config"):
            config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
        else:
            if ephemeral_gpu_offload:
                warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

        if hasattr(model, "hf_device_map"):
            weight_map = dict(named_module_tensors(model, recurse=True))

            disk_modules = set()
            index = None
            for name, module in model.named_modules():
                if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                    if hasattr(module._hf_hook.weights_map, "dataset"):
                        index = module._hf_hook.weights_map.dataset.index
                    for key in module._hf_hook.original_devices.keys():
                        if module._hf_hook.original_devices[key] == torch.device("meta"):
                            disk_modules.add(str(name) + "." + str(key))

            if disk_modules and not kwargs.get("use_safetensors", True):
                raise ValueError("Disk offloading currently only supported for safetensors")

            if index:
                offload_index = {
                    p: {
                        "safetensors_file": index[p]["safetensors_file"],
                        "weight_name": p,
                        "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                    }
                    for p in weight_map.keys()
                    if p in disk_modules
                }
                kwargs["offload_index"] = offload_index

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        if isinstance(getattr(model, "base_model", None), XLoraModel):
            if not isinstance(config, XLoraConfig):
                raise TypeError(f"Expected 'XLoraConfig', got '{type(config)}' instead.")
            if "adapters" in kwargs:
                config.adapters = kwargs["adapters"]
            else:
                # If the path is on HF hub, then we get the adapter names to create a subfolders list which tells
                # `load_adapter` where the adapters are.
                if not os.path.exists(model_id):
                    s = HfFileSystem()

                    # The names of the adapters which must be in folders
                    adapter_names = [
                        file["name"][len(model_id) + 1 :] for file in s.ls(model_id) if file["type"] == "directory"
                    ]
                    # Prepare a dict of adapter paths, which really just point to the hf id; we will use the subfolders
                    adapter_paths = {}
                    for adapter_name in adapter_names:
                        adapter_paths[adapter_name] = os.path.join(model_id, model_id)
                    config.adapters = adapter_paths
                    config._subfolders = adapter_names
                else:
                    if "adapters" not in kwargs:
                        raise ValueError("If model_id is a local path, then `adapters` must be passed in kwargs.")

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        load_result = model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )

        missing_keys = [
            k for k in load_result.missing_keys if "vblora_vector_bank" not in k and "prompt_encoder" not in k
        ]
        if missing_keys:
            warnings.warn(f"Found missing adapter keys while loading the checkpoint: {missing_keys}")

        return model

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.base_model, name)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If the base model has a method called _enable_peft_forward_hooks, it is invoked as a context. Otherwise, this
        # runs without any changes
        if hasattr(self.base_model, "_enable_peft_forward_hooks"):
            with self.base_model._enable_peft_forward_hooks(*args, **kwargs):
                yield
            return
        else:
            # nothing to enable
            yield
            return

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model().generate(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return (
            self.base_model
            if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
            else self.base_model.model
        )

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(
                    self.base_model.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
                )
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)  # this may add a new ModulesToSaveWrapper

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def _update_offload(self, offload_index: dict[str, dict[str, str]], adapters_weights: dict[str, torch.tensor]):
        if not offload_index:
            return offload_index

        prefix = "base_model.model."
        # rename offload index weight and model names
        adapter_names = list(self.peft_config.keys())
        for adapter_name in adapter_names:
            keys = list(offload_index.keys())
            block_id = keys[0].split(".")[0] + "."  # for writing safetensors key,

            # replace original offload index keys with PeftModel keys
            for key in keys:
                suffix_pos = key.rfind(".")
                extended_prefix = prefix + key[:suffix_pos]
                module = dict(self.named_modules())[extended_prefix]
                if isinstance(module, BaseTunerLayer):
                    new_key = prefix + key[:suffix_pos] + ".base_layer" + key[suffix_pos:]
                else:
                    new_key = prefix + key
                offload_index[key]["weight_name"] = new_key
                offload_index[new_key] = offload_index[key]
                del offload_index[key]

            files_seen = set()
            # rename safetensors for dispatch
            for new_key in list(offload_index.keys()):
                fname = offload_index[new_key]["safetensors_file"]

                # make a new file name
                new_fname_list = list(fname.split(os.sep))
                for i, name in enumerate(new_fname_list):
                    if "--" in name:
                        new_fname_list[i] += "-peft"
                        break
                new_fname = os.path.join(*new_fname_list)

                if fname in files_seen:
                    continue
                safe_dict = {}
                with safe_open(fname, framework="pt") as f:
                    for safe_key in f.keys():
                        safe_tensor = f.get_tensor(safe_key)
                        metadata = f.metadata()
                        suffix_pos = safe_key.rfind(".")
                        extended_prefix = prefix + block_id + safe_key[:suffix_pos]
                        safe_module = dict(self.named_modules())[extended_prefix]
                        if isinstance(safe_module, BaseTunerLayer):
                            final_key = extended_prefix + ".base_layer" + safe_key[suffix_pos:]
                            lora_dict = {key: val for key, val in adapters_weights.items() if extended_prefix in key}

                            # add LoRA keys and values to disk offload
                            for lora_key, lora_val in lora_dict.items():
                                divide = lora_key.rfind(".")
                                new_key = lora_key[:divide] + f".{adapter_name}" + lora_key[divide:]
                                safe_dict[new_key] = lora_val
                        else:
                            final_key = prefix + block_id + safe_key
                        safe_dict[final_key] = safe_tensor
                    files_seen.add(new_fname)

                    # avoid overwriting original safetensors
                    for key in safe_dict.keys():
                        offload_index[key] = {"safetensors_file": new_fname, "weight_name": key}

                    base_name = os.path.dirname(new_fname)
                    if not os.path.exists(base_name):
                        os.makedirs(base_name)
                    safe_save_file(safe_dict, new_fname, metadata=metadata)

    def _check_new_adapter_config(self, peft_config: PeftConfig, is_trainable: bool) -> None:
        """Perform checks on newly added PEFT configs to ensure integrity."""
        if peft_config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")

        # Since PiSSA/OLoRA modifies the base weights, it should not be combined with other adapters.
        all_configs = [peft_config] + list(self.peft_config.values())
        if len(all_configs) > 1:
            if any(getattr(config, "init_lora_weights", None) == "pissa" for config in all_configs):
                msg = (
                    "PiSSA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the PiSSA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/pissa_finetuning#convert-pissa-to-lora"
                )
                warnings.warn(msg)
            elif any(getattr(config, "init_lora_weights", None) == "olora" for config in all_configs):
                msg = (
                    "OLoRA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the OLoRA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/olora_finetuning#olora-and-lora"
                )
                warnings.warn(msg)

    def load_adapter(
        self,
        model_id: Union[str, os.PathLike],
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ):
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                ephemeral_gpu_offload=ephemeral_gpu_offload,
                **hf_hub_download_kwargs,
            )
            self._check_new_adapter_config(peft_config, is_trainable=is_trainable)
            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self,
            adapters_weights,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        tuner = self.peft_config[adapter_name].peft_type
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
        adapter_missing_keys = []

        # Filter missing keys specific to the current adapter and tuner prefix.
        for key in load_result.missing_keys:
            if tuner_prefix in key and adapter_name in key:
                adapter_missing_keys.append(key)

        load_result.missing_keys.clear()
        load_result.missing_keys.extend(adapter_missing_keys)

        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )

            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )

            self._update_offload(offload_index, adapters_weights)
            dispatch_model_kwargs["offload_index"] = offload_index

            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )

            hook = AlignDevicesHook(io_same_device=True)
            if self.peft_config[adapter_name].is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        
        filename = os.path.join(output_dir, "README.md")

        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

        card.data["library_name"] = "peft"

        model_config = BaseTuner.get_model_config(self)
        model_config = None if model_config == DUMMY_MODEL_CONFIG else model_config
        if model_config is not None and "_name_or_path" in model_config:
            card.data["base_model"] = model_config["_name_or_path"]

        lines = card.text.splitlines()

        quantization_config = None
        if hasattr(model_config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += f"\n{quantization_prefix}\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure"
        if quantization_prefix not in lines and bool(training_config_text):
            if training_procedure_heading in lines:
                lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
            else:
                lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions"
        if f"- PEFT {__version__}" not in lines:
            if framework_block_heading in lines:
                lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}")
            else:
                lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}")

        card.text = "\n".join(lines)
        card.save(filename)


class PeftModelForCausalLM(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, task_ids=None, **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.peft_type == PeftType.CPT:
            return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(transformers.__version__) > packaging.version.parse("4.43.3"):
            transformers_new_cache_archs.append("bloom")

        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if model_kwargs.get("attention_mask", None) is not None:
                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
                isinstance(model_kwargs["past_key_values"], transformers.Cache)
                and not model_kwargs["past_key_values"].get_seq_length()
            )

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                new_past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None

        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs


