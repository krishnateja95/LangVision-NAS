
from __future__ import annotations

import copy
import logging
import os
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union

import torch
from accelerate import init_empty_weights
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

import sys
sys.path.append("../")

from ..utils.constants import (
    DUMMY_MODEL_CONFIG,
    DUMMY_TARGET_MODULES,
    EMBEDDING_LAYER_NAMES,
    MIN_TARGET_MODULES_FOR_OPTIMIZATION,
    SEQ_CLS_HEAD_NAMES,
    INCLUDE_LINEAR_LAYERS_SHORTHAND
)
from ..utils.peft_types import PeftType, TaskType

from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
from ._buffer_dict import BufferDict


logger = logging.getLogger(__name__)


@contextmanager
def onload_layer(layer):
    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # find the disk-offload index (maps modules to safetensors) from the `dataset` (OffloadedWeightsLoader object)
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any module will do
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []
            # get effective dir name
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    yield

    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }
        # offload weights map to disk if original device is the disk
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # rewrite directory with merged weights
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))


class BaseTuner(nn.Module, ABC):
    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, search_space: list = []):
        super().__init__()

        self.model = model
        self.targeted_module_names: list[str] = []

        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)


        self.active_adapter: str | list[str] = adapter_name
        self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
        if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
            self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, search_space = search_space)

        self.model.peft_config = self.peft_config

    def get_sampled_network(self, rank):
        print("rank is", rank)
        return 
    
    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def _pre_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        pass

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        ...

    def _prepare_model(self, peft_config: PeftConfig, model: nn.Module):
        pass

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        ...

    @abstractmethod
    def disable_adapter_layers(self) -> None:
        ...

    @abstractmethod
    def enable_adapter_layers(self) -> None:
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        pass

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        if not autocast_adapter_dtype:
            return

        dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}

        for module in self.model.modules():
            if not isinstance(module, BaseTunerLayer):
                continue

            for submodule in module.modules():
                if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                    continue

                if adapter_name not in submodule:
                    continue

                if isinstance(submodule[adapter_name], nn.Parameter):
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name].data = submodule[adapter_name].data.to(torch.float32)
                    continue

                if isinstance(submodule[adapter_name], torch.Tensor):  # e.g. from a BufferDict
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name] = submodule[adapter_name].to(torch.float32)
                    continue

                for param in submodule[adapter_name].parameters():
                    if param.dtype in dtypes_to_convert_to_fp32:
                        param.data = param.data.to(torch.float32)

    def _check_merge_allowed(self):
        example_code = textwrap.dedent(
            """
            ```python
            from transformers import AutoModelForCausalLM

            # Load original tied model
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", tie_word_embeddings=False)

            # Set the randomly initialized lm_head to the previously tied embeddings
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

            # Save the untied model
            untied_model_dir = "dir/for/untied/model"
            model.save_pretrained(untied_model_dir)
            model.config.save_pretrained(untied_model_dir)

            # Now use the original model but in untied format
            model = AutoModelForCausalLM.from_pretrained(untied_model_dir)
            ```
            """
        )
        tied_target_modules = self._get_tied_target_modules(self.model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules} are part of the adapter. "
                "This can lead to complications. "
                "You can opt to merge the adapter after cloning the weights (to untie the embeddings). "
                "You can untie the embeddings by loading the model with `tie_word_embeddings=False`. For example:"
                + example_code
            )

    def inject_adapter(self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False, search_space: list = []):

        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        self._check_new_adapter_config(peft_config)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = self.get_model_config(model)
        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        key_list = [key for key, _ in model.named_modules()]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            key_list = []

        peft_config = _maybe_include_all_linear_layers(peft_config, model)

        if (
            isinstance(peft_config.target_modules, (list, set))
            and len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION
        ):
            names_no_target = [
                name
                for name in key_list
                if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        for key in key_list:
            if not key:
                continue
            if _check_for_modules_to_save and any(key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save):
                parent, target, target_name = _get_submodules(model, key)
                
                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                continue

            result = self._check_target_module_exists(peft_config, key)
            if isinstance(result, _ExcludedModule):
                excluded_modules.append(key)
            elif not result:
                unmatched_modules.append(key)
            else:
                self.targeted_module_names.append(key)
                parent, target, target_name = _get_submodules(model, key)
                ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                with ctx():
                    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key, search_space = search_space)

        if not self.targeted_module_names and not uses_dummy_target_modules:
            if excluded_modules and not unmatched_modules:
                raise ValueError(
                    "All modules were excluded. This is likely unintended. "
                    "Check your `target_modules` and `exclude_modules` configuration."
                )
            elif not excluded_modules and unmatched_modules:
                # None of the targeted modules matched
                error_msg = (
                    f"Target modules {peft_config.target_modules} not found in the base model. "
                    f"Please check the target modules and try again."
                )
                if peft_config.layers_to_transform is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if peft_config.layers_pattern is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)
            else:
                # Some modules did not match and some matched but were excluded
                error_msg = (
                    "No modules were targeted for adaptation. "
                    "This might be caused by a combination of mismatched target modules and excluded modules. "
                    "Please check your `target_modules` and `exclude_modules` configuration."
                )
                if peft_config.layers_to_transform is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if peft_config.layers_pattern is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)

        elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
            # exclude_modules was passed but was not used
            warnings.warn(
                f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
                "Please check that exclude_modules was set correctly."
            )

        tied_target_modules = self._get_tied_target_modules(model=model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules} are part of the adapter. "
                "This can lead to complications, for example when merging the adapter "
                "or converting your model to formats other than safetensors. "
                "See for example https://github.com/huggingface/peft/issues/2018."
            )

        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def merge_adapter(self, adapter_names: Optional[list[str]] = None) -> None:
        self._check_merge_allowed()
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.merge(adapter_names=adapter_names)

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.unmerge()

    def _unloading_checks(self, adapter_names: Optional[list[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")

    @staticmethod
    def get_model_config(model: nn.Module) -> dict:
        model_config = getattr(model, "config", DUMMY_MODEL_CONFIG)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        return model_config

    def _get_tied_target_modules(self, model: nn.Module) -> list[str]:
        tied_target_modules = []
        model_config = self.get_model_config(model)
        if model_config.get("tie_word_embeddings"):
            for target_module in self.targeted_module_names:
                if target_module in EMBEDDING_LAYER_NAMES:
                    tied_target_modules.append(target_module)
        return tied_target_modules


class BaseTunerLayer(ABC):
    adapter_layer_names: tuple[str, ...] = ()
    other_param_names: tuple[str, ...] = ()

    _disable_adapters: bool = False

    _active_adapter: str | list[str] = "default"

    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def get_sampled_network(self, rank):
        print("rank is", rank)
        return
    
    @property
    def weight(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        return self._active_adapter

    def _get_available_adapters(self) -> set[str]:
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)


def _find_minimal_target_modules(target_modules: list[str] | set[str], other_module_names: list[str] | set[str]) -> set[str]:
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules should be a list or set of strings.")

    target_modules = set(target_modules)
    if "" in target_modules:
        raise ValueError("target_modules should not contain an empty string.")

    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = (
            "target_modules and other_module_names contain common elements, this should not happen, please "
            "open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue"
        )
        raise ValueError(msg)

    # it is assumed that module name parts are separated by a "."
    def generate_suffixes(s):
        parts = s.split(".")
        return [".".join(parts[i:]) for i in range(len(parts))][::-1]

    # Create a reverse lookup for other_module_names to quickly check suffix matches
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}

    # Find all potential suffixes from target_modules
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}

    # Initialize a set for required suffixes
    required_suffixes = set()

    # We sort the target_modules_suffix_map simply to get deterministic behavior, since sets have no order. In theory
    # the order should not matter but in case there is a bug, it's better for the bug to be deterministic.
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        # Go through target_modules items, shortest suffixes first
        for suffix in suffixes:
            # If the suffix is already in required_suffixes or matches other_module_names, skip it
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            # Check if adding this suffix covers the item
            if not any(item.endswith("." + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break

    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


class _ExcludedModule:
    def __bool__(self):
        return False


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return _ExcludedModule()

    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {"matched": [], "unmatched": []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)
        else:
            module_dict["unmatched"].append(key)
    return module_dict


def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    if not hasattr(peft_config, "target_modules"):
        return peft_config

    # if `target_modules` is a string, convert to lower case and check if it matches "all-linear"
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    if not isinstance(model, PreTrainedModel):
        raise ValueError(
            f"Only instances of PreTrainedModel support `target_modules={INCLUDE_LINEAR_LAYERS_SHORTHAND!r}`"
        )

    linear_classes = (torch.nn.Linear, Conv1D)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(names)

    # Try to remove linear layers that should not be targeted as best as possible. We have to rely on convention as
    # there are no hard rules to detect these modules.
    module_names_to_exclude = set()
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        # ignore the last classification head for text generation models
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        module_names_to_exclude.add(last_module_name)
    elif peft_config.task_type == TaskType.SEQ_CLS:
        # ignore classifier head for classification models (issue 2027)
        # there is no fix name for the classifier head, so check the common ones
        for name in SEQ_CLS_HEAD_NAMES:
            cls_head = getattr(model, name, None)
            if cls_head is not None:
                last_module_name = [name for name, module in model.named_modules() if module is cls_head][0]
                module_names_to_exclude.add(last_module_name)
                break

    linear_module_names -= module_names_to_exclude
    peft_config.target_modules = linear_module_names
    return peft_config


def check_adapters_to_merge(module: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names


def clone_module(module: nn.Module, share_weights=False):
    clone = copy.deepcopy(module)

    def _share_weights(src: nn.Module, dst: nn.Module):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)

    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))

    return clone


def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    while hasattr(model, "model"):
        model = model.model
    # Some variants of the bert model nest the main model under the bert attribute.
    if hasattr(model, "bert"):
        model = model.bert

    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, "layers"):
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        model_type = "falcon"
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            # This is a hack needed to work around the layer_idx introduced in HF transformers.
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")
    if hasattr(model.config, "num_hidden_layers"):  # Common to Llama, Bert, Falcon.
        model.config.num_hidden_layers = len(new_layers)