
import inspect
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

from .utils import CONFIG_NAME, PeftType, TaskType


MIN_EXPECTED_CONFIG_KEYS = {"peft_type"}


def _check_and_remove_unused_kwargs(cls, kwargs):
    signature_parameters = inspect.signature(cls.__init__).parameters
    unexpected_kwargs = set(kwargs.keys()) - set(signature_parameters.keys())
    for key in unexpected_kwargs:
        del kwargs[key]
    return kwargs, unexpected_kwargs


@dataclass
class PeftConfigMixin(PushToHubMixin):
    task_type: Optional[TaskType] = field(default=None, metadata={"help": "The type of task."})
    peft_type: Optional[PeftType] = field(default=None, metadata={"help": "The type of PEFT model."})
    auto_mapping: Optional[dict] = field(
        default=None, metadata={"help": "An auto mapping dict to help retrieve the base model class if needed."}
    )

    def __post_init__(self):
        # check for invalid task type
        if (self.task_type is not None) and (self.task_type not in list(TaskType)):
            raise ValueError(
                f"Invalid task type: '{self.task_type}'. Must be one of the following task types: {', '.join(TaskType)}."
            )

    def to_dict(self) -> Dict:
        r"""
        Returns the configuration for your adapter model as a dictionary.
        """
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = self.to_dict()
        # converting set type to list
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_peft_type(cls, **kwargs):
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        if "peft_type" in kwargs:
            peft_type = kwargs["peft_type"]
            config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        else:
            config_cls = cls

        try:
            config = config_cls(**kwargs)
        except TypeError as exc:
            if "got an unexpected keyword argument" not in str(exc):
                raise exc

            filtered_kwargs, unexpected_kwargs = _check_and_remove_unused_kwargs(cls, kwargs)
            if not MIN_EXPECTED_CONFIG_KEYS.issubset(set(filtered_kwargs.keys())):
                raise TypeError(f"The config that is trying to be loaded is not a valid {cls.__name__} config.")

            warnings.warn(
                f"Unexpected keyword arguments {sorted(unexpected_kwargs)} for class {cls.__name__}, these are "
                "ignored. This probably means that you're loading a configuration file that was saved using a "
                "higher version of the library and additional parameters have been introduced since. It is "
                "highly recommended to upgrade the PEFT version before continuing (e.g. by running `pip install "
                "-U peft`)."
            )
            config = config_cls.from_peft_type(**filtered_kwargs)
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception as exc:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'") from exc

        loaded_attributes = cls.from_json_file(config_file)
        kwargs = {**class_kwargs, **loaded_attributes}
        kwargs = cls.check_kwargs(**kwargs)
        return cls.from_peft_type(**kwargs)

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file) as file:
            json_object = json.load(file)

        # Sanity check that config does not contain a runtime_config
        if "runtime_config" in json_object:
            warnings.warn(
                "The configuration file contains a `runtime_config` key. This is ignored. Runtime configurations are only valid at runtime."
            )
            del json_object["runtime_config"]

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def _get_peft_type(
        cls,
        model_id: str,
        **hf_hub_download_kwargs,
    ):
        subfolder = hf_hub_download_kwargs.get("subfolder", None)

        path = os.path.join(model_id, subfolder) if subfolder is not None else model_id

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    model_id,
                    CONFIG_NAME,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{model_id}'")

        loaded_attributes = cls.from_json_file(config_file)
        return loaded_attributes["peft_type"]

    @classmethod
    def check_kwargs(cls, **kwargs):
        """Check kwargs before initializing the config instance.

        Subclasses can override this method to add specific checks.

        """
        return kwargs

    @property
    def is_prompt_learning(self) -> bool:
        r"""
        Utility method to check if the configuration is for prompt learning.
        """
        return False

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return False


@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a [`PeftModel`].

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    revision: Optional[str] = field(default=None, metadata={"help": "The specific base model version to use."})
    peft_type: Optional[Union[str, PeftType]] = field(default=None, metadata={"help": "Peft type"})
    task_type: Optional[Union[str, TaskType]] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


# @dataclass
# class PromptLearningConfig(PeftConfig):
#     """
#     This is the base configuration class to store the configuration of [`PrefixTuning`], [`PromptEncoder`], or
#     [`PromptTuning`].

#     Args:
#         num_virtual_tokens (`int`): The number of virtual tokens to use.
#         token_dim (`int`): The hidden embedding dimension of the base transformer model.
#         num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
#         num_attention_heads (`int`): The number of attention heads in the base transformer model.
#         num_layers (`int`): The number of layers in the base transformer model.
#     """

#     num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
#     token_dim: int = field(
#         default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
#     )
#     num_transformer_submodules: Optional[int] = field(
#         default=None, metadata={"help": "Number of transformer submodules"}
#     )
#     num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
#     num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})

#     @property
#     def is_prompt_learning(self) -> bool:
#         r"""
#         Utility method to check if the configuration is for prompt learning.
#         """
#         return True