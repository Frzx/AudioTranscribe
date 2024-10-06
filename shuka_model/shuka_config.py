####################################################################
## Built on top of Ultravox: https://github.com/fixie-ai/ultravox ##
####################################################################

import dataclasses
from typing import Any, Dict, List, Optional

import transformers


@dataclasses.dataclass
class LoraConfigSimplified:
    """
    Low Rank Approximation (LoRA) configuration.

    Used for language and audio models separately.
    """

    # The rank of the approximation
    r: int = 0
    lora_alpha: float = 8
    target_modules: Optional[List[str]] = dataclasses.field(
        default_factory=lambda: ["k_proj", "q_proj", "linear_k", "linear_q"]
    )


class ShukaConfig(transformers.PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ShukaForConditionalGeneration`]. It is used to instantiate an
    Shuka model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Wav2Vec2Config`,  *optional*):
            Custom audio config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        encoder_ds_factor (`int`, *optional*, defaults to 320):
            The audio encoder downsample factor.
        stack_factor (`int`, *optional*, defaults to 8):
            Audio downsampling factor for the multimodal projector.
        norm_init (`float`, *optional*, defaults to 0.4):
            The initialization value for the layer normalization.
        projector_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used by the multimodal projector.
        text_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the text model.
        audio_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the audio model.


    Example:

    ```python
    >>> from transformers import ShukaForConditionalGeneration, Wav2Vec2Config, ShukaConfig, LlamaConfig

    >>> # Initializing an audio encoder config
    >>> audio_config = Wav2Vec2Config()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = ShukaConfig(audio_config, text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = ShukaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = ShukaConfig(audio_model_id="facebook/wav2vec2-base-960h", text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "shuka"
    is_composition = False

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        audio_model_id: Optional[str] = None,
        text_model_id: Optional[str] = None,
        ignore_index: int = -100,
        audio_token_index: int = 32000,
        hidden_size: int = 4096,
        encoder_ds_factor: int = 320,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "silu",
        text_model_lora_config: Optional[LoraConfigSimplified] = None,
        audio_model_lora_config: Optional[LoraConfigSimplified] = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.audio_token_index = audio_token_index

        self.hidden_size = hidden_size
        self.encoder_ds_factor = encoder_ds_factor
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act

        if text_model_id is not None:
            self.text_config: transformers.LlamaConfig = (
                transformers.AutoConfig.from_pretrained(text_model_id)
            )
        else:
            text_config = text_config or {}
            self.text_config = transformers.CONFIG_MAPPING[
                text_config.get("model_type", "llama")
            ](**text_config)

        if audio_model_id is not None:
            self.audio_config: transformers.PretrainedConfig = (
                transformers.AutoConfig.from_pretrained(audio_model_id)
            )
        else:
            audio_config = audio_config or {}
            self.audio_config = transformers.CONFIG_MAPPING[
                audio_config.get("model_type", "wav2vec2")
            ](**audio_config)

        self.text_model_lora_config = (
            text_model_lora_config
            if isinstance(text_model_lora_config, dict)
            else dataclasses.asdict(text_model_lora_config or LoraConfigSimplified())
        )
        self.audio_model_lora_config = (
            audio_model_lora_config
            if isinstance(audio_model_lora_config, dict)
            else dataclasses.asdict(audio_model_lora_config or LoraConfigSimplified())
        )

        self.vocab_size = self.text_config.vocab_size

        self.initializer_range = self.text_config.initializer_range

        super().__init__(**kwargs)
