# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters

DEFAULT_INT = -42


def _resolve_dtype(config) -> ir.DataType | None:
    """Extract model dtype from a HuggingFace config.

    Handles string dtypes (e.g. "float16"), torch.dtype objects,
    and the "auto" sentinel (returns None).
    """
    torch_dtype = getattr(config, "dtype", None)
    if torch_dtype is not None and torch_dtype != "auto":
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)
        if torch_dtype is not None:
            return tensor_adapters.from_torch_dtype(torch_dtype)
    return None


def _nested_rope_theta(rope_scaling: dict, key: str) -> float | None:
    """Extract rope_theta from a nested rope_scaling dict (e.g. Gemma3)."""
    sub = rope_scaling.get(key)
    if isinstance(sub, dict):
        return sub.get("rope_theta")
    return None


def _nested_rope_type(rope_scaling: dict, key: str) -> str | None:
    """Extract rope_type from a nested rope_scaling dict (e.g. Gemma3)."""
    sub = rope_scaling.get(key)
    if isinstance(sub, dict):
        return sub.get("rope_type")
    return None


def _normalize_rope_scaling(rope_scaling: dict) -> dict:
    """Flatten nested rope_scaling dicts (e.g. Gemma3).

    Gemma3 stores per-attention-type configs::

        {"full_attention": {"rope_type": "linear", "factor": 8.0, ...},
         "sliding_attention": {"rope_type": "default", ...}}

    This normalizes to the ``full_attention`` sub-dict so downstream
    code (e.g. ``LinearRope``) can find ``rope_scaling["factor"]``.
    """
    if not rope_scaling:
        return rope_scaling
    if "full_attention" in rope_scaling and isinstance(rope_scaling["full_attention"], dict):
        return rope_scaling["full_attention"]
    return rope_scaling


@dataclasses.dataclass
class RoPEConfig:
    """Configuration for rotary position embeddings (RoPE).

    Groups the 7 RoPE-related fields that were previously spread across
    :class:`ArchitectureConfig` as flat attributes.
    """

    rope_type: str = "default"
    rope_theta: float = 10_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0
    rope_local_base_freq: float | None = None
    original_max_position_embeddings: int | None = None
    rope_interleave: bool = False


@dataclasses.dataclass
class VisionConfig:
    """Configuration for the vision encoder in multimodal models.

    This groups all vision-related fields that were previously scattered
    as ``vision_*`` prefixed fields on :class:`ArchitectureConfig`.
    """

    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    image_size: int | None = None
    patch_size: int | None = None
    norm_eps: float = 1e-6
    mm_tokens_per_image: int | None = None
    image_token_id: int | None = None
    # Qwen VL-specific
    out_hidden_size: int | None = None
    in_channels: int = 3
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int | None = None
    deepstack_visual_indexes: list[int] | None = None
    fullatt_block_indexes: list[int] | None = None
    window_size: int | None = None
    # MRoPE section (for multimodal position encoding)
    mrope_section: list[int] | None = None
    # Phi4MM image embedding
    image_crop_size: int | None = None
    # LoRA config
    lora: dict | None = None


@dataclasses.dataclass
class CodecDecoderConfig:
    """Configuration for the codec decoder (codes → waveform)."""

    codebook_dim: int = 512
    codebook_size: int = 2048
    latent_dim: int = 1024
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8000
    decoder_dim: int = 1536
    num_quantizers: int = 16
    upsample_rates: list[int] = dataclasses.field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: list[int] = dataclasses.field(default_factory=lambda: [2, 2])


@dataclasses.dataclass
class CodecEncoderConfig:
    """Configuration for the codec encoder (waveform → codes)."""

    codebook_dim: int = 256
    codebook_size: int = 2048
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8000
    num_quantizers: int = 32
    num_semantic_quantizers: int = 1


@dataclasses.dataclass
class SpeakerEncoderConfig:
    """Configuration for the ECAPA-TDNN speaker encoder in TTS models."""

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = dataclasses.field(
        default_factory=lambda: [512, 512, 512, 512, 1536]
    )
    enc_kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128


@dataclasses.dataclass
class CodePredictorConfig:
    """Configuration for the TTS code predictor sub-model."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 2048
    num_code_groups: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    hidden_act: str = "silu"
    layer_types: list[str] | None = None


@dataclasses.dataclass
class TTSConfig:
    """Configuration for Qwen3-TTS models.

    Groups TTS-specific fields: talker parameters, code predictor config,
    and speaker encoder config.
    """

    # Talker parameters
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    num_code_groups: int = 16
    # Special token IDs
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_pad_id: int = 2148
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    # Sub-configs
    code_predictor: CodePredictorConfig | None = None
    speaker_encoder: SpeakerEncoderConfig | None = None


@dataclasses.dataclass
class AudioConfig:
    """Configuration for the audio encoder in multimodal models."""

    attention_dim: int | None = None
    attention_heads: int | None = None
    num_blocks: int | None = None
    linear_units: int | None = None
    kernel_size: int | None = None
    input_size: int | None = None
    conv_channels: int | None = None
    t5_bias_max_distance: int | None = None
    projection_hidden_size: int | None = None
    token_id: int | None = None
    # Qwen3-ASR encoder config
    d_model: int | None = None
    encoder_layers: int | None = None
    encoder_attention_heads: int | None = None
    encoder_ffn_dim: int | None = None
    num_mel_bins: int | None = None
    max_source_positions: int | None = None
    downsample_hidden_size: int | None = None
    output_dim: int | None = None
    activation_function: str = "gelu"
    audio_token_id: int | None = None
    audio_start_token_id: int | None = None
    audio_end_token_id: int | None = None
    classify_num: int | None = None
    # LoRA config
    lora: dict | None = None
    # HTSAT Swin (CLAP audio encoder)
    spec_size: int | None = None
    patch_size: int | None = None
    window_size: int | None = None
    depths: list[int] | None = None
    num_attention_heads: list[int] | None = None
    patch_embeds_hidden_size: int | None = None


def _first_not_none(*values, default=None):
    """Return the first value that is not None, or *default*."""
    for v in values:
        if v is not None:
            return v
    return default


def _extract_rope_config(config) -> RoPEConfig:
    """Extract and normalize RoPE-related config fields.

    Reads ``rope_scaling``, ``rope_parameters``, and related attributes
    from a HuggingFace config and returns a :class:`RoPEConfig`.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_parameters = getattr(config, "rope_parameters", None) or {}

    return RoPEConfig(
        rope_type=_first_not_none(
            rope_scaling.get("rope_type", None),
            rope_scaling.get("type", None),
            rope_parameters.get("rope_type", None),
            _nested_rope_type(rope_scaling, "full_attention"),
            default="default",
        ),
        rope_theta=_first_not_none(
            getattr(config, "rope_theta", None),
            rope_scaling.get("rope_theta", None),
            rope_parameters.get("rope_theta", None),
            _nested_rope_theta(rope_scaling, "full_attention"),
            default=10_000.0,
        ),
        rope_scaling=(_normalize_rope_scaling(rope_scaling) or None),
        partial_rotary_factor=_first_not_none(
            getattr(config, "partial_rotary_factor", None),
            rope_scaling.get("partial_rotary_factor", None),
            rope_parameters.get("partial_rotary_factor", None),
            default=1.0,
        ),
        rope_local_base_freq=_first_not_none(
            getattr(config, "rope_local_base_freq", None),
            _nested_rope_theta(rope_scaling, "sliding_attention"),
        ),
        original_max_position_embeddings=(
            getattr(
                config,
                "original_max_position_embeddings",
                rope_scaling.get("original_max_position_embeddings", None),
            )
        ),
    )


def _extract_mrope_fields(config) -> dict:
    """Extract MRoPE fields from a HuggingFace config.

    Returns a dict with ``mrope_interleaved`` and ``mrope_section``
    keys (only if present).  These are separate from :class:`RoPEConfig`
    because they affect multimodal position encoding and are shared
    with :class:`VisionConfig`.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    result: dict = {}
    mrope_interleaved = rope_scaling.get("mrope_interleaved", False) or rope_parameters.get(
        "mrope_interleaved", False
    )
    if mrope_interleaved:
        result["mrope_interleaved"] = True
        section = rope_scaling.get("mrope_section", None) or rope_parameters.get(
            "mrope_section", None
        )
        if section is not None:
            result["mrope_section"] = section
    return result


def _extract_vision_config(config, parent_config, model_type: str) -> dict:
    """Extract vision sub-config from a HuggingFace config.

    Builds a :class:`VisionConfig` (if vision fields are found) and
    returns a dict of options to merge into :class:`ArchitectureConfig`
    kwargs.
    """
    # Use parent_config (composite config) when available
    vision_source = parent_config or config
    hf_vision_config = getattr(vision_source, "vision_config", None)
    if hf_vision_config is None:
        hf_vision_config = getattr(config, "vision_config", None)

    vision_fields: dict = {}
    if hf_vision_config is not None:
        vc = (
            hf_vision_config
            if not isinstance(hf_vision_config, dict)
            else type("VC", (), hf_vision_config)()
        )
        vision_fields.update(
            hidden_size=getattr(vc, "hidden_size", None),
            intermediate_size=getattr(vc, "intermediate_size", None),
            num_hidden_layers=(
                getattr(vc, "num_hidden_layers", None) or getattr(vc, "depth", None)
            ),
            num_attention_heads=(
                getattr(vc, "num_attention_heads", None)
                or getattr(vc, "num_heads", None)
                or getattr(vc, "attention_heads", None)
            ),
            image_size=getattr(vc, "image_size", None),
            patch_size=getattr(vc, "patch_size", None),
            norm_eps=getattr(vc, "layer_norm_eps", 1e-6),
            # Qwen VL-specific vision fields
            out_hidden_size=getattr(vc, "out_hidden_size", None),
            in_channels=getattr(vc, "in_channels", 3),
            spatial_merge_size=getattr(vc, "spatial_merge_size", 2),
            temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
            num_position_embeddings=getattr(vc, "num_position_embeddings", None),
            deepstack_visual_indexes=getattr(vc, "deepstack_visual_indexes", None),
            fullatt_block_indexes=getattr(vc, "fullatt_block_indexes", None),
            window_size=getattr(vc, "window_size", None),
        )
    vision_fields["mm_tokens_per_image"] = getattr(vision_source, "mm_tokens_per_image", None)
    vision_fields["image_token_id"] = getattr(vision_source, "image_token_id", None)

    # MRoPE section — only for composite VL models
    # (parent_config != config)
    if parent_config is not None and parent_config is not config:
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        mrope_section = rope_scaling.get("mrope_section", None) or rope_parameters.get(
            "mrope_section", None
        )
        if mrope_section is not None:
            vision_fields["mrope_section"] = mrope_section

    # LoRA config (e.g. Phi4-MM)
    vision_lora = getattr(config, "vision_lora", None)
    if vision_lora is not None:
        vision_fields["lora"] = (
            vision_lora if isinstance(vision_lora, dict) else vars(vision_lora)
        )

    # Phi4MM image embedding config
    embd_layer = getattr(config, "embd_layer", None)
    if isinstance(embd_layer, dict):
        img_cfg = embd_layer.get("image_embd_layer", {})
        vision_fields["image_crop_size"] = img_cfg.get("crop_size")

    # TODO(Phase 1): Move phi4mm model-specific logic to a config subclass
    # override instead of hardcoding here. These SigLIP vision encoder params
    # are baked into the HF model code and not in the config JSON.
    if model_type == "phi4mm":
        vision_fields.update(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=(vision_fields.get("image_crop_size") or 448),
            patch_size=14,
            norm_eps=1e-6,
            image_token_id=getattr(config, "special_image_token_id", 200010),
        )

    # InternVL2 doesn't expose image_token_id in its config — default to
    # the Qwen2 <IMG_CONTEXT> token id used by InternVL2-* models.
    parent_model_type = getattr(vision_source, "model_type", None)
    if parent_model_type in ("internvl_chat", "internvl2", "internvl") or model_type in (
        "internvl_chat",
        "internvl2",
        "internvl",
    ):
        if vision_fields.get("image_token_id") is None:
            vision_fields["image_token_id"] = getattr(
                vision_source, "img_context_token_id", 151667
            )

    # Build VisionConfig sub-config if any vision fields are set
    has_vision = any(
        v is not None
        for k, v in vision_fields.items()
        if k
        not in (
            "norm_eps",
            "in_channels",
            "spatial_merge_size",
            "temporal_patch_size",
        )
    )

    result: dict = {}
    if has_vision:
        result["vision"] = VisionConfig(**vision_fields)
        # Also set shared fields at top-level for direct access
        # (e.g. config.image_token_id, config.spatial_merge_size).
        for shared in (
            "mm_tokens_per_image",
            "image_token_id",
            "spatial_merge_size",
            "temporal_patch_size",
            "deepstack_visual_indexes",
            "fullatt_block_indexes",
            "window_size",
            "mrope_section",
            "image_crop_size",
        ):
            val = vision_fields.get(shared)
            if val is not None:
                result[shared] = val

    return result


def _extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Extract audio sub-config from a HuggingFace config.

    Builds an :class:`AudioConfig` (if audio fields are found) and
    returns a dict of options to merge into :class:`ArchitectureConfig`
    kwargs.
    """
    audio_fields: dict = {}
    audio_processor = getattr(config, "audio_processor", None)
    if isinstance(audio_processor, dict) and "config" in audio_processor:
        ac = audio_processor["config"]
        nemo = ac.get("nemo_conv_settings", {})
        rel_bias = ac.get("relative_attention_bias_args", {})
        audio_fields.update(
            attention_dim=ac.get("attention_dim"),
            attention_heads=ac.get("attention_heads"),
            num_blocks=ac.get("num_blocks"),
            linear_units=ac.get("linear_units"),
            kernel_size=ac.get("kernel_size"),
            input_size=ac.get("input_size"),
            conv_channels=nemo.get("conv_channels", ac.get("attention_dim")),
            t5_bias_max_distance=rel_bias.get("t5_bias_max_distance"),
        )

    embd_layer = getattr(config, "embd_layer", None)
    if isinstance(embd_layer, dict):
        audio_fields["projection_hidden_size"] = config.hidden_size

    # Phi4MM audio token ID
    if model_type == "phi4mm":
        audio_config_dict = getattr(config, "audio_config", None)
        if audio_config_dict is not None:
            ac_dict = (
                audio_config_dict
                if isinstance(audio_config_dict, dict)
                else vars(audio_config_dict)
            )
            audio_fields["token_id"] = ac_dict.get("audio_token_id")

    speech_lora = getattr(config, "speech_lora", None)
    if speech_lora is not None:
        audio_fields["lora"] = (
            speech_lora if isinstance(speech_lora, dict) else vars(speech_lora)
        )

    # Qwen3-ASR audio config (from thinker_config)
    thinker_config_source = parent_config or config
    hf_thinker_config = getattr(thinker_config_source, "thinker_config", None)
    if hf_thinker_config is not None:
        tc = (
            hf_thinker_config
            if not isinstance(hf_thinker_config, dict)
            else type("TC", (), hf_thinker_config)()
        )
        hf_audio_config = getattr(tc, "audio_config", None)
        if hf_audio_config is not None:
            ac = (
                hf_audio_config
                if not isinstance(hf_audio_config, dict)
                else type("AC", (), hf_audio_config)()
            )
            audio_fields.update(
                d_model=getattr(ac, "d_model", None),
                encoder_layers=getattr(ac, "encoder_layers", None),
                encoder_attention_heads=getattr(ac, "encoder_attention_heads", None),
                encoder_ffn_dim=getattr(ac, "encoder_ffn_dim", None),
                num_mel_bins=getattr(ac, "num_mel_bins", None),
                max_source_positions=getattr(ac, "max_source_positions", None),
                downsample_hidden_size=getattr(ac, "downsample_hidden_size", None),
                output_dim=getattr(ac, "output_dim", None),
                activation_function=getattr(ac, "activation_function", "gelu"),
            )
        # Special tokens from thinker config
        audio_fields["audio_token_id"] = getattr(tc, "audio_token_id", None)
        audio_fields["audio_start_token_id"] = getattr(tc, "audio_start_token_id", None)
        audio_fields["audio_end_token_id"] = getattr(tc, "audio_end_token_id", None)
        audio_fields["classify_num"] = getattr(tc, "classify_num", None)

    # CLAP audio config (laion/clap-htsat-fused and similar)
    if model_type in ("clap_audio_model", "clap"):
        hf_audio_config = getattr(config, "audio_config", None)
        if hf_audio_config is not None:
            ac = (
                hf_audio_config
                if not isinstance(hf_audio_config, dict)
                else type("AC", (), hf_audio_config)()
            )
            audio_fields.update(
                spec_size=getattr(ac, "spec_size", 256),
                num_mel_bins=getattr(ac, "num_mel_bins", 64),
                patch_size=getattr(ac, "patch_size", 4),
                window_size=getattr(ac, "window_size", 8),
                depths=getattr(ac, "depths", [2, 2, 6, 2]),
                num_attention_heads=getattr(ac, "num_attention_heads", [4, 8, 16, 32]),
                patch_embeds_hidden_size=getattr(ac, "patch_embeds_hidden_size", 96),
            )

    # Build AudioConfig sub-config if any audio fields are set
    has_audio = any(v is not None for v in audio_fields.values())

    result: dict = {}
    if has_audio:
        result["audio"] = AudioConfig(**audio_fields)

    return result


@dataclasses.dataclass
class QuantizationConfig:
    """Weight quantization parameters parsed from HuggingFace configs.

    Captures the settings from ``quantization_config`` in HuggingFace model
    configs (GPTQ, AWQ, etc.) so models can decide whether to use
    :class:`~mobius.components.QuantizedLinear` instead of
    :class:`~mobius.components.Linear`.
    """

    bits: int = 4
    group_size: int = 128
    quant_method: str = "none"
    sym: bool = True

    @classmethod
    def from_transformers(cls, hf_config) -> QuantizationConfig | None:
        """Parse ``quantization_config`` from a HuggingFace config.

        Returns ``None`` when no quantization is configured.
        """
        qc = getattr(hf_config, "quantization_config", None)
        if qc is None:
            return None
        # qc can be a dict or a HF QuantizationConfig object
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        if not isinstance(qc, dict):
            return None
        method = qc.get("quant_method", "none")
        if method == "none":
            return None
        return cls(
            bits=qc.get("bits", 4),
            group_size=qc.get("group_size", 128),
            quant_method=method,
            sym=qc.get("sym", True),
        )


@dataclasses.dataclass
class BaseModelConfig:
    """Base configuration shared by all model architectures.

    Contains the minimal set of fields needed by the task/exporter
    infrastructure (KV cache shapes, dtype casting, etc.).
    """

    vocab_size: int = DEFAULT_INT
    hidden_size: int = DEFAULT_INT
    intermediate_size: int = DEFAULT_INT
    num_hidden_layers: int = DEFAULT_INT
    num_attention_heads: int = DEFAULT_INT
    num_key_value_heads: int = DEFAULT_INT
    head_dim: int = DEFAULT_INT
    hidden_act: str | None = None
    pad_token_id: int = DEFAULT_INT
    tie_word_embeddings: bool = False
    attn_qkv_bias: bool = False
    attn_o_bias: bool = False

    # Model dtype (from HF config dtype)
    dtype: ir.DataType = ir.DataType.FLOAT


@dataclasses.dataclass
class ArchitectureConfig(BaseModelConfig):
    """Configuration for decoder-only model architectures."""

    max_position_embeddings: int = DEFAULT_INT

    # attention config
    layer_types: list[str] | None = None
    full_attention_interval: int | None = None
    sliding_window: int | None = None

    # Linear attention (DeltaNet) config
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None

    rms_norm_eps: float = 1e-6

    # Rotary embedding config
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0
    rope_local_base_freq: float | None = None
    original_max_position_embeddings: int | None = None

    attn_qk_norm: bool = False
    attn_qk_norm_full: bool = False
    mlp_bias: bool = False

    # Encoder-specific config
    type_vocab_size: int = 0

    # Encoder-decoder config
    num_decoder_layers: int | None = None
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    is_gated_act: bool = False
    scale_decoder_outputs: bool | None = None

    # MoE config
    num_local_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    # When True, the decoder layer uses post-norm style (FlexOLMo): norms are applied
    # to sub-layer outputs instead of inputs, with an extra post_feedforward_layernorm.
    post_feedforward_norm: bool = False
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    first_k_dense_replace: int = 0
    n_shared_experts: int | None = None

    # Multi-head Latent Attention (MLA) config — DeepSeek-V2/V3
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    rope_interleave: bool = False

    # Vision shared fields (accessed as top-level config.X by tasks)
    mm_tokens_per_image: int | None = None
    image_token_id: int | None = None
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    deepstack_visual_indexes: list[int] | None = None
    fullatt_block_indexes: list[int] | None = None
    window_size: int = 112

    # Q-Former config (for BLIP-2 style models)
    num_query_tokens: int | None = None
    qformer_hidden_size: int | None = None
    qformer_num_hidden_layers: int | None = None
    qformer_num_attention_heads: int | None = None
    qformer_intermediate_size: int | None = None

    # MRoPE config (for multimodal position encoding)
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False

    # Standalone vision config
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    # Audio config (for multimodal models like Phi4-MM)
    audio_attention_dim: int | None = None
    audio_attention_heads: int | None = None
    audio_num_blocks: int | None = None
    audio_linear_units: int | None = None
    audio_kernel_size: int | None = None
    audio_input_size: int | None = None
    audio_conv_channels: int | None = None
    audio_t5_bias_max_distance: int | None = None
    audio_token_id: int | None = None

    # LoRA config (for multimodal models like Phi4-MM)
    speech_lora: dict | None = None

    # Phi4MM image embedding config
    image_crop_size: int | None = None

    # Falcon config
    alibi: bool = False
    parallel_attn: bool = False
    dual_ln: bool = False  # True for models with two separate norms in parallel layers (MPT, GPT-NeoX-Falcon)

    # Post-norm vs pre-norm architecture toggle (used by OpenAI-GPT vs standard GPT-2)
    post_norm: bool = False

    # Granite scaling multipliers
    embedding_multiplier: float = 1.0
    attention_multiplier: float | None = None
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0

    # Cohere logit scale: multiplied into the final logits before softmax
    logit_scale: float = 1.0

    # YOLOS object detection config
    num_labels: int = 91

    # CLIPSeg segmentation decoder config
    projection_dim: int | None = None
    reduce_dim: int = 64
    extract_layers: list[int] | None = None
    conditional_layer: int = 0
    decoder_num_attention_heads: int = 4
    decoder_intermediate_size: int | None = None
    decoder_hidden_act: str = "quick_gelu"

    # Composed sub-configs
    rope: RoPEConfig | None = None
    vision: VisionConfig | None = None
    audio: AudioConfig | None = None
    tts: TTSConfig | None = None
    codec_decoder: CodecDecoderConfig | None = None
    codec_encoder: CodecEncoderConfig | None = None
    quantization: QuantizationConfig | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ArchitectureConfig:
        model_type = config.model_type
        rope_config = _extract_rope_config(config)
        mrope_fields = _extract_mrope_fields(config)

        # Some hierarchical models (Segformer, Swin) use plural list attrs
        # instead of scalar ones.  Resolve to a scalar for the base config.
        hidden_size = (
            getattr(config, "hidden_size", None)
            or _first(getattr(config, "hidden_sizes", None))
            or 0
        )
        num_attention_heads = (
            getattr(config, "num_attention_heads", None)
            or _first(getattr(config, "num_heads", None))
            or 1
        )
        num_hidden_layers = (
            getattr(config, "num_hidden_layers", None)
            or getattr(config, "num_encoder_blocks", None)
            or 0
        )

        # rope_interleave depends on model_type / qk_rope_head_dim
        rope_interleave = getattr(
            config,
            "rope_interleave",
            (getattr(config, "qk_rope_head_dim", None) or 0) > 0
            or model_type in ("glm", "glm4", "glm4_moe", "chatglm"),
        )
        rope_config = dataclasses.replace(rope_config, rope_interleave=rope_interleave)

        options = dict(
            head_dim=(
                config.head_dim
                if (hasattr(config, "head_dim") and config.head_dim is not None)
                else getattr(config, "d_kv", None)
                or _as_int(hidden_size) // _as_int(num_attention_heads)
            ),
            num_attention_heads=_as_int(num_attention_heads),
            num_key_value_heads=_as_int(
                getattr(config, "num_key_value_heads", num_attention_heads)
            ),
            num_hidden_layers=_as_int(num_hidden_layers),
            vocab_size=getattr(config, "vocab_size", None) or 0,
            hidden_size=_as_int(hidden_size),
            intermediate_size=(
                getattr(config, "intermediate_size", None)
                or getattr(config, "n_inner", None)
                or getattr(config, "d_ff", None)
                or getattr(config, "ffn_dim", None)
                or getattr(config, "ffn_hidden_size", None)
                or getattr(config, "encoder_ffn_dim", None)
                or getattr(config, "decoder_ffn_dim", None)
                or 4 * _as_int(hidden_size)
            ),
            hidden_act=(
                getattr(config, "hidden_act", None)
                or getattr(config, "hidden_activation", None)
                or getattr(config, "activation_function", None)
                or getattr(config, "dense_act_fn", None)
                or getattr(config, "activation", None)
                # Qwen v1 configs have no activation attr; default to silu
                or ("silu" if model_type in ("qwen",) else None)
            ),
            layer_types=(getattr(config, "layer_types", None)),
            full_attention_interval=(getattr(config, "full_attention_interval", None)),
            sliding_window=(getattr(config, "sliding_window", None)),
            # Linear attention (DeltaNet) parameters
            linear_conv_kernel_dim=(getattr(config, "linear_conv_kernel_dim", 4)),
            linear_key_head_dim=(getattr(config, "linear_key_head_dim", None)),
            linear_value_head_dim=(getattr(config, "linear_value_head_dim", None)),
            linear_num_key_heads=(getattr(config, "linear_num_key_heads", None)),
            linear_num_value_heads=(getattr(config, "linear_num_value_heads", None)),
            pad_token_id=(getattr(config, "pad_token_id", 0)),
            rms_norm_eps=(
                getattr(config, "rms_norm_eps", None)
                or getattr(config, "layer_norm_eps", None)
                or getattr(config, "layer_norm_epsilon", None)
                or getattr(config, "norm_epsilon", None)
                or getattr(config, "norm_eps", None)
                or 1e-6
            ),
            attn_qkv_bias=(
                getattr(
                    config,
                    "attention_bias",
                    getattr(
                        config,
                        "enable_bias",
                        getattr(
                            config,
                            "bias",
                            model_type
                            in (
                                "gpt2",
                                "bloom",
                                "qwen2",
                                "qwen2_5_vl_text",
                                "qwen2_moe",
                            ),
                        ),
                    ),
                )
            ),
            attn_o_bias=(
                getattr(
                    config,
                    "attention_bias",
                    getattr(
                        config,
                        "enable_bias",
                        getattr(
                            config,
                            "bias",
                            model_type in ("gpt2", "bloom"),
                        ),
                    ),
                )
            ),
            attn_qk_norm=(
                model_type
                in (
                    "gemma3_text",
                    "flex_olmo",
                    "olmoe",
                    "olmo2",
                    "olmo3",
                    "qwen3",
                    "qwen3_moe",
                    "qwen3_tts_talker",
                    "qwen3_5_vl",
                    "qwen3_vl",
                    "qwen3_vl_text",
                )
                or getattr(config, "use_qk_norm", False)
            ),
            attn_qk_norm_full=(model_type in ("flex_olmo", "olmoe", "olmo2", "olmo3")),
            mlp_bias=(getattr(config, "use_mlp_bias", False)),
            rope=rope_config,
            # Set flat rope fields for direct access by components
            rope_type=rope_config.rope_type,
            rope_theta=rope_config.rope_theta,
            rope_scaling=rope_config.rope_scaling,
            partial_rotary_factor=rope_config.partial_rotary_factor,
            rope_local_base_freq=rope_config.rope_local_base_freq,
            original_max_position_embeddings=rope_config.original_max_position_embeddings,
            rope_interleave=rope_config.rope_interleave,
            **mrope_fields,
            max_position_embeddings=getattr(config, "max_position_embeddings", 0),
            tie_word_embeddings=(
                getattr(config, "tie_word_embeddings", None)
                if getattr(config, "tie_word_embeddings", None) is not None
                else getattr(parent_config, "tie_word_embeddings", False)
            ),
            # MoE
            num_local_experts=(
                getattr(config, "num_local_experts", None)
                or getattr(config, "num_experts", None)
                or getattr(config, "n_routed_experts", None)
            ),
            num_experts_per_tok=(getattr(config, "num_experts_per_tok", None)),
            moe_intermediate_size=(getattr(config, "moe_intermediate_size", None)),
            shared_expert_intermediate_size=(
                getattr(config, "shared_expert_intermediate_size", None)
            ),
            norm_topk_prob=(getattr(config, "norm_topk_prob", True)),
            post_feedforward_norm=(model_type in ("flex_olmo",)),
            n_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            scoring_func=getattr(config, "scoring_func", "softmax"),
            topk_method=getattr(config, "topk_method", "greedy"),
            first_k_dense_replace=getattr(config, "first_k_dense_replace", 0),
            n_shared_experts=getattr(config, "n_shared_experts", None),
            # Multi-head Latent Attention (MLA)
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=getattr(config, "kv_lora_rank", None),
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", None),
            v_head_dim=getattr(config, "v_head_dim", None),
            # Encoder-specific
            type_vocab_size=getattr(config, "type_vocab_size", 0),
            # Encoder-decoder
            num_decoder_layers=(
                getattr(config, "num_decoder_layers", None)
                or getattr(config, "decoder_layers", None)
            ),
            relative_attention_num_buckets=getattr(
                config, "relative_attention_num_buckets", 32
            ),
            relative_attention_max_distance=getattr(
                config, "relative_attention_max_distance", 128
            ),
            is_gated_act=getattr(config, "is_gated_act", False),
            scale_decoder_outputs=getattr(config, "scale_decoder_outputs", None),
            # Standalone vision (coerce list to int — some HF configs use [H, W])
            image_size=_as_int(getattr(config, "image_size", 224)),
            patch_size=_as_int(getattr(config, "patch_size", 16)),
            num_channels=getattr(config, "num_channels", 3),
            # Granite scaling multipliers
            embedding_multiplier=getattr(config, "embedding_multiplier", 1.0),
            attention_multiplier=getattr(config, "attention_multiplier", None),
            logits_scaling=getattr(config, "logits_scaling", 1.0),
            residual_multiplier=getattr(config, "residual_multiplier", 1.0),
            # Cohere logit scale
            logit_scale=getattr(config, "logit_scale", 1.0),
            # Falcon config
            alibi=getattr(config, "alibi", False),
            parallel_attn=getattr(config, "parallel_attn", False),
        )

        # Falcon/Bloom model-specific overrides
        if model_type in ("falcon", "falcon_h1"):
            # Falcon MQA: multi_query=True with old architecture → 1 KV head
            if getattr(config, "multi_query", False) and not getattr(
                config, "new_decoder_architecture", False
            ):
                options["num_key_value_heads"] = 1
            # Falcon uses config.bias for both attention and MLP
            options["mlp_bias"] = getattr(config, "bias", False)
        elif model_type == "bloom":
            options["alibi"] = True
            options["mlp_bias"] = True

        # Convert rotary_dim to partial_rotary_factor (GPT-J, CodeGen, etc.)
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is not None and options["head_dim"] > 0:
            options["partial_rotary_factor"] = rotary_dim / options["head_dim"]
            rope_config = dataclasses.replace(
                rope_config,
                partial_rotary_factor=options["partial_rotary_factor"],
            )
            options["rope"] = rope_config

        # Compute layer_types from full_attention_interval if not provided
        if options.get("layer_types") is None:
            full_attention_interval = options.get("full_attention_interval")
            if full_attention_interval is not None:
                num_hidden_layers = options["num_hidden_layers"]
                layer_types = []
                for i in range(num_hidden_layers):
                    if (i + 1) % full_attention_interval == 0:
                        layer_types.append("full_attention")
                    else:
                        layer_types.append("linear_attention")
                options["layer_types"] = layer_types

        # Vision config (from multimodal models)
        options.update(_extract_vision_config(config, parent_config, model_type))

        # Audio config
        options.update(_extract_audio_config(config, parent_config, model_type))

        # CLAP contrastive projection config
        if model_type in ("clap", "clap_text_model", "clap_audio_model"):
            clap_src = parent_config or config
            options["projection_dim"] = getattr(clap_src, "projection_dim", 512)

        # CLIPSeg segmentation decoder config (fields on parent_config)
        seg_source = parent_config or config
        if getattr(seg_source, "reduce_dim", None) is not None:
            options["projection_dim"] = getattr(seg_source, "projection_dim", 512)
            options["reduce_dim"] = getattr(seg_source, "reduce_dim", 64)
            options["extract_layers"] = getattr(seg_source, "extract_layers", [3, 6, 9])
            options["conditional_layer"] = getattr(seg_source, "conditional_layer", 0)
            options["decoder_num_attention_heads"] = getattr(
                seg_source, "decoder_num_attention_heads", 4
            )
            options["decoder_intermediate_size"] = getattr(
                seg_source, "decoder_intermediate_size", 2048
            )
            options["decoder_hidden_act"] = getattr(
                seg_source, "decoder_hidden_act", "quick_gelu"
            )

        # Q-Former config (for BLIP-2 style models)
        qformer_source = parent_config or config
        hf_qformer_config = getattr(qformer_source, "qformer_config", None)
        if hf_qformer_config is not None:
            qc = (
                hf_qformer_config
                if not isinstance(hf_qformer_config, dict)
                else type("QC", (), hf_qformer_config)()
            )
            options["num_query_tokens"] = getattr(qformer_source, "num_query_tokens", 32)
            options["qformer_hidden_size"] = getattr(qc, "hidden_size", 768)
            options["qformer_num_hidden_layers"] = getattr(qc, "num_hidden_layers", 12)
            options["qformer_num_attention_heads"] = getattr(qc, "num_attention_heads", 12)
            options["qformer_intermediate_size"] = getattr(qc, "intermediate_size", 3072)

        # TTS config (Qwen3-TTS talker + code predictor + speaker encoder)
        tts_source = parent_config or config
        talker_cfg = getattr(tts_source, "talker_config", None)
        if talker_cfg is not None:
            tc = (
                talker_cfg
                if not isinstance(talker_cfg, dict)
                else type("TC", (), talker_cfg)()
            )
            tts_fields: dict = {}
            tts_fields["text_hidden_size"] = getattr(tc, "text_hidden_size", 2048)
            tts_fields["text_vocab_size"] = getattr(tc, "text_vocab_size", 151936)
            tts_fields["num_code_groups"] = getattr(tc, "num_code_groups", 16)
            tts_fields["codec_bos_id"] = getattr(tc, "codec_bos_id", 2149)
            tts_fields["codec_eos_token_id"] = getattr(tc, "codec_eos_token_id", 2150)
            tts_fields["codec_pad_id"] = getattr(tc, "codec_pad_id", 2148)
            tts_fields["codec_think_id"] = getattr(tc, "codec_think_id", 2154)
            tts_fields["codec_nothink_id"] = getattr(tc, "codec_nothink_id", 2155)

            # Code predictor config
            cp_cfg = getattr(tc, "code_predictor_config", None)
            if cp_cfg is not None:
                cp = cp_cfg if not isinstance(cp_cfg, dict) else type("CP", (), cp_cfg)()
                tts_fields["code_predictor"] = CodePredictorConfig(
                    hidden_size=getattr(cp, "hidden_size", 1024),
                    intermediate_size=getattr(cp, "intermediate_size", 3072),
                    num_hidden_layers=getattr(cp, "num_hidden_layers", 5),
                    num_attention_heads=getattr(cp, "num_attention_heads", 16),
                    num_key_value_heads=getattr(cp, "num_key_value_heads", 8),
                    head_dim=getattr(cp, "head_dim", 128),
                    vocab_size=getattr(cp, "vocab_size", 2048),
                    num_code_groups=getattr(cp, "num_code_groups", 16),
                    rms_norm_eps=getattr(cp, "rms_norm_eps", 1e-6),
                    rope_theta=getattr(cp, "rope_theta", 1_000_000.0),
                    hidden_act=getattr(cp, "hidden_act", "silu"),
                    layer_types=getattr(cp, "layer_types", None),
                )

            # Speaker encoder config
            se_cfg = getattr(tts_source, "speaker_encoder_config", None)
            if se_cfg is not None:
                se = se_cfg if not isinstance(se_cfg, dict) else type("SE", (), se_cfg)()
                tts_fields["speaker_encoder"] = SpeakerEncoderConfig(
                    mel_dim=getattr(se, "mel_dim", 128),
                    enc_dim=getattr(se, "enc_dim", 1024),
                    enc_channels=getattr(se, "enc_channels", [512, 512, 512, 512, 1536]),
                    enc_kernel_sizes=getattr(se, "enc_kernel_sizes", [5, 3, 3, 3, 1]),
                    enc_dilations=getattr(se, "enc_dilations", [1, 2, 3, 4, 1]),
                    enc_attention_channels=getattr(se, "enc_attention_channels", 128),
                    enc_res2net_scale=getattr(se, "enc_res2net_scale", 8),
                    enc_se_channels=getattr(se, "enc_se_channels", 128),
                )

            options["tts"] = TTSConfig(**tts_fields)

        # Codec tokenizer config (Qwen3-TTS-Tokenizer-12Hz)
        codec_source = parent_config or config
        hf_decoder_cfg = getattr(codec_source, "decoder_config", None)
        hf_encoder_cfg = getattr(codec_source, "encoder_config", None)
        if hf_decoder_cfg is not None and model_type == "qwen3_tts_tokenizer_12hz":
            dc = (
                hf_decoder_cfg
                if not isinstance(hf_decoder_cfg, dict)
                else type("DC", (), hf_decoder_cfg)()
            )
            options["codec_decoder"] = CodecDecoderConfig(
                codebook_dim=getattr(dc, "codebook_dim", 512),
                codebook_size=getattr(dc, "codebook_size", 2048),
                latent_dim=getattr(dc, "latent_dim", 1024),
                hidden_size=getattr(dc, "hidden_size", 512),
                intermediate_size=getattr(dc, "intermediate_size", 1024),
                num_hidden_layers=getattr(dc, "num_hidden_layers", 8),
                num_attention_heads=getattr(dc, "num_attention_heads", 16),
                num_key_value_heads=getattr(dc, "num_key_value_heads", 16),
                head_dim=getattr(dc, "head_dim", 64),
                rms_norm_eps=getattr(dc, "rms_norm_eps", 1e-5),
                rope_theta=getattr(dc, "rope_theta", 10000.0),
                max_position_embeddings=getattr(dc, "max_position_embeddings", 8000),
                decoder_dim=getattr(dc, "decoder_dim", 1536),
                num_quantizers=getattr(dc, "num_quantizers", 16),
                upsample_rates=getattr(dc, "upsample_rates", [8, 5, 4, 3]),
                upsampling_ratios=getattr(dc, "upsampling_ratios", [2, 2]),
            )
        if hf_encoder_cfg is not None and model_type == "qwen3_tts_tokenizer_12hz":
            ec = (
                hf_encoder_cfg
                if not isinstance(hf_encoder_cfg, dict)
                else type("EC", (), hf_encoder_cfg)()
            )
            options["codec_encoder"] = CodecEncoderConfig(
                codebook_dim=getattr(ec, "codebook_dim", 256),
                codebook_size=getattr(ec, "codebook_size", 2048),
                hidden_size=getattr(ec, "hidden_size", 512),
                intermediate_size=getattr(ec, "intermediate_size", 2048),
                num_hidden_layers=getattr(ec, "num_hidden_layers", 8),
                num_attention_heads=getattr(ec, "num_attention_heads", 8),
                num_key_value_heads=getattr(ec, "num_key_value_heads", 8),
                head_dim=getattr(ec, "head_dim", 64),
                rope_theta=getattr(ec, "rope_theta", 10000.0),
                max_position_embeddings=getattr(ec, "max_position_embeddings", 8000),
                num_quantizers=getattr(ec, "num_quantizers", 32),
                num_semantic_quantizers=getattr(ec, "num_semantic_quantizers", 1),
            )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        # Quantization config
        quant = QuantizationConfig.from_transformers(config)
        if quant is None and parent_config is not None:
            quant = QuantizationConfig.from_transformers(parent_config)
        if quant is not None:
            options["quantization"] = quant

        return cls(**options)

    @classmethod
    def from_file(cls, path: str, parent_config=None) -> ArchitectureConfig:
        """Create config from a local model directory or config.json file.

        Args:
            path: Path to a directory containing ``config.json``, or a
                direct path to a JSON config file.
            parent_config: Optional parent HF config (for composite models).

        Returns:
            An ``ArchitectureConfig`` instance.
        """
        import transformers

        config = transformers.AutoConfig.from_pretrained(path)
        hf_config = config
        if parent_config is None:
            parent_config = config
        if hasattr(config, "text_config"):
            hf_config = config.text_config
        elif hasattr(config, "language_config"):
            hf_config = config.language_config
        return cls.from_transformers(hf_config, parent_config=parent_config)

    def validate(self) -> None:
        """Validate config field consistency.

        Raises:
            ValueError: If the config has invalid or inconsistent fields.
        """
        errors: list[str] = []
        if self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            errors.append(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.vocab_size < 0:
            errors.append(f"vocab_size must be non-negative, got {self.vocab_size}")
        if self.head_dim <= 0:
            errors.append(f"head_dim must be positive, got {self.head_dim}")
        if (
            self.num_key_value_heads > 0
            and self.num_attention_heads % self.num_key_value_heads != 0
        ):
            errors.append(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if (
            self.hidden_size > 0
            and self.num_attention_heads > 0
            and self.head_dim == DEFAULT_INT
            and self.hidden_size % self.num_attention_heads != 0
        ):
            errors.append(
                f"hidden_size ({self.hidden_size}) must be "
                f"divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive, got {self.intermediate_size}")
        if errors:
            raise ValueError(
                "Invalid ArchitectureConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def _shallow_fields(config) -> dict:
    """Extract fields from a dataclass without recursive conversion.

    Unlike ``dataclasses.asdict()``, this preserves nested dataclass
    instances (:class:`VisionConfig`, :class:`AudioConfig`, etc.) as-is.
    """
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


def _as_int(value) -> int:
    """Coerce *value* to int, taking the first element if it is a list/tuple.

    Some HuggingFace configs express ``image_size`` or ``patch_size`` as
    ``[H, W]`` lists.  We take the first element (height) for simplicity.
    """
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _first(value):
    """Return the first element of a list/tuple, or *value* unchanged."""
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


# ---------------------------------------------------------------------------
# Category subclasses — type markers for model categories
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CausalLMConfig(ArchitectureConfig):
    """Configuration for decoder-only causal language models.

    Used by Llama, Mistral, Qwen, GPT-2, and similar architectures.
    Inherits all shared transformer fields from :class:`ArchitectureConfig`.
    """


@dataclasses.dataclass
class EncoderConfig(ArchitectureConfig):
    """Configuration for encoder-only models (BERT, ViT, etc.)."""


@dataclasses.dataclass
class VisionLanguageConfig(CausalLMConfig):
    """Configuration for vision-language models (LLaVA, Qwen-VL, etc.).

    Inherits :class:`CausalLMConfig` for the text decoder component.
    Vision-specific fields live in the :class:`VisionConfig` sub-config.
    """


# ---------------------------------------------------------------------------
# Model-family subclasses — add model-specific fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Gemma2Config(CausalLMConfig):
    """Configuration for Gemma2 models with attention soft-capping.

    Adds ``attn_logit_softcapping``, ``final_logit_softcapping``, and
    ``query_pre_attn_scalar`` used exclusively by :mod:`models.gemma`.
    """

    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 0.0
    query_pre_attn_scalar: float | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            attn_logit_softcapping=(getattr(config, "attn_logit_softcapping", 0.0) or 0.0),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
            query_pre_attn_scalar=getattr(config, "query_pre_attn_scalar", None),
        )


@dataclasses.dataclass
class NanoChatConfig(CausalLMConfig):
    """Configuration for NanoChat models with final logit soft-capping.

    Adds ``final_logit_softcapping`` used by :mod:`models.nanochat`.
    """

    final_logit_softcapping: float = 0.0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NanoChatConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
        )


@dataclasses.dataclass
class LongcatFlashConfig(CausalLMConfig):
    """Configuration for LongCat Flash dual-sublayer models.

    Adds ``zero_expert_num`` for identity/pass-through MoE experts.
    Unlike standard MoE, LongCat uses a fixed shortcut MoE block per
    physical layer alongside two dense sub-attentions and two dense MLPs.
    """

    zero_expert_num: int = 0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> LongcatFlashConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # LongCat uses ffn_hidden_size for dense MLP (not the generic intermediate_size)
        ffn_hidden_size = getattr(config, "ffn_hidden_size", None)
        if ffn_hidden_size is not None:
            base = dataclasses.replace(base, intermediate_size=ffn_hidden_size)
        # LongCat uses moe_topk (not num_experts_per_tok)
        moe_topk = getattr(config, "moe_topk", None)
        if moe_topk is not None:
            base = dataclasses.replace(base, num_experts_per_tok=moe_topk)
        # LongCat uses expert_ffn_hidden_size (not moe_intermediate_size)
        expert_ffn_hidden_size = getattr(config, "expert_ffn_hidden_size", None)
        if expert_ffn_hidden_size is not None:
            base = dataclasses.replace(base, moe_intermediate_size=expert_ffn_hidden_size)
        return cls(
            **_shallow_fields(base),
            zero_expert_num=getattr(config, "zero_expert_num", 0),
        )


@dataclasses.dataclass
class Gemma3nConfig(CausalLMConfig):
    """Configuration for Gemma3n models with AltUp and Laurel compression.

    Adds AltUp prediction/correction parameters and per-layer input
    dimension fields used exclusively by :mod:`models.gemma3n`.
    """

    altup_num_inputs: int = 4
    altup_active_idx: int = 0
    altup_correct_scale: bool = True
    laurel_rank: int = 64
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262_144

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma3nConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            altup_num_inputs=getattr(config, "altup_num_inputs", 4),
            altup_active_idx=getattr(config, "altup_active_idx", 0),
            altup_correct_scale=getattr(config, "altup_correct_scale", True),
            laurel_rank=getattr(config, "laurel_rank", 64),
            hidden_size_per_layer_input=getattr(config, "hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input=getattr(config, "vocab_size_per_layer_input", 262_144),
        )


@dataclasses.dataclass
class MllamaConfig(VisionLanguageConfig):
    """Configuration for Mllama (Llama 3.2 Vision) cross-attention models.

    Adds ``cross_attention_layers`` specifying which decoder layers use
    cross-attention with vision features.
    """

    cross_attention_layers: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MllamaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            cross_attention_layers=getattr(config, "cross_attention_layers", None),
        )


@dataclasses.dataclass
class YolosConfig(EncoderConfig):
    """Configuration for YOLOS object detection models.

    Adds ``num_detection_tokens`` for the learned detection token count.
    ``num_labels`` remains on :class:`ArchitectureConfig` (shared with
    :class:`SegformerConfig`).
    """

    num_detection_tokens: int = 100

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> YolosConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            num_detection_tokens=getattr(config, "num_detection_tokens", 100),
        )


@dataclasses.dataclass
class DepthAnythingConfig(ArchitectureConfig):
    """Configuration for Depth Anything DPT depth estimation models.

    Adds DPT neck, reassembly, and fusion head parameters.
    """

    neck_hidden_sizes: list[int] | None = None
    reassemble_factors: list[float] | None = None
    fusion_hidden_size: int = 64
    head_hidden_size: int = 32
    backbone_out_indices: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> DepthAnythingConfig:
        # HF DepthAnythingConfig stores backbone (ViT) fields on a nested
        # backbone_config (e.g. Dinov2Config).  Extract it so the base
        # ArchitectureConfig resolver can find hidden_size, num_heads, etc.
        backbone = getattr(config, "backbone_config", config)
        base = ArchitectureConfig.from_transformers(backbone, parent_config)
        return cls(
            **_shallow_fields(base),
            neck_hidden_sizes=getattr(config, "neck_hidden_sizes", None),
            reassemble_factors=getattr(config, "reassemble_factors", None),
            fusion_hidden_size=getattr(config, "fusion_hidden_size", 64),
            head_hidden_size=getattr(config, "head_hidden_size", 32),
            # backbone_out_indices lives on the backbone config as out_indices
            backbone_out_indices=(
                getattr(backbone, "out_indices", None)
                or getattr(config, "backbone_out_indices", None)
            ),
        )


@dataclasses.dataclass
class ZoeDepthConfig(ArchitectureConfig):
    """Configuration for ZoeDepth monocular depth estimation models.

    Extends DepthAnythingConfig-style fields with ZoeDepth-specific
    bin head parameters (attractors, temperature, etc.).
    """

    neck_hidden_sizes: list[int] | None = None
    reassemble_factors: list[float] | None = None
    fusion_hidden_size: int = 256
    backbone_out_indices: list[int] | None = None
    bottleneck_features: int = 256
    bin_configurations: list[dict] | None = None
    num_attractors: list[int] | None = None
    bin_embedding_dim: int = 128
    num_relative_features: int = 32
    min_temp: float = 0.0212
    max_temp: float = 50.0
    head_in_index: int = -1

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ZoeDepthConfig:
        backbone = getattr(config, "backbone_config", config)
        base = ArchitectureConfig.from_transformers(backbone, parent_config)
        return cls(
            **_shallow_fields(base),
            neck_hidden_sizes=getattr(config, "neck_hidden_sizes", None),
            reassemble_factors=getattr(config, "reassemble_factors", None),
            fusion_hidden_size=getattr(config, "fusion_hidden_size", 256),
            backbone_out_indices=(
                getattr(backbone, "out_indices", None)
                or getattr(config, "backbone_out_indices", None)
            ),
            bottleneck_features=getattr(config, "bottleneck_features", 256),
            bin_configurations=getattr(config, "bin_configurations", None),
            num_attractors=getattr(config, "num_attractors", None),
            bin_embedding_dim=getattr(config, "bin_embedding_dim", 128),
            num_relative_features=getattr(config, "num_relative_features", 32),
            min_temp=getattr(config, "min_temp", 0.0212),
            max_temp=getattr(config, "max_temp", 50.0),
            head_in_index=getattr(config, "head_in_index", -1),
        )


@dataclasses.dataclass
class SegformerConfig(EncoderConfig):
    """Configuration for Segformer hierarchical vision transformers.

    Adds per-stage encoder parameters and decode head hidden size.
    ``num_labels`` remains on :class:`ArchitectureConfig`.
    """

    segformer_hidden_sizes: list[int] | None = None
    segformer_num_attention_heads: list[int] | None = None
    segformer_depths: list[int] | None = None
    segformer_sr_ratios: list[int] | None = None
    segformer_mlp_ratios: list[int] | None = None
    segformer_patch_sizes: list[int] | None = None
    segformer_strides: list[int] | None = None
    decoder_hidden_size: int = 256

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> SegformerConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            segformer_hidden_sizes=getattr(config, "segformer_hidden_sizes", None),
            segformer_num_attention_heads=getattr(
                config, "segformer_num_attention_heads", None
            ),
            segformer_depths=getattr(config, "segformer_depths", None),
            segformer_sr_ratios=getattr(config, "segformer_sr_ratios", None),
            segformer_mlp_ratios=getattr(config, "segformer_mlp_ratios", None),
            segformer_patch_sizes=getattr(config, "segformer_patch_sizes", None),
            segformer_strides=getattr(config, "segformer_strides", None),
            decoder_hidden_size=getattr(config, "decoder_hidden_size", 256),
        )


@dataclasses.dataclass
class Sam2Config(ArchitectureConfig):
    """Configuration for SAM2 Hiera backbone vision models.

    Adds Hiera-specific per-stage dimensions, block counts, and FPN
    parameters.
    """

    sam2_embed_dims: list[int] | None = None
    sam2_blocks_per_stage: list[int] | None = None
    sam2_num_heads_per_stage: list[int] | None = None
    sam2_mlp_ratio: float | None = None
    sam2_fpn_hidden_size: int | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Sam2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            sam2_embed_dims=getattr(config, "sam2_embed_dims", None),
            sam2_blocks_per_stage=getattr(config, "sam2_blocks_per_stage", None),
            sam2_num_heads_per_stage=getattr(config, "sam2_num_heads_per_stage", None),
            sam2_mlp_ratio=getattr(config, "sam2_mlp_ratio", None),
            sam2_fpn_hidden_size=getattr(config, "sam2_fpn_hidden_size", None),
        )


@dataclasses.dataclass
class DetrConfig(ArchitectureConfig):
    """Configuration for DETR (DEtection TRansformer) object detection models.

    DETR uses a ResNet-50 backbone (timm-style weight naming) with a
    transformer encoder-decoder.  Backbone fields are embedded directly
    to avoid an extra nested config object.
    """

    d_model: int = 256
    num_queries: int = 100
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    # ResNet-50 backbone defaults
    backbone_embedding_size: int = 64
    backbone_hidden_sizes: list[int] = dataclasses.field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    backbone_depths: list[int] = dataclasses.field(default_factory=lambda: [3, 4, 6, 3])
    backbone_layer_type: str = "bottleneck"

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> DetrConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            d_model=getattr(config, "d_model", 256),
            num_queries=getattr(config, "num_queries", 100),
            encoder_layers=getattr(config, "encoder_layers", 6),
            decoder_layers=getattr(config, "decoder_layers", 6),
            encoder_attention_heads=getattr(config, "encoder_attention_heads", 8),
            decoder_attention_heads=getattr(config, "decoder_attention_heads", 8),
            encoder_ffn_dim=getattr(config, "encoder_ffn_dim", 2048),
            decoder_ffn_dim=getattr(config, "decoder_ffn_dim", 2048),
            # Backbone is a fixed ResNet-50 (timm) — no nested config
            backbone_embedding_size=64,
            backbone_hidden_sizes=[256, 512, 1024, 2048],
            backbone_depths=[3, 4, 6, 3],
            backbone_layer_type="bottleneck",
        )


@dataclasses.dataclass
class RtDetrConfig(ArchitectureConfig):
    """Configuration for RT-DETR real-time object detection models.

    RT-DETR uses a ResNet backbone with a hybrid encoder (FPN/PAN/AIFI)
    and a transformer decoder with multi-scale deformable attention.
    """

    d_model: int = 256
    num_queries: int = 300
    encoder_layers: int = 1
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024
    decoder_n_points: int = 4
    num_feature_levels: int = 3
    positional_encoding_temperature: float = 10000.0
    # Backbone (RT-DETR ResNet variant)
    backbone_embedding_size: int = 64
    backbone_hidden_sizes: list[int] = dataclasses.field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    backbone_depths: list[int] = dataclasses.field(
        default_factory=lambda: [3, 4, 6, 3]
    )
    backbone_out_indices: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 3]
    )
    encoder_in_channels: list[int] = dataclasses.field(
        default_factory=lambda: [512, 1024, 2048]
    )
    feat_strides: list[int] = dataclasses.field(
        default_factory=lambda: [8, 16, 32]
    )

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> RtDetrConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # Extract backbone config (RTDetrResNetConfig object)
        bc = getattr(config, "backbone_config", None)
        if bc is not None and hasattr(bc, "hidden_sizes"):
            b_hidden = list(bc.hidden_sizes)
            b_depths = list(bc.depths)
            b_embed = getattr(bc, "embedding_size", 64)
            b_out_idx = list(getattr(bc, "out_indices", [2, 3, 4]))
            # Convert 1-indexed HF out_indices to 0-indexed stages
            b_out_idx = [i - 1 for i in b_out_idx]
        else:
            b_hidden = [256, 512, 1024, 2048]
            b_depths = [3, 4, 6, 3]
            b_embed = 64
            b_out_idx = [1, 2, 3]

        return cls(
            **_shallow_fields(base),
            d_model=getattr(config, "d_model", 256),
            num_queries=getattr(config, "num_queries", 300),
            encoder_layers=getattr(config, "encoder_layers", 1),
            decoder_layers=getattr(config, "decoder_layers", 6),
            encoder_attention_heads=getattr(config, "encoder_attention_heads", 8),
            decoder_attention_heads=getattr(config, "decoder_attention_heads", 8),
            encoder_ffn_dim=getattr(config, "encoder_ffn_dim", 1024),
            decoder_ffn_dim=getattr(config, "decoder_ffn_dim", 1024),
            decoder_n_points=getattr(config, "decoder_n_points", 4),
            num_feature_levels=getattr(config, "num_feature_levels", 3),
            positional_encoding_temperature=getattr(
                config, "positional_encoding_temperature", 10000.0
            ),
            backbone_embedding_size=b_embed,
            backbone_hidden_sizes=b_hidden,
            backbone_depths=b_depths,
            backbone_out_indices=b_out_idx,
            encoder_in_channels=list(
                getattr(config, "encoder_in_channels", [512, 1024, 2048])
            ),
            feat_strides=list(
                getattr(config, "feat_strides", [8, 16, 32])
            ),
        )


@dataclasses.dataclass
class ResNetConfig(ArchitectureConfig):
    """Configuration for ResNet CNN backbone models.

    Adds ResNet-specific per-stage depths, channel sizes, and block type.
    """

    embedding_size: int = 64
    hidden_sizes: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024, 2048])
    depths: list[int] = dataclasses.field(default_factory=lambda: [3, 4, 6, 3])
    layer_type: str = "bottleneck"
    downsample_in_bottleneck: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ResNetConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            embedding_size=getattr(config, "embedding_size", 64),
            hidden_sizes=getattr(config, "hidden_sizes", [256, 512, 1024, 2048]),
            depths=getattr(config, "depths", [3, 4, 6, 3]),
            layer_type=getattr(config, "layer_type", "bottleneck"),
            downsample_in_bottleneck=getattr(config, "downsample_in_bottleneck", False),
        )


@dataclasses.dataclass
class ConvNextConfig(ArchitectureConfig):
    """Configuration for ConvNeXT CNN backbone models.

    ConvNeXT modernizes ResNet: depth-wise conv, LayerNorm, GELU,
    inverted bottleneck, per-channel layer scale.
    """

    hidden_sizes: list[int] = dataclasses.field(default_factory=lambda: [96, 192, 384, 768])
    depths: list[int] = dataclasses.field(default_factory=lambda: [3, 3, 9, 3])
    layer_scale_init_value: float = 1e-6

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ConvNextConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            hidden_sizes=getattr(config, "hidden_sizes", [96, 192, 384, 768]),
            depths=getattr(config, "depths", [3, 3, 9, 3]),
            layer_scale_init_value=getattr(config, "layer_scale_init_value", 1e-6),
        )


@dataclasses.dataclass
class MoondreamConfig(ArchitectureConfig):
    """Configuration for Moondream vision-language models.

    Moondream uses a custom config format with nested text/vision sub-configs.
    The HF config is nearly empty — defaults are hardcoded from
    ``vikhyatk/moondream2``'s custom Python code.
    """

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MoondreamConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # Moondream's HF config stores params in a nested 'config' dict
        cfg = getattr(config, "config", None) or {}
        text_cfg = cfg.get("text", {})
        vision_cfg = cfg.get("vision", {})

        text_dim = text_cfg.get("dim", 2048)
        n_heads = text_cfg.get("n_heads", 32)
        enc_dim = vision_cfg.get("enc_dim", 1152)
        enc_ff_dim = vision_cfg.get("enc_ff_dim", 4304)
        enc_n_layers = vision_cfg.get("enc_n_layers", 27)
        enc_n_heads = vision_cfg.get("enc_n_heads", 16)
        crop_size = vision_cfg.get("crop_size", 378)
        patch_size = vision_cfg.get("enc_patch_size", 14)
        proj_inner_dim = vision_cfg.get("proj_inner_dim", 8192)

        return cls(
            **_shallow_fields(base),
            hidden_size=text_dim,
            intermediate_size=text_cfg.get("ff_dim", 8192),
            num_hidden_layers=text_cfg.get("n_layers", 24),
            vocab_size=text_cfg.get("vocab_size", 51200),
            num_attention_heads=n_heads,
            num_key_value_heads=text_cfg.get("n_kv_heads", 32),
            head_dim=text_dim // n_heads,
            hidden_act="gelu_tanh",
            rms_norm_eps=1e-5,
            attn_qkv_bias=True,
            attn_o_bias=True,
            max_position_embeddings=text_cfg.get("max_context", 2048),
            image_token_id=50256,  # Placeholder for ORT GenAI pipeline
            vision=VisionConfig(
                hidden_size=enc_dim,
                intermediate_size=enc_ff_dim,
                num_hidden_layers=enc_n_layers,
                num_attention_heads=enc_n_heads,
                image_size=crop_size,
                patch_size=patch_size,
                norm_eps=1e-6,
            ),
        )


@dataclasses.dataclass
class MambaConfig(BaseModelConfig):
    """Configuration for Mamba SSM (Selective State Space) models.

    Mamba replaces transformer attention with a selective scan mechanism.
    Fields map to HuggingFace ``MambaConfig``.

    State carried per layer:
        conv_state: (batch, d_inner, conv_kernel - 1)
        ssm_state:  (batch, d_inner, state_size)
    """

    state_size: int = 16
    conv_kernel: int = 4
    expand: int = 2
    time_step_rank: int = 48
    layer_norm_epsilon: float = 1e-5
    use_conv_bias: bool = True
    residual_in_fp32: bool = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MambaConfig:
        del parent_config  # unused
        expand = getattr(config, "expand", 2)
        d_inner = getattr(config, "intermediate_size", 0)
        if not d_inner:
            d_inner = config.hidden_size * expand

        tr = getattr(config, "time_step_rank", 48)
        if tr == "auto":
            import math

            tr = math.ceil(config.hidden_size / 16)

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=d_inner,
            num_hidden_layers=config.num_hidden_layers,
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            state_size=getattr(config, "state_size", 16),
            conv_kernel=getattr(config, "conv_kernel", 4),
            expand=expand,
            time_step_rank=tr,
            layer_norm_epsilon=getattr(config, "layer_norm_epsilon", 1e-5),
            use_conv_bias=getattr(config, "use_conv_bias", True),
            residual_in_fp32=getattr(config, "residual_in_fp32", True),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class Mamba2Config(BaseModelConfig):
    """Configuration for standalone Mamba2/SSD models.

    Pure Mamba2 (no attention, no MLP). Each layer is:
        RMSNorm -> Mamba2Block -> residual add

    State per layer:
        conv_state: (batch, conv_dim, d_conv - 1)
        ssm_state:  (batch, num_heads, head_dim, state_size)

    HuggingFace reference: ``Mamba2Config``.
    """

    num_heads: int = 128
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    expand: int = 2
    layer_norm_epsilon: float = 1e-5
    use_conv_bias: bool = True
    norm_before_gate: bool = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Mamba2Config:
        del parent_config  # unused
        expand = getattr(config, "expand", 2)
        d_inner = getattr(config, "intermediate_size", 0)
        if not d_inner:
            d_inner = config.hidden_size * expand

        num_heads = getattr(config, "num_heads", 128)
        head_dim = getattr(config, "head_dim", "auto")
        if head_dim == "auto":
            if d_inner % num_heads != 0:
                raise ValueError(
                    f"Mamba2Config: d_inner ({d_inner}) must be divisible "
                    f"by num_heads ({num_heads}) to compute head_dim."
                )
            head_dim = d_inner // num_heads

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=d_inner,
            num_hidden_layers=config.num_hidden_layers,
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=getattr(config, "state_size", 128),
            n_groups=getattr(config, "n_groups", 8),
            conv_kernel=getattr(config, "conv_kernel", 4),
            expand=expand,
            layer_norm_epsilon=getattr(config, "layer_norm_epsilon", 1e-5),
            use_conv_bias=getattr(config, "use_conv_bias", True),
            norm_before_gate=getattr(config, "norm_before_gate", True),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class JambaConfig(ArchitectureConfig):
    """Configuration for Jamba hybrid SSM+Attention models.

    Jamba interleaves Mamba SSM layers with Transformer attention layers.
    Some layers use MoE (multiple expert MLPs) instead of dense MLP.

    Layer type selection:
        - Attention if ``(i - attn_layer_offset) % attn_layer_period == 0``
        - Mamba otherwise
        - MoE MLP if ``(i - expert_layer_offset) % expert_layer_period == 0``
        - Dense MLP otherwise
    """

    # Mamba SSM parameters
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # Layer interleaving
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    expert_layer_period: int = 2
    expert_layer_offset: int = 1

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> JambaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Build layer_types list for HybridCausalLMTask
        n = base.num_hidden_layers
        attn_period = getattr(config, "attn_layer_period", 8)
        attn_offset = getattr(config, "attn_layer_offset", 4)
        layer_types = []
        for i in range(n):
            if (i - attn_offset) % attn_period == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("mamba")

        num_experts = getattr(config, "num_experts", 16)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)

        # Exclude fields we set explicitly below to avoid duplicate keyword args
        _exclude = {"layer_types", "num_local_experts", "num_experts_per_tok"}
        base_fields = {k: v for k, v in _shallow_fields(base).items() if k not in _exclude}
        return cls(
            **base_fields,
            layer_types=layer_types,
            # HF uses "num_experts"; we use inherited "num_local_experts"
            num_local_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            mamba_d_state=getattr(config, "mamba_d_state", 16),
            mamba_d_conv=getattr(config, "mamba_d_conv", 4),
            mamba_expand=getattr(config, "mamba_expand", 2),
            mamba_dt_rank=getattr(config, "mamba_dt_rank", 256),
            mamba_conv_bias=getattr(config, "mamba_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
            attn_layer_period=attn_period,
            attn_layer_offset=attn_offset,
            expert_layer_period=getattr(config, "expert_layer_period", 2),
            expert_layer_offset=getattr(config, "expert_layer_offset", 1),
        )


@dataclasses.dataclass
class BambaConfig(ArchitectureConfig):
    """Configuration for Bamba hybrid Mamba2+Attention models.

    Uses multi-head Mamba2/SSD layers interleaved with attention layers.
    """

    mamba_n_heads: int = 128
    mamba_d_head: int = 64
    mamba_d_state: int = 256
    mamba_n_groups: int = 1
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> BambaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        n = base.num_hidden_layers
        attn_indices = set(getattr(config, "attn_layer_indices", None) or [])
        layers_block_type = getattr(config, "layers_block_type", None)

        layer_types: list[str] = []
        for i in range(n):
            if layers_block_type and i < len(layers_block_type):
                ltype = layers_block_type[i]
                layer_types.append("full_attention" if ltype == "attention" else "mamba2")
            elif i in attn_indices:
                layer_types.append("full_attention")
            else:
                layer_types.append("mamba2")

        mamba_expand = getattr(config, "mamba_expand", 2)
        d_inner = config.hidden_size * mamba_expand

        mamba_n_heads = getattr(config, "mamba_n_heads", 128)
        mamba_d_head = getattr(config, "mamba_d_head", "auto")
        if mamba_d_head == "auto":
            mamba_d_head = d_inner // mamba_n_heads

        # Exclude layer_types from base fields — we built it explicitly above
        base_fields = {k: v for k, v in _shallow_fields(base).items() if k != "layer_types"}
        return cls(
            **base_fields,
            layer_types=layer_types,
            mamba_n_heads=mamba_n_heads,
            mamba_d_head=mamba_d_head,
            mamba_d_state=getattr(config, "mamba_d_state", 256),
            mamba_n_groups=getattr(config, "mamba_n_groups", 1),
            mamba_d_conv=getattr(config, "mamba_d_conv", 4),
            mamba_expand=mamba_expand,
            mamba_conv_bias=getattr(config, "mamba_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
        )


@dataclasses.dataclass
class GraniteMoeHybridConfig(BambaConfig):
    """Configuration for GraniteMoeHybrid: Mamba2+Attention hybrid with MoE on all layers.

    Extends BambaConfig with ``shared_intermediate_size`` for the dense shared MLP
    that runs alongside the routed MoE block on every layer.
    """

    shared_intermediate_size: int = 1024

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> GraniteMoeHybridConfig:
        # Reuse BambaConfig.from_transformers for mamba fields + layer_types conversion
        # (converts HF "mamba"→"mamba2" and "attention"→"full_attention")
        bamba = BambaConfig.from_transformers(config, parent_config)
        bamba_fields = _shallow_fields(bamba)
        return cls(
            **bamba_fields,
            shared_intermediate_size=getattr(config, "shared_intermediate_size", 1024),
        )


@dataclasses.dataclass
class NemotronHConfig(ArchitectureConfig):
    """Configuration for NemotronH hybrid Mamba2+Attention+MLP models.

    Uses multi-head Mamba2/SSD layers interleaved with attention and
    standalone MLP layers.  Each layer is a single-mixer block
    (RMSNorm → mixer → residual).
    """

    mamba_n_heads: int = 128
    mamba_d_head: int = 64
    mamba_d_state: int = 128
    mamba_n_groups: int = 8
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronHConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Get layer types from layers_block_type or hybrid_override_pattern
        layers_block_type = getattr(config, "layers_block_type", None)
        if layers_block_type is None:
            pattern = getattr(config, "hybrid_override_pattern", "")
            # Map pattern chars: M=mamba2, *=full_attention, -=mlp
            char_map = {"M": "mamba2", "*": "full_attention", "-": "mlp"}
            layers_block_type = [char_map.get(c, "mamba2") for c in pattern]
        else:
            # Convert HF names to mobius names
            type_map = {
                "mamba": "mamba2",
                "attention": "full_attention",
                "moe": "mlp",
            }
            layers_block_type = [type_map.get(t, t) for t in layers_block_type]

        # Override num_hidden_layers based on actual pattern length
        n = len(layers_block_type) if layers_block_type else base.num_hidden_layers

        mamba_expand = getattr(config, "expand", getattr(config, "mamba_expand", 2))
        d_inner = config.hidden_size * mamba_expand

        mamba_n_heads = getattr(config, "mamba_num_heads", 128)
        mamba_d_head = getattr(config, "mamba_head_dim", "auto")
        if mamba_d_head == "auto":
            mamba_d_head = d_inner // mamba_n_heads

        # Exclude fields we set explicitly to avoid duplicate keyword args
        base_fields = {
            k: v
            for k, v in _shallow_fields(base).items()
            if k not in ("layer_types", "num_hidden_layers", "hidden_act")
        }
        return cls(
            **base_fields,
            num_hidden_layers=n,
            layer_types=layers_block_type,
            hidden_act="relu2",
            mamba_n_heads=mamba_n_heads,
            mamba_d_head=mamba_d_head,
            mamba_d_state=getattr(config, "ssm_state_size", 128),
            mamba_n_groups=getattr(config, "n_groups", 8),
            mamba_d_conv=getattr(config, "conv_kernel", 4),
            mamba_expand=mamba_expand,
            mamba_conv_bias=getattr(config, "use_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
        )


@dataclasses.dataclass
class JetMoeConfig(CausalLMConfig):
    """Configuration for JetMoE: Mixture-of-Attention + MoE FFN model.

    JetMoE uses ``kv_channels`` as the per-head key/value dimension rather
    than deriving it from ``hidden_size // num_attention_heads``.  The
    standard formula gives the wrong answer because ``num_attention_heads``
    is the *total* Q head count (``top_k * num_kv_heads``), not the KV head
    count.  We therefore read ``kv_channels`` directly from the HF config
    and store it as ``head_dim``.
    """

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> JetMoeConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # Override head_dim to use kv_channels directly, not hidden/num_heads.
        kv_channels = getattr(config, "kv_channels", base.head_dim)
        # Also map num_kv_heads → num_key_value_heads if present (HF JetMoE
        # uses num_kv_heads instead of the standard num_key_value_heads).
        num_kv = getattr(config, "num_kv_heads", None)
        base_fields = _shallow_fields(base)
        base_fields["head_dim"] = kv_channels
        if num_kv is not None:
            base_fields["num_key_value_heads"] = num_kv
        return cls(**base_fields)


@dataclasses.dataclass
class WhisperConfig(BaseModelConfig):
    """Configuration for Whisper encoder-decoder models."""

    encoder_layers: int = DEFAULT_INT
    encoder_attention_heads: int = DEFAULT_INT
    encoder_ffn_dim: int = DEFAULT_INT
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    scale_embedding: bool = False
    decoder_start_token_id: int | None = None
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_transformers(cls, config) -> WhisperConfig:
        if config.model_type != "whisper":
            raise ValueError(
                f"WhisperConfig expects model_type='whisper', got '{config.model_type}'"
            )

        d_model = getattr(config, "d_model", config.hidden_size)
        decoder_heads = getattr(config, "decoder_attention_heads", config.num_attention_heads)

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=d_model,
            intermediate_size=getattr(config, "decoder_ffn_dim", 4 * d_model),
            num_hidden_layers=config.decoder_layers,
            num_attention_heads=decoder_heads,
            num_key_value_heads=decoder_heads,
            head_dim=d_model // decoder_heads,
            hidden_act=getattr(config, "activation_function", "gelu"),
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            attn_qkv_bias=True,
            attn_o_bias=True,
            encoder_layers=config.encoder_layers,
            encoder_attention_heads=getattr(config, "encoder_attention_heads", decoder_heads),
            encoder_ffn_dim=getattr(config, "encoder_ffn_dim", 4 * d_model),
            num_mel_bins=getattr(config, "num_mel_bins", 80),
            max_source_positions=getattr(config, "max_source_positions", 1500),
            max_target_positions=getattr(config, "max_target_positions", 448),
            scale_embedding=getattr(config, "scale_embedding", False),
            decoder_start_token_id=getattr(config, "decoder_start_token_id", None),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)
