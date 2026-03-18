# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GGUF → HuggingFace tensor name mapping.

Maps GGUF tensor names (e.g. ``blk.0.attn_q.weight``) to their
HuggingFace equivalents (e.g. ``model.layers.0.self_attn.q_proj.weight``).
Mappings are architecture-specific because different HF models use
different naming conventions.

The GGUF standard tensor names are defined in
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

Usage::

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
        build_gguf_to_hf_map,
    )

    hf_name = map_gguf_to_hf_names("blk.0.attn_q.weight", "llama")
    # Returns: "model.layers.0.self_attn.q_proj.weight"

    hf_name = map_gguf_to_hf_names("tokenizer.ggml.tokens", "llama")
    # Returns: None  (tokenizer tensors are skipped)
"""

from __future__ import annotations

__all__ = [
    "build_gguf_to_hf_map",
    "is_known_skip",
    "map_gguf_to_hf_names",
]

import functools
import re
from types import MappingProxyType

# ---------------------------------------------------------------------------
# Architecture-specific GGUF → HF stem mappings
# ---------------------------------------------------------------------------
# Keys are GGUF tensor name stems (without .weight/.bias suffix).
# ``{bid}`` is a placeholder for the block/layer index.
# Values are the corresponding HuggingFace tensor name stems.
#
# Verified against HuggingFace transformers model implementations
# (e.g. LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM, etc.).
# ---------------------------------------------------------------------------

# Llama-family naming convention. Used by Llama, Mistral, Qwen2, Qwen3,
# StarCoder2, InternLM2, Nemotron, StableLM, and DeciLM.
_LLAMA_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "blk.{bid}.attn_q": "model.layers.{bid}.self_attn.q_proj",
    "blk.{bid}.attn_k": "model.layers.{bid}.self_attn.k_proj",
    "blk.{bid}.attn_v": "model.layers.{bid}.self_attn.v_proj",
    "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj"),
    "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm"),
    "blk.{bid}.ffn_gate": "model.layers.{bid}.mlp.gate_proj",
    "blk.{bid}.ffn_up": "model.layers.{bid}.mlp.up_proj",
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

# Gemma2/3 adds pre/post feedforward layernorms.
_GEMMA2_EXTRAS: dict[str, str] = {
    "blk.{bid}.pre_ffn_norm": ("model.layers.{bid}.pre_feedforward_layernorm"),
    "blk.{bid}.post_ffn_norm": ("model.layers.{bid}.post_feedforward_layernorm"),
}

# Phi-3 uses fused QKV and gate-up projections.
_PHI3_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "blk.{bid}.attn_qkv": ("model.layers.{bid}.self_attn.qkv_proj"),
    "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj"),
    "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm"),
    "blk.{bid}.ffn_up": ("model.layers.{bid}.mlp.gate_up_proj"),
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

# Falcon uses transformer.h.* naming.
_FALCON_MAPPING: dict[str, str] = {
    "token_embd": "transformer.word_embeddings",
    "output": "lm_head",
    "output_norm": "transformer.ln_f",
    "blk.{bid}.attn_qkv": ("transformer.h.{bid}.self_attention.query_key_value"),
    "blk.{bid}.attn_output": ("transformer.h.{bid}.self_attention.dense"),
    "blk.{bid}.attn_norm": ("transformer.h.{bid}.input_layernorm"),
    "blk.{bid}.ffn_up": ("transformer.h.{bid}.mlp.dense_h_to_4h"),
    "blk.{bid}.ffn_down": ("transformer.h.{bid}.mlp.dense_4h_to_h"),
    "blk.{bid}.ffn_norm": "transformer.h.{bid}.ln_2",
}

# GPT-2 uses transformer.h.* with c_attn/c_proj naming.
_GPT2_MAPPING: dict[str, str] = {
    "token_embd": "transformer.wte",
    "position_embd": "transformer.wpe",
    "output": "lm_head",
    "output_norm": "transformer.ln_f",
    "blk.{bid}.attn_qkv": "transformer.h.{bid}.attn.c_attn",
    "blk.{bid}.attn_output": ("transformer.h.{bid}.attn.c_proj"),
    "blk.{bid}.attn_norm": "transformer.h.{bid}.ln_1",
    "blk.{bid}.ffn_up": "transformer.h.{bid}.mlp.c_fc",
    "blk.{bid}.ffn_down": "transformer.h.{bid}.mlp.c_proj",
    "blk.{bid}.ffn_norm": "transformer.h.{bid}.ln_2",
}

# Mamba uses backbone.* naming.
_MAMBA_MAPPING: dict[str, str] = {
    "token_embd": "backbone.embeddings",
    "output": "lm_head",
    "output_norm": "backbone.norm_f",
    "blk.{bid}.attn_norm": "backbone.layers.{bid}.norm",
    "blk.{bid}.ssm_in": ("backbone.layers.{bid}.mixer.in_proj"),
    "blk.{bid}.ssm_out": ("backbone.layers.{bid}.mixer.out_proj"),
    "blk.{bid}.ssm_conv1d": ("backbone.layers.{bid}.mixer.conv1d"),
    "blk.{bid}.ssm_dt": ("backbone.layers.{bid}.mixer.dt_proj"),
    "blk.{bid}.ssm_a": "backbone.layers.{bid}.mixer.A_log",
    "blk.{bid}.ssm_d": "backbone.layers.{bid}.mixer.D",
    "blk.{bid}.ssm_x": ("backbone.layers.{bid}.mixer.x_proj"),
}

# MoE extensions for Qwen2MoE/Qwen3MoE/DeepSeek.
_MOE_EXTRAS: dict[str, str] = {
    "blk.{bid}.ffn_gate_inp": ("model.layers.{bid}.mlp.gate"),
    "blk.{bid}.ffn_gate_exps": ("model.layers.{bid}.mlp.experts.gate_proj"),
    "blk.{bid}.ffn_up_exps": ("model.layers.{bid}.mlp.experts.up_proj"),
    "blk.{bid}.ffn_down_exps": ("model.layers.{bid}.mlp.experts.down_proj"),
    "blk.{bid}.ffn_gate_inp_shexp": ("model.layers.{bid}.mlp.shared_expert_gate"),
    "blk.{bid}.ffn_gate_shexp": ("model.layers.{bid}.mlp.shared_expert.gate_proj"),
    "blk.{bid}.ffn_up_shexp": ("model.layers.{bid}.mlp.shared_expert.up_proj"),
    "blk.{bid}.ffn_down_shexp": ("model.layers.{bid}.mlp.shared_expert.down_proj"),
}

# Qwen3.5-MoE hybrid extensions: DeltaNet (SSM) + full-attention + MoE.
# DeltaNet layers use linear_attn.* naming; full-attention layers add
# q_norm/k_norm under self_attn; both use post_attention_layernorm.
_QWEN35MOE_EXTRAS: dict[str, str] = {
    # DeltaNet (linear attention) layers
    "blk.{bid}.attn_qkv": "model.layers.{bid}.linear_attn.in_proj_qkv",
    "blk.{bid}.attn_gate": "model.layers.{bid}.linear_attn.in_proj_z",
    "blk.{bid}.ssm_beta": "model.layers.{bid}.linear_attn.in_proj_b",
    "blk.{bid}.ssm_alpha": "model.layers.{bid}.linear_attn.in_proj_a",
    "blk.{bid}.ssm_conv1d": "model.layers.{bid}.linear_attn.conv1d",
    "blk.{bid}.ssm_dt": "model.layers.{bid}.linear_attn.dt_bias",
    "blk.{bid}.ssm_a": "model.layers.{bid}.linear_attn.A_log",
    "blk.{bid}.ssm_norm": "model.layers.{bid}.linear_attn.norm",
    "blk.{bid}.ssm_out": "model.layers.{bid}.linear_attn.out_proj",
    # Full-attention layers — QK norms
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
    # Both layer types — post-attention layernorm
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

# Architectures sharing the llama HF naming convention.
_LLAMA_FAMILY = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "starcoder2",
        "internlm2",
        "nemotron",
        "stablelm",
        "deci",
    }
)

_GEMMA_FAMILY = frozenset({"gemma", "gemma2", "gemma3"})

_MOE_FAMILY = frozenset(
    {
        "qwen2moe",
        "qwen2_moe",
        "qwen3moe",
        "qwen3_moe",
        "qwen35moe",
    }
)


def is_known_skip(gguf_name: str) -> bool:
    """Return ``True`` if *gguf_name* is intentionally skipped.

    Known skip patterns include tokenizer tensors and rotary
    embedding frequency tensors that are computed, not loaded.
    """
    if gguf_name.startswith("tokenizer."):
        return True
    if "rope_freqs" in gguf_name or "attn_rot_embd" in gguf_name:
        return True
    return False


@functools.lru_cache(maxsize=16)
def _build_mapping(
    architecture: str,
) -> MappingProxyType[str, str]:
    """Return the GGUF→HF stem mapping for *architecture*.

    Cached per architecture to avoid rebuilding on every tensor.
    Returns an immutable proxy to prevent mutation of the cache.
    """
    arch = architecture.lower()

    if arch in _LLAMA_FAMILY:
        result = dict(_LLAMA_MAPPING)
    elif arch in _GEMMA_FAMILY:
        result = dict(_LLAMA_MAPPING)
        result.update(_GEMMA2_EXTRAS)
    elif arch == "phi3":
        result = dict(_PHI3_MAPPING)
    elif arch == "falcon":
        result = dict(_FALCON_MAPPING)
    elif arch == "gpt2":
        result = dict(_GPT2_MAPPING)
    elif arch == "mamba":
        result = dict(_MAMBA_MAPPING)
    elif arch in _MOE_FAMILY:
        result = dict(_LLAMA_MAPPING)
        result.update(_MOE_EXTRAS)
        if arch == "qwen35moe":
            result.update(_QWEN35MOE_EXTRAS)
    else:
        supported = sorted(
            _LLAMA_FAMILY | _GEMMA_FAMILY | _MOE_FAMILY | {"phi3", "falcon", "gpt2", "mamba"}
        )
        raise ValueError(
            f"Unsupported GGUF architecture: {architecture!r}. "
            f"Supported: {', '.join(supported)}"
        )
    return MappingProxyType(result)


# Regex to extract the block index from "blk.0.attn_q" etc.
_BLK_PATTERN = re.compile(r"blk\.(\d+)\.")
_BLK_TEMPLATE = "blk.{bid}."


def _split_suffix(name: str) -> tuple[str, str]:
    """Split ``"blk.0.attn_q.weight"`` → ``("blk.0.attn_q", ".weight")``.

    Returns ``("blk.0.attn_q", "")`` if no suffix is found.
    """
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def map_gguf_to_hf_names(
    gguf_name: str,
    architecture: str,
) -> str | None:
    """Map a GGUF tensor name to the equivalent HuggingFace name.

    Args:
        gguf_name: Full GGUF tensor name
            (e.g. ``"blk.0.attn_q.weight"``).
        architecture: GGUF architecture string
            (e.g. ``"llama"``, ``"qwen2"``).

    Returns:
        The corresponding HuggingFace tensor name, or ``None``
        if the tensor should be skipped (e.g. tokenizer tensors,
        rotary embedding frequencies).
    """
    # Skip known non-model tensors (tokenizer, rope freqs, etc.)
    if is_known_skip(gguf_name):
        return None

    stem, suffix = _split_suffix(gguf_name)
    mapping = _build_mapping(architecture)

    # Block-indexed tensors: blk.{N}.xxx → model.layers.{N}.xxx
    blk_match = _BLK_PATTERN.match(stem)
    if blk_match:
        bid = blk_match.group(1)
        lookup = _BLK_PATTERN.sub(_BLK_TEMPLATE, stem)
        hf_pattern = mapping.get(lookup)
        if hf_pattern is not None:
            return hf_pattern.replace("{bid}", bid) + suffix
    else:
        hf_stem = mapping.get(stem)
        if hf_stem is not None:
            return hf_stem + suffix

    return None


def build_gguf_to_hf_map(
    gguf_names: list[str],
    architecture: str,
) -> dict[str, str]:
    """Build a complete GGUF→HF name mapping for a list of tensors.

    Convenience function that calls :func:`map_gguf_to_hf_names`
    for each name and collects the results.

    Args:
        gguf_names: All GGUF tensor names from the file.
        architecture: GGUF architecture string.

    Returns:
        Dict mapping GGUF tensor names → HF tensor names.
        Tensors that should be skipped are omitted.
    """
    result: dict[str, str] = {}
    for name in gguf_names:
        hf_name = map_gguf_to_hf_names(name, architecture)
        if hf_name is not None:
            result[name] = hf_name
    return result
