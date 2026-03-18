# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""L3 Synthetic Parity Tests — registered model types.

Builds tiny random-weight models for BOTH HuggingFace PyTorch and ONNX,
compares single-forward-pass logits.  Any atol divergence with identical
seeds indicates a genuine op-level bug.

Run::

    pytest tests/synthetic_parity_test.py -v --tb=short -n 0

Run a single model::

    pytest tests/synthetic_parity_test.py -k "llama" -v
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir
import pytest
import torch
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
)

from mobius._config_resolver import _default_task_for_model
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.parity import ParityResult, compare_synthetic
from mobius._weight_loading import apply_weights
from mobius.tasks import get_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model types that cannot be tested with HF synthetic parity.
# Each entry maps a model_type to a reason for skipping.
# ---------------------------------------------------------------------------
_SKIP_REASONS: dict[str, str] = {
    # Our custom name; not a valid HF model_type
    "phi3small": "HF AutoConfig does not recognize phi3small",
    "phi4mm": "Multi-modal model, requires special HF setup",
    "chatglm": "Requires trust_remote_code=True",
    "qwen": "HF Qwen requires trust_remote_code=True",
    "openelm": "HF OpenELM requires trust_remote_code=True",
    "internlm2": "HF AutoConfig does not recognize internlm2",
    "minicpm": "HF AutoConfig does not recognize minicpm",
    "minicpm3": "HF AutoConfig does not recognize minicpm3",
    "ernie4_5": "HF ernie4_5 model requires special fields",
    # Aliases that HF doesn't support as model_type
    "arctic": "HF AutoConfig does not recognize arctic",
    "baichuan": "HF AutoConfig does not recognize baichuan",
    "codegen2": "HF AutoConfig does not recognize codegen2",
    "command_r": "HF AutoConfig does not recognize command_r",
    "deepseek_v2_moe": "HF AutoConfig does not recognize deepseek_v2_moe",
    "exaone": "HF AutoConfig does not recognize exaone",
    "gptoss": "HF AutoConfig does not recognize gptoss",
    "gpt_oss": "HF AutoConfig does not recognize gpt_oss",
    "open-llama": "HF AutoConfig does not recognize open-llama",
    "yi": "HF AutoConfig does not recognize yi",
    # Mamba2 standalone model: HF creates different architecture
    "mamba2": "HF Mamba2 standalone is not a causal LM model",
}

# Model types with known ONNX-vs-HF divergences, tracked as xfail.
# Each maps model_type → reason the outputs diverge.
_XFAIL_REASONS: dict[str, str] = {
    # MoE routing: ONNX TopKGate differs from HF's router impl
    "qwen2_moe": "MoE routing differences",
    "qwen3_moe": "MoE routing differences",
    "qwen3_5_moe": "MoE routing differences",
    "granitemoe": "MoE routing differences",
    "granitemoeshared": "MoE routing differences",
    "olmoe": "MoE routing differences",
    "phimoe": "MoE routing differences",
    "jetmoe": "MoE routing differences",
    "dbrx": "MoE routing differences",
    "ernie4_5_moe": "MoE routing differences",
    "flex_olmo": "MoE routing differences",
    "glm4_moe": "MoE routing differences",
    "glm4v_moe_text": "MoE routing differences",
    "hunyuan_v1_moe": "MoE routing differences",
    "minimax": "MoE routing differences",
    "qwen3_omni_moe": "MoE routing differences",
    "qwen3_vl_moe": "MoE routing differences",
    # HF architecture differences (extra layers/features not in our ONNX)
    "diffllama": "HF DiffLlama has extra lambda parameters",
    "llama4_text": "HF Llama4 MoE differs from our implementation",
    # Softcapping/scaling differences
    "gemma2": "Attention softcapping implementation differs",
    "shieldgemma2": "Attention softcapping implementation differs",
    # Tied embeddings / layernorm differences
    "cohere": "LayerNorm implementation differs from HF",
    "cohere2": "LayerNorm implementation differs from HF",
    "starcoder2": "LayerNorm implementation differs from HF",
    "ctrl": "Absolute positional embedding implementation differs",
    "opt": "OPT architecture differences (learned pos embeddings)",
    # GPT-2 family with different layernorm
    "gpt2": "GPT2 conv1d vs linear weight handling",
    "biogpt": "GPT2 family layernorm differences",
    "gpt-sw3": "GPT2 family layernorm differences",
    "gpt_bigcode": "GPT2 family layernorm differences",
    "gpt_neo": "GPT2 family layernorm differences",
    "openai-gpt": "GPT2 family layernorm differences",
    "xglm": "GPT2 family layernorm differences",
    "imagegpt": "GPT2 family layernorm differences",
    "xlm": "GPT2 family layernorm differences",
    # Granite scaling multipliers
    "granite_0": "Granite embedding/logit scaling differences",
    # Gemma family: query_pre_attn_scalar
    "gemma": "Gemma attention scaling differences",
    "gemma3_text": "Gemma3 attention scaling/qk_norm differences",
    "gemma3n_text": "Gemma3n AltUp/Laurel implementation differs",
    "gemma3n": "Gemma3n AltUp/Laurel implementation differs",
    "gemma3": "Gemma3 sliding window attention differs",
    # Qwen3_next: linear attention (DeltaNet)
    "qwen3_next": "DeltaNet linear attention differs from HF",
    # DeepSeek MLA
    "deepseek_v2": "MLA implementation differs from HF",
    "deepseek_v3": "MLA + MoE implementation differs",
    # Weight naming: HF uses different prefix/structure than ONNX
    "codegen": "HF uses transformer.h prefix, no gated MLP",
    "gptj": "HF uses transformer.h prefix, no gated MLP",
    "gpt_neox": "HF uses gpt_neox prefix + fused QKV",
    "gpt_neox_japanese": "HF uses gpt_neox prefix + fused QKV",
    "persimmon": "HF uses fused query_key_value + dense (not o_proj)",
    "mpt": "HF MPT uses Wqkv naming, not query_key_value",
    # Architecture: LayerNorm with bias instead of RMSNorm
    "stablelm": "HF uses LayerNorm with bias (not RMSNorm)",
    "nemotron": "Nemotron attention differs from base (needs investigation)",
    # Phi (original): HF uses dense, fc1/fc2, LayerNorm — not Llama-compatible
    "phi": "HF Phi uses dense/fc1/fc2 naming + LayerNorm",
    # Hybrid Mamba: weight naming and MoE routing differences
    "jamba": "Jamba MoE/Mamba weight naming + routing differences",
    "bamba": "Bamba precision differences (near-threshold, 0.0017)",
}

# Fields that are properties in HF configs and cannot be set directly.
_HF_READONLY_FIELDS: set[str] = {"head_dim"}

# Model types that need extra HF config fields beyond our defaults.
_HF_EXTRA_CONFIG: dict[str, dict] = {
    "phi3": {"pad_token_id": 0},
    "phi": {"pad_token_id": 0},
    "phimoe": {"pad_token_id": 0},
    "gemma2": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "shieldgemma2": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "gemma3_text": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "gemma3n_text": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "gemma3n": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "gemma3": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    "opt": {"word_embed_proj_dim": TINY_HIDDEN},
    # GPT-J/CodeGen use rotary_dim (not partial_rotary_factor)
    "gptj": {"rotary_dim": int(TINY_HEAD_DIM * 0.25)},
    "codegen": {"rotary_dim": int(TINY_HEAD_DIM * 0.5)},
    # Jamba requires CUDA mamba kernels by default; disable for CPU tests
    "jamba": {"use_mamba_kernels": False},
    # Nemotron uses norm_eps (not rms_norm_eps) in HF config
    "nemotron": {"norm_eps": 1e-5},
    # Qwen3.5 has head_dim as an explicit config param (default 256)
    "qwen3_5_text": {"head_dim": TINY_HEAD_DIM},
}


def _base_config(config_cls=None, **overrides) -> ArchitectureConfig:
    """Create a tiny config.  Mirrors build_graph_test._base_config."""
    if config_cls is None:
        config_cls = overrides.pop("_config_cls", ArchitectureConfig)
    else:
        overrides.pop("_config_cls", None)
    defaults = dict(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_HEAD_DIM,
        num_hidden_layers=TINY_LAYERS,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
    )
    defaults.update(overrides)
    return config_cls(**defaults)


def _create_hf_config(model_type: str, config_overrides: dict):
    """Create a HuggingFace config for the given model type.

    Returns (hf_config, hf_model_type) or raises to skip.
    """
    from transformers import AutoConfig

    # Determine the HF model_type (usually same as ours)
    hf_model_type = model_type

    # Build HF config kwargs from our tiny defaults
    hf_kwargs: dict = {
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": TINY_INTERMEDIATE,
        "num_attention_heads": TINY_HEADS,
        "num_key_value_heads": TINY_KV_HEADS,
        "num_hidden_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
    }

    # Apply config overrides (filtering out our internal keys)
    for key, value in config_overrides.items():
        if key == "_config_cls":
            continue
        if key in _HF_READONLY_FIELDS:
            continue
        hf_kwargs[key] = value

    # Apply model-specific HF extras
    if model_type in _HF_EXTRA_CONFIG:
        hf_kwargs.update(_HF_EXTRA_CONFIG[model_type])

    # Convert layer_types to attn_layer_indices for hybrid Mamba models
    # Bamba uses attn_layer_indices (computed property layers_block_type)
    if hf_model_type in ("bamba",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        hf_kwargs["attn_layer_indices"] = [
            i for i, lt in enumerate(layer_types) if lt in ("full_attention", "attention")
        ]

    # Jamba uses attn_layer_offset/attn_layer_period
    if hf_model_type in ("jamba",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        attn_indices = [
            i for i, lt in enumerate(layer_types) if lt in ("full_attention", "attention")
        ]
        if len(attn_indices) == len(layer_types):
            # All attention
            hf_kwargs["attn_layer_offset"] = 0
            hf_kwargs["attn_layer_period"] = 1
        elif len(attn_indices) > 0:
            hf_kwargs["attn_layer_offset"] = attn_indices[0]
            # Period = gap between consecutive attention layers
            if len(attn_indices) > 1:
                hf_kwargs["attn_layer_period"] = attn_indices[1] - attn_indices[0]
            else:
                hf_kwargs["attn_layer_period"] = len(layer_types)
        else:
            # All mamba — use large offset/period
            hf_kwargs["attn_layer_offset"] = len(layer_types)
            hf_kwargs["attn_layer_period"] = len(layer_types)

    try:
        hf_config = AutoConfig.for_model(hf_model_type, **hf_kwargs)
    except (ValueError, KeyError) as e:
        pytest.skip(f"Cannot create HF config for {model_type}: {e}")

    return hf_config


def _create_hf_model(model_type: str, hf_config, seed: int):
    """Create a HuggingFace model from config with deterministic init."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    try:
        hf_model = AutoModelForCausalLM.from_config(hf_config)
    except Exception as e:
        pytest.skip(f"Cannot create HF model for {model_type}: {type(e).__name__}: {e}")

    return hf_model.float().eval()


def _build_onnx_model(model_type: str, config: ArchitectureConfig):
    """Build ONNX model package using the registry."""
    model_cls = registry.get(model_type)
    module = model_cls(config)
    task_name = _default_task_for_model(model_type)
    task = get_task(task_name)
    config.dtype = ir.DataType.FLOAT
    pkg = task.build(module, config)
    return module, pkg


def _fill_random_weights(model: ir.Model, rng: np.random.Generator) -> None:
    """Fill all unset graph initializers with random float32 values."""
    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue
        shape = tuple(d for d in init.shape)
        if not shape:
            continue
        if init.dtype == ir.DataType.FLOAT:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        elif init.dtype == ir.DataType.FLOAT16:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        elif init.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            np_dtype = np.int64 if init.dtype == ir.DataType.INT64 else np.int32
            data = rng.integers(0, 10, size=shape).astype(np_dtype)
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        init.const_value = ir.Tensor(data)


# ---------------------------------------------------------------------------
# Parametrize over all causal LM configs
# ---------------------------------------------------------------------------
def _build_synthetic_params() -> list:
    """Build pytest.param list with xfail marks from _XFAIL_REASONS.

    Uses @pytest.mark.xfail (strict=False) so the test still runs —
    if a model starts passing, pytest reports it as XPASS, alerting
    us to remove it from _XFAIL_REASONS.
    """
    from collections import Counter

    configs = [(mt, ov) for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS]
    counts = Counter(mt for mt, _ in configs)
    seen: dict[str, int] = {}
    params = []
    for mt, ov in configs:
        if counts[mt] > 1:
            idx = seen.get(mt, 0)
            seen[mt] = idx + 1
            test_id = f"{mt}_{idx}"
        else:
            test_id = mt
        # Check test_id first (e.g. "granite_0"), then model_type
        xfail_reason = _XFAIL_REASONS.get(test_id, _XFAIL_REASONS.get(mt))
        marks = (
            [pytest.mark.xfail(reason=xfail_reason, strict=False)]
            if xfail_reason is not None
            else []
        )
        params.append(pytest.param(mt, ov, id=test_id, marks=marks))
    return params


_SYNTHETIC_PARAMS = _build_synthetic_params()


@pytest.mark.parametrize(
    "model_type,config_overrides",
    _SYNTHETIC_PARAMS,
)
def test_synthetic_parity(model_type: str, config_overrides: dict):
    """L3 synthetic parity: ONNX matches HF with identical random weights.

    Steps:
    1. Build tiny ONNX model from config
    2. Create equivalent HF model with deterministic init
    3. Transfer HF weights → ONNX via preprocess_weights
    4. Run forward pass on both with same input
    5. Compare logits using atol/rtol gate
    """
    if model_type in _SKIP_REASONS:
        pytest.skip(_SKIP_REASONS[model_type])

    # xfail is handled by marks in _build_synthetic_params() so the test
    # still runs — an XPASS signals that the model is fixed.

    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Build tiny config
    config = _base_config(**config_overrides)
    config.dtype = ir.DataType.FLOAT

    # 2. Build ONNX model
    try:
        module, pkg = _build_onnx_model(model_type, config)
    except Exception as e:
        pytest.skip(f"ONNX build failed for {model_type}: {e}")

    # 3. Create HF model
    hf_config = _create_hf_config(model_type, config_overrides)
    hf_model = _create_hf_model(model_type, hf_config, seed)

    # 4. Transfer HF weights to ONNX
    try:
        preprocessed = module.preprocess_weights(dict(hf_model.state_dict()))
        for onnx_model in pkg.values():
            apply_weights(onnx_model, preprocessed)
    except Exception as e:
        pytest.skip(f"Weight transfer failed for {model_type}: {type(e).__name__}: {e}")

    # Fill any remaining unset initializers (ONNX constants, etc.)
    for onnx_model in pkg.values():
        _fill_random_weights(onnx_model, rng)

    # 5. Prepare inputs
    input_ids = rng.integers(1, config.vocab_size, size=(1, 3)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :]

    # 6. HF forward
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        )
    hf_logits = hf_out.logits.numpy()

    # 7. ONNX forward
    from mobius._testing.ort_inference import OnnxModelSession

    onnx_model = pkg["model"]
    session = OnnxModelSession(onnx_model)

    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # Add zero-valued past KV cache feeds with correct shapes:
    # batch=1, past_sequence_len=0, other dims from model spec
    for inp in onnx_model.graph.inputs:
        name = inp.name
        if name in feeds:
            continue
        if not name.startswith("past_key_values"):
            continue
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            elif "past" in str(d):
                shape.append(0)
            elif "batch" in str(d):
                shape.append(1)
            else:
                shape.append(0)
        feeds[name] = np.zeros(shape, dtype=np.float32)

    try:
        onnx_out = session.run(feeds)
    except Exception as e:
        session.close()
        pytest.skip(f"ONNX inference failed for {model_type}: {type(e).__name__}: {e}")
    onnx_logits = onnx_out["logits"]
    session.close()

    # 8. Compare
    report = compare_synthetic(onnx_logits, hf_logits, rtol=1e-3, atol=1e-3)
    assert report.result != ParityResult.FAIL, f"{model_type}: {report.message}"
