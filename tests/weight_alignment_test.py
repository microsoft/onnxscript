# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tier 1: Weight alignment tests for preprocess_weights().

Verifies that preprocess_weights() doesn't drop or mangle ONNX
initializer names when the state dict already uses ONNX-aligned names
(identity mapping).  This catches bugs like:

- Falcon's ``h.`` prefix replacement corrupting weight names
- MoE fused weight names being dropped
- Unintended name collisions in weight renaming

Test design:
    1. Build the ONNX graph with a tiny config
    2. Collect parameter initializer names (excluding computed constants)
    3. Create an identity state dict: ``{name: ones(shape)}``
    4. Run ``preprocess_weights()`` on that dict
    5. Assert every original parameter name is still present

Models whose ``preprocess_weights()`` intentionally filters through
HF-specific patterns (GPT2, OPT, etc.) are marked ``xfail`` since
they cannot roundtrip ONNX-aligned names by design.
"""

from __future__ import annotations

import pytest
import torch
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    SEQ2SEQ_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    VISION_CONFIGS,
)

from mobius._config_resolver import _default_task_for_model
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius.tasks import get_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_config(**overrides) -> ArchitectureConfig:
    config_cls = overrides.pop("_config_cls", ArchitectureConfig)
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


def _collect_parameter_names(pkg: dict) -> set[str]:
    """Return ONNX initializer names that represent model parameters.

    Excludes computed constants (RoPE caches, scalar constants, etc.)
    which have ``const_value is not None``.
    """
    names: set[str] = set()
    for sub_model in pkg.values():
        for name in sub_model.graph.initializers:
            init = sub_model.graph.initializers[name]
            if init.const_value is None:
                names.add(name)
    return names


def _build_identity_state_dict(pkg: dict, param_names: set[str]) -> dict[str, torch.Tensor]:
    """Create a state dict with ones for each parameter initializer."""
    state_dict: dict[str, torch.Tensor] = {}
    for sub_model in pkg.values():
        for name in sub_model.graph.initializers:
            if name not in param_names or name in state_dict:
                continue
            init = sub_model.graph.initializers[name]
            if init.shape is not None and len(init.shape) > 0:
                shape = list(init.shape)
                state_dict[name] = torch.ones(shape)
            else:
                # Scalar parameter
                state_dict[name] = torch.ones(())
    return state_dict


# Models whose preprocess_weights() intentionally filters names through
# HF-specific patterns. These cannot roundtrip ONNX-aligned names.
_FILTERING_PREPROCESS_MODELS: set[str] = {
    # OPT: expects model.decoder.* HF format
    "opt",
    # ModernBert decoder: expects model.layers.* HF format with renames
    "modernbert-decoder",
}


def _mark_xfail_if_filtering(
    configs: list[tuple[str, dict, bool]],
) -> list:
    """Build pytest params, marking filtering models as xfail."""
    return _mark_xfail_if_filtering_set(configs, _FILTERING_PREPROCESS_MODELS)


def _mark_xfail_if_filtering_set(
    configs: list[tuple[str, dict, bool]],
    filtering_models: set[str],
) -> list:
    """Build pytest params, marking specified models as xfail."""
    params = []
    for model_type, overrides, _ in configs:
        if model_type in filtering_models:
            params.append(
                pytest.param(
                    model_type,
                    overrides,
                    id=model_type,
                    marks=pytest.mark.xfail(
                        reason="preprocess_weights() filters HF-only patterns",
                        strict=True,
                    ),
                )
            )
        else:
            params.append(pytest.param(model_type, overrides, id=model_type))
    return params


def _assert_identity_roundtrip(model_type: str, config_overrides: dict) -> None:
    """Build model, run identity state dict through preprocess_weights()."""
    config = _base_config(**config_overrides)
    model_cls = registry.get(model_type)
    module = model_cls(config)
    task_name = _default_task_for_model(model_type)
    task = get_task(task_name)
    pkg = task.build(module, config)

    param_names = _collect_parameter_names(pkg)
    if not param_names:
        pytest.skip("No parameter initializers in model")

    if not hasattr(module, "preprocess_weights"):
        pytest.skip("Model has no preprocess_weights()")

    state_dict = _build_identity_state_dict(pkg, param_names)
    result = module.preprocess_weights(state_dict)

    missing = param_names - set(result.keys())
    assert not missing, (
        f"preprocess_weights() dropped {len(missing)} parameter(s): {sorted(missing)[:10]}"
    )


# ---------------------------------------------------------------------------
# Causal LM weight alignment
# ---------------------------------------------------------------------------
_CAUSAL_LM_PARAMS = _mark_xfail_if_filtering(ALL_CAUSAL_LM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _CAUSAL_LM_PARAMS)
class TestCausalLMWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for causal LM models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Encoder-only weight alignment
# ---------------------------------------------------------------------------
# Encoder models that filter HF-only patterns in preprocess_weights().
_FILTERING_ENCODER_MODELS: set[str] = {
    # CLIP text: expects text_model.encoder.* HF format
    "clip_text_model",
    # ModernBert: expects model.layers.* HF format with renames
    "modernbert",
}

_ENCODER_PARAMS = _mark_xfail_if_filtering_set(ENCODER_CONFIGS, _FILTERING_ENCODER_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_PARAMS)
class TestEncoderWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for encoder models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Seq2Seq weight alignment
# ---------------------------------------------------------------------------
# T5-family models: preprocess_weights() renames from HF block.N.layer.K.*
# format which doesn't roundtrip ONNX-aligned decoder.block.N.* names.
_FILTERING_SEQ2SEQ_MODELS: set[str] = {
    "longt5",
    "mt5",
    "t5",
    "switch_transformers",
    "umt5",
}

_SEQ2SEQ_PARAMS = _mark_xfail_if_filtering_set(SEQ2SEQ_CONFIGS, _FILTERING_SEQ2SEQ_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_PARAMS)
class TestSeq2SeqWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for seq2seq models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Vision model weight alignment
# ---------------------------------------------------------------------------
# Vision models: preprocess_weights() renames from HF encoder.layers.*
# format which doesn't roundtrip ONNX-aligned encoder.N.* names.
_FILTERING_VISION_MODELS: set[str] = {mt for mt, _, _ in VISION_CONFIGS}

_VISION_PARAMS = _mark_xfail_if_filtering_set(VISION_CONFIGS, _FILTERING_VISION_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _VISION_PARAMS)
class TestVisionWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for vision models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Detection model weight alignment
# ---------------------------------------------------------------------------
# Detection models: preprocess_weights() renames from HF encoder.layers.*
# format which doesn't roundtrip ONNX-aligned encoder.N.* names.
_FILTERING_DETECTION_MODELS: set[str] = {mt for mt, _, _ in DETECTION_CONFIGS}

_DETECTION_PARAMS = _mark_xfail_if_filtering_set(
    DETECTION_CONFIGS, _FILTERING_DETECTION_MODELS
)


@pytest.mark.parametrize("model_type,config_overrides", _DETECTION_PARAMS)
class TestDetectionWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for detection models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)
