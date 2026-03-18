# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tier 0.5: ONNX checker validation for all model architectures.

Runs ``onnx_ir.passes.common.CheckerPass`` on every built model.
This catches op attribute errors, invalid input/output counts, and
malformed protos that graph construction alone does not detect.

Fast (~30s), no weights or network access required.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
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
from onnx_ir.passes.common import CheckerPass

from mobius._config_resolver import _default_task_for_model
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius.tasks import get_task

_checker = CheckerPass()


def _fill_dummy_weights(model: ir.Model) -> None:
    """Fill initializers that have no const_value with zero tensors.

    Models built without weights leave initializers empty. The
    CheckerPass requires const_value to be set so it can serialize
    the model for the ONNX C checker.
    """
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        initializer.const_value = ir.Tensor(
            np.zeros(dims, dtype=dtype.numpy()),
        )


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


# ---------------------------------------------------------------------------
# Semantic test IDs (duplicated from build_graph_test to keep independent)
# ---------------------------------------------------------------------------
_SEMANTIC_IDS: dict[tuple[str, int], str] = {
    ("deepseek_v2", 0): "deepseek_v2_mla",
    ("deepseek_v2", 1): "deepseek_v2_no_mla",
    ("qwen3_5_text", 0): "qwen3_5_text_default",
    ("qwen3_5_text", 1): "qwen3_5_text_linear_attn",
}


def _make_params(configs: list[tuple[str, dict, bool]]) -> list:
    """Create pytest.param entries with stable unique IDs."""
    from collections import Counter

    stripped = [(mt, ov) for mt, ov, _ in configs]
    counts = Counter(mt for mt, _ in stripped)
    seen: dict[str, int] = {}
    params = []
    for model_type, overrides in stripped:
        if counts[model_type] > 1:
            idx = seen.get(model_type, 0)
            seen[model_type] = idx + 1
            test_id = _SEMANTIC_IDS.get((model_type, idx), f"{model_type}_{idx}")
        else:
            test_id = model_type
        params.append(pytest.param(model_type, overrides, id=test_id))
    return params


_ALL_CHECKER_PARAMS = (
    _make_params(ALL_CAUSAL_LM_CONFIGS)
    + _make_params(ENCODER_CONFIGS)
    + _make_params(SEQ2SEQ_CONFIGS)
    + _make_params(VISION_CONFIGS)
    + _make_params(DETECTION_CONFIGS)
)


@pytest.mark.parametrize(
    "model_type,config_overrides",
    _ALL_CHECKER_PARAMS,
)
class TestOnnxChecker:
    """Run onnx.checker on every model architecture."""

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        for model in pkg.values():
            _fill_dummy_weights(model)
            _checker(model)
