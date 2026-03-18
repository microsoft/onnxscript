# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Implicit code-path detection from config field values.

Maps config field conditions to code-path labels for dashboard
visualization. Zero-maintenance: adding a test config automatically
detects what it exercises.

Usage::

    from mobius._testing.code_paths import detect_code_paths

    paths = detect_code_paths({"num_local_experts": 8, "num_experts_per_tok": 2})
    # → {"moe"}

    paths = detect_code_paths({"sliding_window": 4096, "layer_types": ["full_attention", "linear_attention"]})
    # → {"sliding_window", "full_attn", "linear_attn"}
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class CodePathIndicator:
    """A rule that maps a config field condition to a code-path label.

    Attributes:
        label: Short label for the code path (e.g. ``"moe"``).
        field: Config field name to inspect.
        check: A callable ``(field_value) -> bool`` that returns True
            when the code path is active.
        description: Human-readable description for the dashboard.
        example_config: A minimal config override dict that activates
            this code path, for copy-paste in the dashboard.
    """

    label: str
    field: str
    check: Callable[[Any], bool]
    description: str
    example_config: dict[str, Any]


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _is_true(value: Any) -> bool:
    return value is True


def _is_less_than_one(value: Any) -> bool:
    return isinstance(value, (int, float)) and value < 1.0


def _contains_linear_attention(value: Any) -> bool:
    return isinstance(value, list) and "linear_attention" in value


def _contains_full_attention(value: Any) -> bool:
    return isinstance(value, list) and "full_attention" in value


def _contains_mamba(value: Any) -> bool:
    return isinstance(value, list) and any("mamba" in v for v in value)


# All known code-path indicators. Each maps a config field condition
# to a descriptive label. The dashboard uses these to show which code
# paths are exercised by each model type's test configs.
CODE_PATH_INDICATORS: list[CodePathIndicator] = [
    # -- Routing (MoE) --
    CodePathIndicator(
        label="moe",
        field="num_local_experts",
        check=_is_positive_int,
        description="Mixture of Experts routing",
        example_config={
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
    ),
    CodePathIndicator(
        label="shared_expert",
        field="shared_expert_intermediate_size",
        check=_is_positive_int,
        description="Shared expert alongside routed experts",
        example_config={"shared_expert_intermediate_size": 128},
    ),
    # -- Attention variants --
    CodePathIndicator(
        label="sliding_window",
        field="sliding_window",
        check=_is_positive_int,
        description="Sliding window attention",
        example_config={"sliding_window": 4096},
    ),
    CodePathIndicator(
        label="linear_attn",
        field="layer_types",
        check=_contains_linear_attention,
        description="Linear attention (DeltaNet) layers",
        example_config={
            "layer_types": ["full_attention", "linear_attention"],
        },
    ),
    CodePathIndicator(
        label="full_attn",
        field="layer_types",
        check=_contains_full_attention,
        description="Full attention layers in hybrid model",
        example_config={
            "layer_types": ["full_attention", "linear_attention"],
        },
    ),
    CodePathIndicator(
        label="mamba",
        field="layer_types",
        check=_contains_mamba,
        description="Mamba/SSM layers in hybrid model",
        example_config={
            "layer_types": ["attention", "mamba"],
        },
    ),
    # -- Normalization and position encoding --
    CodePathIndicator(
        label="qk_norm",
        field="attn_qk_norm",
        check=_is_true,
        description="QK normalization in attention",
        example_config={"attn_qk_norm": True},
    ),
    CodePathIndicator(
        label="partial_rope",
        field="partial_rotary_factor",
        check=_is_less_than_one,
        description="Partial rotary position embeddings",
        example_config={"partial_rotary_factor": 0.5},
    ),
    # -- Architecture-specific --
    CodePathIndicator(
        label="mla",
        field="q_lora_rank",
        check=_is_positive_int,
        description="Multi-head Latent Attention (DeepSeek)",
        example_config={
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
        },
    ),
    # -- Multimodal and embedding --
    CodePathIndicator(
        label="vision",
        field="image_token_id",
        check=_is_positive_int,
        description="Vision/multimodal image processing",
        example_config={"image_token_id": 1},
    ),
    CodePathIndicator(
        label="tie_embeddings",
        field="tie_word_embeddings",
        check=_is_true,
        description="Tied input/output embeddings",
        example_config={"tie_word_embeddings": True},
    ),
]


def detect_code_paths(
    config_overrides: dict[str, Any],
) -> set[str]:
    """Detect which code paths a config exercises.

    Args:
        config_overrides: A dict of config field names to values.
            Can be a raw dict of overrides from test configs, or
            ``dataclasses.asdict(some_config)``.

    Returns:
        Set of code-path labels that the config activates.
    """
    paths: set[str] = set()
    for indicator in CODE_PATH_INDICATORS:
        if indicator.field in config_overrides:
            value = config_overrides[indicator.field]
            if indicator.check(value):
                paths.add(indicator.label)
    return paths


def detect_code_paths_from_config(config: Any) -> set[str]:
    """Detect code paths from a dataclass config object.

    Unlike :func:`detect_code_paths` which works on override dicts,
    this inspects all fields of a config object (e.g.
    ``ArchitectureConfig``) and checks each indicator.

    Args:
        config: A dataclass config object with fields matching
            the indicator field names.

    Returns:
        Set of code-path labels that the config activates.
    """
    paths: set[str] = set()
    for indicator in CODE_PATH_INDICATORS:
        value = getattr(config, indicator.field, None)
        if value is not None and indicator.check(value):
            paths.add(indicator.label)
    return paths


def get_all_code_path_labels() -> list[str]:
    """Return sorted list of all known code-path labels."""
    return sorted({ind.label for ind in CODE_PATH_INDICATORS})


def get_indicator_by_label(label: str) -> CodePathIndicator | None:
    """Look up a code-path indicator by its label."""
    for indicator in CODE_PATH_INDICATORS:
        if indicator.label == label:
            return indicator
    return None
