# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius.models.base import CausalLMModel


class QwenCausalLMModel(CausalLMModel):
    """Qwen (v1) model. Identical architecture to the standard CausalLMModel."""


class Qwen3CausalLMModel(CausalLMModel):
    """Qwen3 model with Q/K normalization.

    Uses RMSNorm on query and key projections before attention,
    configured via attn_qk_norm=True in ArchitectureConfig.
    """
