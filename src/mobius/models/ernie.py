# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius.models.base import CausalLMModel


class ErnieCausalLMModel(CausalLMModel):
    """Ernie model with interleaved RoPE and compression ratio scaling.

    Inherits the base CausalLMModel. RoPE scaling is handled via the
    ArchitectureConfig rope_scaling parameter.
    """
