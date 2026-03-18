# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for optional graph transformations.

These rules are **not applied by default**. Users can apply them after
model export to replace standard ONNX patterns with optimised custom ops.

Example::

    from mobius.rewrite_rules import (
        bias_gelu_rules,
        gelu_fusion_rules,
        layer_norm_fusion_rules,
        packed_attention_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
    )
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=packed_attention_rules())
    rewrite(model, pattern_rewrite_rules=skip_norm_rules())

    gpt2_model = build("openai-community/gpt2")
    rewrite(gpt2_model, pattern_rewrite_rules=skip_layer_norm_rules())
    rewrite(gpt2_model, pattern_rewrite_rules=bias_gelu_rules())
"""

__all__ = [
    "bias_gelu_rules",
    "fused_matmul_rules",
    "gelu_fusion_rules",
    "group_query_attention_rules",
    "layer_norm_fusion_rules",
    "packed_attention_rules",
    "skip_layer_norm_rules",
    "skip_norm_rules",
]

from mobius.rewrite_rules._bias_gelu import bias_gelu_rules
from mobius.rewrite_rules._fused_matmul import fused_matmul_rules
from mobius.rewrite_rules._gelu_fusion import gelu_fusion_rules
from mobius.rewrite_rules._group_query_attention import (
    group_query_attention_rules,
)
from mobius.rewrite_rules._layer_norm_fusion import (
    layer_norm_fusion_rules,
)
from mobius.rewrite_rules._packed_attention import packed_attention_rules
from mobius.rewrite_rules._skip_layer_norm import skip_layer_norm_rules
from mobius.rewrite_rules._skip_norm import skip_norm_rules
