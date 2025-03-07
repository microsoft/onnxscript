# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deprecated. This module is kept for backward compatibility."""

from __future__ import annotations

from typing import Any

import onnx

from onnxscript.rewriter import pattern
from onnxscript.rewriter import rewrite as _rewrite
from onnxscript.rewriter.ort_fusions import ORT_PATTERN_REWRITE_RULES

__all__ = [
    "rewrite",
    "ORT_PATTERN_REWRITE_RULES",
    "ORT_FUNCTION_REWRITE_RULES",
]

ORT_FUNCTION_REWRITE_RULES: list[Any] = []


def rewrite(
    model_proto: onnx.ModelProto,
    /,
    function_rules=None,
    pattern_rules: list[pattern.RewriteRule] | None = None,
) -> onnx.ModelProto:
    """Rewrite the model using the given rules.

    Args:
        model_proto: The model to rewrite.
        function_rules: The function rewrite rules to apply. If None, the default rules
            for onnxruntime are used.
        pattern_rules: The pattern rewrite rules to apply. If None, the default rules
            for onnxruntime are used.

    Returns:
        The rewritten model.
    """
    pattern_rules = pattern_rules or ORT_PATTERN_REWRITE_RULES
    return _rewrite(model_proto, pattern_rewrite_rules=pattern_rules)
