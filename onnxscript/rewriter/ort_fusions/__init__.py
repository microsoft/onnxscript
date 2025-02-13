# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fusion optimizations for ORT backend."""

__all__ = [
    "optimize_for_ort",
    "ORT_PATTERN_REWRITE_RULES",
]


from onnxscript.rewriter.ort_fusions._core import ORT_PATTERN_REWRITE_RULES, optimize_for_ort
