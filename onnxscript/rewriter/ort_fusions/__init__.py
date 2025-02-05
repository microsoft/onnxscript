# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fusion optimizations for ORT backend."""

__all__ = [
    "optimize_for_ort",
]

from onnxscript.rewriter.ort_fusions._core import optimize_for_ort
