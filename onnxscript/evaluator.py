# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Public API for evaluator module.

This module re-exports the public API from the internal evaluator module.
"""

# Re-export all symbols from the internal evaluator module
from onnxscript._internal.evaluator import *  # noqa: F403

# Explicitly list main exports for type checking
from onnxscript._internal.evaluator import (
    Evaluator,
    default,
    default_as,
)

__all__ = ["default", "default_as", "Evaluator"]
