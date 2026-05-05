# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Rewriter-specific context aliases.

This module re-exports ``OpBuilderBase`` and ``TapeBuilder`` from
:mod:`onnxscript._internal.tape_builder` and defines the ``RewriterContext`` alias
used in rewrite-rule signatures.
"""

from __future__ import annotations

from onnxscript._internal.tape_builder import OpBuilderBase, TapeBuilder, UsedOpsets

# Alias used in rewrite rule signatures (the ``op`` parameter type)
RewriterContext = OpBuilderBase

# Backward compatibility aliases
TapeRewriterContext = TapeBuilder
OptimizerContext = OpBuilderBase

__all__ = [
    "OpBuilderBase",
    "OptimizerContext",
    "RewriterContext",
    "TapeBuilder",
    "TapeRewriterContext",
    "UsedOpsets",
]
