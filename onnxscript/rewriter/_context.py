# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Rewriter-specific context aliases.

This module re-exports ``BuilderBase`` and ``TapeBuilder`` from
:mod:`onnxscript._internal.tape_builder` and defines the ``RewriterContext`` alias
used in rewrite-rule signatures.
"""

from __future__ import annotations

from onnxscript._internal.tape_builder import BuilderBase, TapeBuilder, UsedOpsets

# Alias used in rewrite rule signatures (the ``op`` parameter type)
RewriterContext = BuilderBase

# Backward compatibility aliases
TapeRewriterContext = TapeBuilder
OptimizerContext = BuilderBase

__all__ = [
    "BuilderBase",
    "OptimizerContext",
    "RewriterContext",
    "TapeBuilder",
    "TapeRewriterContext",
    "UsedOpsets",
]
