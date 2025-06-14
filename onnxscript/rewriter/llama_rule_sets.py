# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Deprecated module for llama-specific rule sets.

This module is deprecated and kept for backward compatibility.
Use onnxscript.rewriter.basic_rules instead for general optimization rules.
"""
from __future__ import annotations

import warnings

from onnxscript.rewriter import basic_rules
from onnxscript.rewriter import pattern as orp

# Re-export rule classes for backward compatibility
SqueezeReshape = basic_rules.SqueezeReshape
CastIdentity = basic_rules.CastIdentity
CastCast = basic_rules.CastCast
ExpandIdentity = basic_rules.ExpandIdentity
ReshapeReshape = basic_rules.ReshapeReshape
SlicesSplit = basic_rules.SlicesSplit
TransposeIdentity = basic_rules.TransposeIdentity
TransposeTranspose = basic_rules.TransposeTranspose
UnsqueezeUnsqueeze = basic_rules.UnsqueezeUnsqueeze

# Re-export rule instances for backward compatibility
cast_cast_rule = basic_rules.cast_cast_rule
cast_identity_rule = basic_rules.cast_identity_rule
expand_identity_rule = basic_rules.expand_identity_rule
reshape_reshape_rule = basic_rules.reshape_reshape_rule
slice_split_rule = basic_rules.slice_split_rule
transpose_identity_rule = basic_rules.transpose_identity_rule
transpose_transpose_rule = basic_rules.transpose_transpose_rule
unsqueeze_unsqueeze_rule = basic_rules.unsqueeze_unsqueeze_rule
squeeze_reshape_1d_rule = basic_rules.squeeze_reshape_1d_rule


def llama_p0_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before any other one as they usually remove unnecessary computation
    such as the multiplication by 1 or two consecutive transpose.

    .. deprecated:: 
        This function is deprecated. Use ``basic_rules.basic_optimization_rules()`` instead.

    Returns:
        RewriteRuleSet
    """
    warnings.warn(
        "llama_p0_rule_set() is deprecated. Use basic_rules.basic_optimization_rules() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return basic_rules.basic_optimization_rules()
