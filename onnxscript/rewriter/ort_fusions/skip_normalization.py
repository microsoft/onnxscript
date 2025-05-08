# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, pattern

Dim = Union[int, ir.SymbolicDim]

# Fusion rule for SkipRMSNormalization


class SkipRmsNormFusion(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, has_bias: bool = False, bias_pre_add: bool = False):
        """Fusion rule for SkipRMSNormalization."""
        super().__init__(name=name)
        self._has_bias = has_bias
        self._bias_pre_add = bias_pre_add

    def pattern(self, op, input, skip, gamma, bias, epsilon, stash_type):
        if self._has_bias and self._bias_pre_add:
            input = op.Add(input, bias)

        # Support different combinations of addition of input and skip
        skip_sum_pattern_1 = op.Add(skip, input)
        skip_sum_pattern_2 = op.Add(input, skip)
        skip_sum = pattern.OrValue(
            [skip_sum_pattern_1, skip_sum_pattern_2],
            name="skip_sum",
        )

        if self._has_bias and not self._bias_pre_add:
            skip_sum = op.Add(skip_sum, bias)
        # Note: ORT's SimplifiedLayerNormalization was placed in onnx domain by mistake.
        # No need to use com.microsoft domain here; but this is a custom op in ORT.
        normalized = op.SimplifiedLayerNormalization(
            skip_sum,
            gamma,
            axis=-1,
            epsilon=epsilon,
            stash_type=stash_type,
        )
        return normalized, skip_sum

    def check(
        self,
        op,
        input,
        skip,
        gamma,
        bias,
        epsilon,
        stash_type,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        """Check if the pattern matches conditions for use of SkipSimplifiedLayerNormalization op."""
        check_result = pattern.MatchResult()
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(bindings, val, dims)

        if no_match(input, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {input} does not match expected dimensions ['B', 'S', 'D']",
                input,
            )
        if no_match(skip, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {skip} does not match expected dimensions ['B', 'S', 'D']",
                skip,
            )
        if no_match(gamma, ["D"]):
            return check_result.fail(
                f"Shape mismatch: {gamma} does not match expected dimensions ['D']",
                gamma,
            )
        if self._has_bias:
            if no_match(bias, ["D"]):
                return check_result.fail(
                    f"Shape mismatch: {bias} does not match expected dimensions ['D']",
                    bias,
                )

        return check_result

    def rewrite(
        self,
        op,
        input,
        skip,
        gamma,
        bias,
        epsilon,
        stash_type,
        **_,
    ):
        if self._has_bias:
            normalized, _mean, _inv_std_var, skip_sum = op.SkipSimplifiedLayerNormalization(
                input,
                skip,
                gamma,
                bias,
                epsilon=epsilon,
                _outputs=4,
                _domain="com.microsoft",
            )
        else:
            normalized, _mean, _inv_std_var, skip_sum = op.SkipSimplifiedLayerNormalization(
                input,
                skip,
                gamma,
                epsilon=epsilon,
                _outputs=4,
                _domain="com.microsoft",
            )
        return normalized, skip_sum


_skip_rms_add_bias_rule = SkipRmsNormFusion.rule(
    "SkipRmsNormBias", has_bias=True, bias_pre_add=False
)
_skip_rms_pre_add_bias_rule = SkipRmsNormFusion.rule(
    "SkipRmsNormPreBias", has_bias=True, bias_pre_add=True
)
_skip_rms_rule = SkipRmsNormFusion.rule("SkipRmsNorm", has_bias=False)

skip_rms_normalization_ruleset = pattern.RewriteRuleSet(
    [_skip_rms_pre_add_bias_rule, _skip_rms_add_bias_rule, _skip_rms_rule]
)
fuse_skip_rms_normalization = _fusion_utils.apply_fusion_rules(skip_rms_normalization_ruleset)


# Fusion rule for SkipLayerNormalization
class SkipLayerNormFusion(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, has_bias: bool = False, bias_pre_add: bool = False):
        """Fusion rule for SkipLayerNormalization."""
        super().__init__(name=name)
        self._has_bias = has_bias
        self._bias_pre_add = bias_pre_add

    def pattern(self, op, input, skip, gamma, beta, bias, epsilon, stash_type):
        if self._has_bias and self._bias_pre_add:
            input = op.Add(input, bias)

        # Support different combinations of addition of input and skip
        skip_sum_pattern_1 = op.Add(skip, input)
        skip_sum_pattern_2 = op.Add(input, skip)
        skip_sum = pattern.OrValue([skip_sum_pattern_1, skip_sum_pattern_2], name="skip_sum")

        if self._has_bias and not self._bias_pre_add:
            skip_sum = op.Add(skip_sum, bias)
        normalized = op.LayerNormalization(
            skip_sum,
            gamma,
            beta,
            axis=-1,
            epsilon=epsilon,
            stash_type=stash_type,
        )
        return normalized, skip_sum

    def check(
        self,
        op,
        input,
        skip,
        gamma,
        beta,
        bias,
        epsilon,
        stash_type,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        """Check if the pattern matches conditions for use of SimplifiedLayerNormalization op."""
        check_result = pattern.MatchResult()
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(bindings, val, dims)

        if no_match(input, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {input} does not match expected dimensions ['B', 'S', 'D']",
                input,
            )
        if no_match(skip, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {skip} does not match expected dimensions ['B', 'S', 'D']",
                skip,
            )
        if no_match(gamma, ["D"]):
            return check_result.fail(
                f"Shape mismatch: {gamma} does not match expected dimensions ['D']",
                gamma,
            )
        if no_match(beta, ["D"]):
            return check_result.fail(
                f"Shape mismatch: {beta} does not match expected dimensions ['D']",
                beta,
            )
        if self._has_bias:
            if no_match(bias, ["D"]):
                return check_result.fail(
                    f"Shape mismatch: {bias} does not match expected dimensions ['D']",
                    bias,
                )

        return check_result

    def rewrite(
        self,
        op,
        input,
        skip,
        gamma,
        beta,
        bias,
        epsilon,
        stash_type,
        **_,
    ):
        normalized, _mean, _inv_std_var, skip_sum = op.SkipLayerNormalization(
            input,
            skip,
            gamma,
            beta,
            bias,
            epsilon=epsilon,
            _outputs=4,
            _domain="com.microsoft",
        )
        return normalized, skip_sum


_skip_layer_add_bias_rule = SkipLayerNormFusion.rule(
    "SkipLayerNormBias", has_bias=True, bias_pre_add=False
)
_skip_layer_pre_add_bias_rule = SkipLayerNormFusion.rule(
    "SkipLayerNormPreBias", has_bias=True, bias_pre_add=True
)
_skip_layer_rule = SkipLayerNormFusion.rule("SkipLayerNorm", has_bias=False)

skip_layer_normalization_ruleset = pattern.RewriteRuleSet(
    [_skip_layer_pre_add_bias_rule, _skip_layer_add_bias_rule, _skip_layer_rule]
)


fuse_skip_layer_normalization = _fusion_utils.apply_fusion_rules(
    skip_layer_normalization_ruleset
)
