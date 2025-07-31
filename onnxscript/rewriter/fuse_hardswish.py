# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Does the following transformation:
- Div(Clip(Add(x))) -> HardSigmoid
- Mul(HardSigmoid(x), x) -> HardSwish
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import pattern
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import is_singleton_value
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet


class _HardSigmoidFusionBase(pattern.RewriteRuleClassBase):
    def check(
        self,
        op,
        x: ir.Value,
        clip_min: ir.Value,
        clip_max: ir.Value,
        bias: ir.Value,
        divisor: ir.Value,
    ) -> MatchResult:
        check_result = MatchResult()

        if not is_singleton_value(clip_min, 0.0, rtol=1e-4):
            return check_result.fail("Swish requires min value of 0 for clip")
        if not is_singleton_value(clip_max, 6.0, rtol=1e-4):
            return check_result.fail("Swish requires max value of 6 for clip")
        if not is_singleton_value(bias, 3.0, rtol=1e-4):
            return check_result.fail("Swish requires bias value of 3")
        if not is_singleton_value(divisor, 6.0, rtol=1e-4):
            return check_result.fail("Swish requires divisor value of 6")
        return check_result


class HardSwishFusion(_HardSigmoidFusionBase):
    def pattern(
        self,
        op,
        x: ir.Value,
        clip_min: ir.Value,
        clip_max: ir.Value,
        bias: ir.Value,
        divisor: ir.Value,
    ) -> ir.Value:
        out = op.Clip(x + bias, clip_min, clip_max) * x
        out = out / divisor
        return out

    def rewrite(
        self,
        op,
        x: ir.Value,
        clip_min: ir.Value,
        clip_max: ir.Value,
        bias: ir.Value,
        divisor: ir.Value,
    ) -> ir.Value:
        return op.HardSwish(x)


class HardSwishFusionFromHardSigmoid(pattern.RewriteRuleClassBase):
    def pattern(self, op, x: ir.Value) -> ir.Value:
        # Floating point matching for 1/6 is not exact, so we use isclose below
        out = op.HardSigmoid(x, _allow_other_attributes=True, _outputs=["hardsigmoid_out"])
        out = out * x
        return out

    def check(self, op, x: ir.Value, hardsigmoid_out: ir.Value) -> MatchResult:
        check_result = MatchResult()
        hardsigmoid = hardsigmoid_out.producer()
        if not np.isclose(hardsigmoid.attributes["alpha"].value, 1 / 6):
            return check_result.fail(
                "HardSigmoid alpha must be 1/6 to get fused into HardSwish"
            )
        if not np.isclose(hardsigmoid.attributes["beta"].value, 0.5):
            return check_result.fail(
                "HardSigmoid beta must be 0.5 to get fused into HardSwish"
            )
        return check_result

    def rewrite(self, op, x: ir.Value, hardsigmoid_out: ir.Value) -> ir.Value:
        return op.HardSwish(x)


class HardSigmoidFusion(_HardSigmoidFusionBase):
    """Fuse HardSigmoid only for HardSwish hyper-parameters: alpha=1/6, beta=0.5"""

    def pattern(
        self,
        op,
        x: ir.Value,
        clip_min: ir.Value,
        clip_max: ir.Value,
        bias: ir.Value,
        divisor: ir.Value,
    ) -> ir.Value:
        out = op.Clip(x + bias, clip_min, clip_max)
        out = out / divisor
        return out

    def rewrite(
        self,
        op,
        x: ir.Value,
        clip_min: ir.Value,
        clip_max: ir.Value,
        bias: ir.Value,
        divisor: ir.Value,
    ) -> ir.Value:
        return op.HardSigmoid(x, alpha=1 / 6, beta=0.5)


def fuse_hardswish_rules() -> RewriteRuleSet:
    """Returns the rewrite rules for fusing HardSwish and HardSigmoid."""
    return RewriteRuleSet(
        [
            HardSwishFusion().rule(),
            HardSigmoidFusion().rule(),
            HardSwishFusionFromHardSigmoid().rule(),
        ],
        commute=True,
    )
