# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Callable, Sequence, Union

import onnx_ir as ir
import onnx_ir.passes.common as common_passes

from onnxscript.rewriter._basics import MatchFailureError, MatchingTracer
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

Dim = Union[int, ir.SymbolicDim]


def check_shape_bool(bindings: dict[str, Dim], val: ir.Value, shape: Sequence[str]) -> bool:
    if val.shape is None:
        return False
    if val.shape.rank() != len(shape):
        return False
    for actual, expected in zip(val.shape, shape):
        if expected not in bindings:
            bindings[expected] = actual  # type: ignore[assignment]
        elif actual != bindings[expected]:
            return False
    return True


def check_shape(bindings: dict[str, Dim], val: ir.Value, shape: Sequence[str]):
    if val.shape is None:
        raise MatchFailureError(f"The shape of {val} is unknown.", val)
    if val.shape.rank() != len(shape):
        raise MatchFailureError(
            f"The rank of {val} ({val.shape.rank()} does not match the expected rank {len(shape)}.",
            val,
        )
    for i, (actual, expected) in enumerate(zip(val.shape, shape)):
        if expected not in bindings:
            bindings[expected] = actual  # type: ignore[assignment]
        elif actual != bindings[expected]:
            raise MatchFailureError(
                f"Dimension {i} of {val} ({actual}) does not have expected size ({bindings[expected]}).",
                val,
            )


def apply_fusion_rules(rules: RewriteRule | RewriteRuleSet) -> Callable:
    """
    Apply the given fusion rules to the model and return the number of fusions applied.

    model: The input ONNX model represented as an `ir.Model`.
    debug: If debug is True, enable pattern matching tracer for debugging.
    apply_shape_inference: If True, apply shape inference after fusions.
    """

    def apply_to(
        model: ir.Model, debug: bool = False, apply_shape_inference: bool = False, **kwargs
    ) -> int:
        count = rules.apply_to_model(model, **kwargs)
        if apply_shape_inference:
            common_passes.ShapeInferencePass()(model)
        if count == 0 and debug:
            tracer = MatchingTracer()
            rules.apply_to_model(model, tracer=tracer, **kwargs)
            tracer.report()
        return count

    return apply_to
