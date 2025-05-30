# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Callable, Sequence, Union

import onnxscript.ir as ir
import onnxscript.ir.passes.common as common_passes
from onnxscript.rewriter import pattern

Dim = Union[int, ir.SymbolicDim]


def _check_shape(bindings: dict[str, Dim], val: ir.Value, shape: Sequence[str]) -> bool:
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


def apply_fusion_rules(rules: pattern.RewriteRule | pattern.RewriteRuleSet) -> Callable:
    """
    Apply the given fusion rules to the model and return the number of fusions applied.

    model: The input ONNX model represented as an `ir.Model`.
    debug: If debug is True, enable pattern matching tracer for debugging.
    apply_shape_inference: If True, apply shape inference after fusions.
    """

    def apply_to(
        model: ir.Model, debug: bool = False, apply_shape_inference: bool = False
    ) -> int:
        count = rules.apply_to_model(model)
        if apply_shape_inference:
            common_passes.ShapeInferencePass()(model)
        if count == 0 and debug:
            tracer = pattern.MatchingTracer()
            rules.apply_to_model(model, tracer=tracer)
            tracer.report()
        return count

    return apply_to
