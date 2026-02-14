# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Materialize Reshape shape input from known output shape.

When symbolic shape inference has been run, a Reshape node may have a known
output shape even though its shape input is computed dynamically (e.g., via a
Shape → Cast → Split → Concat chain).  This rule replaces the shape input
with a concrete constant, allowing the dynamic chain to become dead code and
be removed by unused-node elimination.

- Fully static output shape → constant with exact dims.
- Exactly one symbolic dim → replace it with ``-1`` (Reshape infers it).
"""

from __future__ import annotations

from onnxscript import ir
from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class MaterializeReshapeShape(RewriteRuleClassBase):
    """Replace a dynamic Reshape shape input with a constant when output shape is known."""

    def pattern(self, op, data, shape):
        return op.Reshape(data, shape)

    def check(self, context, data: ir.Value, shape: ir.Value) -> MatchResult:
        check_result = MatchResult()

        # Shape input must not already be a constant
        if ir_utils.get_numpy_value(shape) is not None:
            return check_result.fail("Shape input is already a constant.")

        output = context.output_values[0]
        if output.shape is None:
            return check_result.fail("Output shape is not known.")

        dims = list(output.shape)
        sym_count = sum(1 for d in dims if not isinstance(d, int))

        if sym_count == 0:
            self._new_dims = [int(d) for d in dims]
        elif sym_count == 1:
            self._new_dims = [-1 if not isinstance(d, int) else int(d) for d in dims]
        else:
            return check_result.fail(
                f"Output shape has {sym_count} symbolic dims, cannot materialize."
            )

        # Preserve allowzero attribute from original node
        self._allowzero = context.nodes[0].attributes.get_int("allowzero", 0)
        return check_result

    def rewrite(self, op, data: ir.Value, shape: ir.Value):
        new_shape = op.Constant(
            value=ir.tensor(self._new_dims, dtype=ir.DataType.INT64),
        )
        return op.Reshape(data, new_shape, allowzero=self._allowzero or None)


materialize_reshape_shape_rule = MaterializeReshapeShape.rule()

rules = RewriteRuleSet([materialize_reshape_shape_rule])
