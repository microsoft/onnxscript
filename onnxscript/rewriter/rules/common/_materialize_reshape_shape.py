# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Materialize Reshape shape inputs proven by shape computations.

Resolve Shape, Gather and Concat computations rather than relying on output
annotations, which do not constrain an arbitrary runtime shape input.

- Fully static target shape → constant with exact dims.
- Exactly one symbolic dim → replace it with ``-1`` (Reshape infers it).

Before opset 14, only shapes without zero dims can be materialized because
Reshape does not support literal zeros via ``allowzero``.
"""

from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _shape_value(value: ir.Value | None) -> list[int | ir.SymbolicDim] | None:
    if value is None or value.is_graph_input():
        return None
    constant = ir_utils.get_numpy_value(value)
    if constant is not None:
        if value.dtype == ir.DataType.INT64 and constant.ndim == 1:
            return constant.tolist()
        return None
    node = value.producer()
    if node is None or node.domain != "":
        return None
    if any(attr.is_ref() for attr in node.attributes.values()):
        return None
    if node.op_type == "Shape":
        source = node.inputs[0]
        if source is None or source.shape is None:
            return None
        start = node.attributes.get_int("start", 0)
        end = node.attributes.get_int("end", None)
        return list(source.shape)[start:end]
    if node.op_type == "Gather":
        if node.attributes.get_int("axis", 0) not in (0, -1):
            return None
        source_dims = _shape_value(node.inputs[0])
        indices_value = node.inputs[1]
        if indices_value is None or indices_value.is_graph_input():
            return None
        indices = ir_utils.get_numpy_value(indices_value)
        if (
            source_dims is None
            or indices is None
            or indices.ndim != 1
            or indices_value.dtype not in (ir.DataType.INT32, ir.DataType.INT64)
        ):
            return None
        if not all(-len(source_dims) <= i < len(source_dims) for i in indices):
            return None
        return [source_dims[int(i)] for i in indices]
    if node.op_type == "Concat":
        if node.attributes.get_int("axis") not in (0, -1):
            return None
        dims: list[int | ir.SymbolicDim] = []
        for input in node.inputs:
            input_dims = _shape_value(input)
            if input_dims is None:
                return None
            dims.extend(input_dims)
        return dims
    return None


class MaterializeReshapeShape(RewriteRuleClassBase):
    """Replace a proven dynamic Reshape shape input with a constant."""

    def pattern(self, op, data, shape):
        return op.Reshape(data, shape)

    def check(self, context, data: ir.Value, shape: ir.Value) -> MatchResult:
        check_result = MatchResult()

        # Shape input must not already be a constant
        if ir_utils.get_numpy_value(shape) is not None:
            return check_result.fail("Shape input is already a constant.")

        dims = _shape_value(shape)
        if dims is None:
            return check_result.fail("Shape input values cannot be proven.")

        allowzero_attr = context.root.attributes.get("allowzero")
        if allowzero_attr is not None and allowzero_attr.is_ref():
            return check_result.fail("Cannot resolve a reference allowzero attribute.")
        allowzero = context.root.attributes.get_int("allowzero", 0)
        if allowzero == 0:
            for i, dim in enumerate(dims):
                if isinstance(dim, int) and dim == 0:
                    if data.shape is None or i >= len(data.shape):
                        return check_result.fail("Cannot resolve a copied input dimension.")
                    dims[i] = data.shape[i]

        sym_count = sum(1 for d in dims if not isinstance(d, int) or d == -1)
        if any(isinstance(d, int) and d < -1 for d in dims):
            return check_result.fail("Shape input contains an invalid negative dimension.")

        if sym_count <= 1:
            self._new_dims = [-1 if not isinstance(d, int) else int(d) for d in dims]
        else:
            return check_result.fail(
                f"Shape input has {sym_count} inferred or symbolic dims, cannot materialize."
            )

        # The rewrite uses allowzero=1, which does not allow both 0 and -1 in the shape.
        if sym_count and 0 in self._new_dims:
            return check_result.fail("Cannot combine a 0 dim with -1 when allowzero=1.")

        opset = context.graph_or_function.opset_imports.get("")
        if opset is None:
            opset = context.model.graph.opset_imports[""]
        self._allowzero = 1 if opset >= 14 else None
        if self._allowzero is None and 0 in self._new_dims:
            return check_result.fail("Literal zero dims require Reshape opset 14 or newer.")

        return check_result

    def rewrite(self, op, data: ir.Value, shape: ir.Value):
        new_shape = op.Constant(
            value=ir.tensor(self._new_dims, dtype=ir.DataType.INT64),
        )
        # Resolved zeros are literal dimensions, not copies from the input.
        return op.Reshape(data, new_shape, allowzero=self._allowzero)


materialize_reshape_shape_rule = MaterializeReshapeShape.rule()

rules = RewriteRuleSet([materialize_reshape_shape_rule])
