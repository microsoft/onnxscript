# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for shape operations."""

from __future__ import annotations

import onnxscript.ir as ir
import onnxscript.rewriter.pattern as pattern


class ExtractDim(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(remove_nodes=False)

    """This is a pattern observed in causal mask generation that hinders fusion optimizations.
    It can be simplified away.
    """

    def pattern(self, op, x, dim0, dim1, dim2, dim3):
        shape = op.Concat(dim0, dim1, dim2, dim3, axis=0)
        reshaped = op.Reshape(x, shape, allowzero=0)
        transposed = op.Transpose(reshaped, perm=[0, 2, 1, 3])
        final_shape = op.Shape(transposed, _outputs=["final_shape"], start=0)
        final_dim = op.Slice(final_shape, [-2], [-1])
        return final_dim

    def check(self, context, dim0, dim1, dim2, dim3, final_shape, **_) -> bool:
        # All of the dimensions should have shape [1]
        for dim in (dim0, dim1, dim2, dim3):
            if dim.shape is None or dim.shape.dims != (1,):
                return False

        # The Shape op should return the full shape, not a slice of the shape.
        shape_node = final_shape.producer()
        if "end" in shape_node.attributes:
            return False
        if "start" in shape_node.attributes:
            start_attr = shape_node.attributes["start"]
            return isinstance(start_attr, ir.Attr) and start_attr.value == 0
        return True

    def rewrite(self, op, dim1, **_):
        return dim1


rules = pattern.RewriteRuleSet([ExtractDim.rule()])
