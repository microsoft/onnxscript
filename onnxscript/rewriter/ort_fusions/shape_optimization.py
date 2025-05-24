# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for shape operations."""

from __future__ import annotations

import onnxscript.ir as ir
import onnxscript.rewriter._ir_utils as _ir_utils
import onnxscript.rewriter.pattern as pattern


class ExtractDim(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(remove_nodes=False)

    """This is a pattern observed in causal mask generation that hinders fusion optimizations.
    It can be simplified away.
    """

    def pattern(self, op, x, dim0, dim1, dim2, dim3, start, end):
        shape = op.Concat(dim0, dim1, dim2, dim3, axis=0)
        # Note: The allowzero=1 attribute enables us to infer that the shape of the
        # reshaped tensor is the same as the value of the shape parameter below.
        # Otherwise, we need to know that there are no zeros in the value of "shape"
        # for this optimization to be valid.
        reshaped = op.Reshape(x, shape, allowzero=1)
        transposed = op.Transpose(reshaped, perm=[0, 2, 1, 3])
        final_shape = op.Shape(transposed, _outputs=["final_shape"])
        final_dim = op.Slice(final_shape, start, end)
        return final_dim

    def check(self, context, dim0, dim1, dim2, dim3, final_shape, start, end, **_) -> bool:
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
            if not (isinstance(start_attr, ir.Attr) and start_attr.value == 0):
                return False
        self._start_val = _ir_utils.get_singleton_value(start)
        self._end_val = _ir_utils.get_singleton_value(end)
        if self._start_val is None or self._end_val is None:
            return False
        return True

    def rewrite(self, op, dim0, dim1, dim2, dim3, **_):
        transposed_dims = [dim0, dim2, dim1, dim3]
        sliced_result = transposed_dims[self._start_val : self._end_val]
        if len(sliced_result) == 0:
            return op.Constant(value_ints=[])
        if len(sliced_result) == 1:
            return op.Identity(sliced_result[0])
        return op.Concat(*sliced_result, axis=0)


rules = pattern.RewriteRuleSet([ExtractDim.rule()])
