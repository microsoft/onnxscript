# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

Dim = Union[int, ir.SymbolicDim]

class SDPAImplementation(pattern.RewriteRuleClassBase):

    def pattern(self,  op, query, key_transposed, value):
        return op.SDPA(query, key_transposed, value, _outputs=["sdpa_output"], _domain="ai.onnxruntime.fusion")

    def check(self, context, query, key_transposed, value, **_):
        bindings: dict[str, Dim] = {}
        _fusion_utils.check_shape(bindings, query, ["B", "H", "S", "Dh"])
        _fusion_utils.check_shape(bindings, key_transposed, ["B", "H", "Dh", "Skv"])
        _fusion_utils.check_shape(bindings, value, ["B", "H", "Skv", "Dv"])

        self._num_heads = bindings["H"]
        return isinstance(self._num_heads, int)

    def rewrite(self, op, query, key_transposed, value, sdpa_output):
        sdpa_node = sdpa_output.producer()
        scale = sdpa_node.attributes.get("scale", None)
        to_3d_shape = op.Constant(value_ints=[0, 0, -1])
        to_4d_shape = op.Constant(value_ints=[0, 0, self._num_heads, -1])
        query_3d = op.Reshape(op.Transpose(query, perm=[0, 2, 1, 3]), to_3d_shape)
        key_3d= op.Reshape(op.Transpose(key_transposed, perm=[0, 3, 1, 2]), to_3d_shape)
        value_3d = op.Reshape(op.Transpose(value, perm=[0, 2, 1, 3]), to_3d_shape)
        # Should we introduce non-causal mask?
        output = op.MultiHeadAttention(
            query_3d,
            key_3d,
            value_3d,
            num_heads=self._num_heads,
            scale=scale,
            _domain="com.microsoft"
        )
        output_4d = op.Reshape(output, to_4d_shape)
        output = op.Transpose(output_4d, perm=[0, 2, 1, 3])
        return output

_rules = pattern.RewriteRuleSet([SDPAImplementation.rule()])

replace_sdpa_by_mha = _fusion_utils.apply_fusion_rules(_rules)
        