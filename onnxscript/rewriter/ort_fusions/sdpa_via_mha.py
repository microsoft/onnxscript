# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Union

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, pattern

Dim = Union[int, ir.SymbolicDim]


class SDPAImplementation(pattern.RewriteRuleClassBase):
    def pattern(self, op, query, key, value):
        return op.SDPA(
            query,
            key,
            value,
            key_format="BHSd",
            _allow_other_inputs=True,  # Mask is optional
            _outputs=["sdpa_output"],
            _domain="ai.onnxruntime._fusion",
        )

    def check(self, context, query, key, value, sdpa_output):
        bindings: dict[str, Dim] = {}
        _fusion_utils.check_shape(bindings, query, ["B", "H", "S", "Dh"])
        _fusion_utils.check_shape(bindings, key, ["B", "H", "Skv", "Dh"])
        _fusion_utils.check_shape(bindings, value, ["B", "H", "Skv", "Dv"])

        self._num_heads = bindings["H"]
        if not isinstance(self._num_heads, int):
            return False
        self._use_mask_broadcast = True  # TODO: optimize to avoid broadcast if not needed
        return isinstance(self._num_heads, int)

    def rewrite(self, op, query, key, value, sdpa_output):
        sdpa_node = sdpa_output.producer()
        scale = sdpa_node.attributes.get("scale", None)
        to_3d_shape = op.Constant(value_ints=[0, 0, -1])
        to_4d_shape = op.Constant(value_ints=[0, 0, self._num_heads, -1])
        query_3d = op.Reshape(op.Transpose(query, perm=[0, 2, 1, 3]), to_3d_shape)
        key_3d = op.Reshape(op.Transpose(key, perm=[0, 2, 1, 3]), to_3d_shape)
        value_3d = op.Reshape(op.Transpose(value, perm=[0, 2, 1, 3]), to_3d_shape)

        inputs = [query_3d, key_3d, value_3d]
        if len(sdpa_node.inputs) > 3:
            mask = sdpa_node.inputs[3]

            if self._use_mask_broadcast:
                one = op.Constant(value_ints=[1])
                query_length = op.Shape(query, start=2, end=3)
                shape_11S1 = op.Concat(one, one, query_length, one, axis=0)
                mask = op.Expand(mask, shape_11S1)

            inputs.extend([None, None, mask])

        output = op.MultiHeadAttention(
            *inputs,
            num_heads=self._num_heads,
            scale=scale,
            _domain="com.microsoft",
        )
        output_4d = op.Reshape(output, to_4d_shape)
        output = op.Transpose(output_4d, perm=[0, 2, 1, 3])
        return output


_rules = pattern.RewriteRuleSet([SDPAImplementation.rule()])

replace_sdpa_by_mha = _fusion_utils.apply_fusion_rules(_rules)
