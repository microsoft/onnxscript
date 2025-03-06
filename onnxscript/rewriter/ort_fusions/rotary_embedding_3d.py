# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

class RotaryEmbeddingFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, input_BSD, position_ids, cos, sin):
        # Reshape input from (B, S, D) to (B, S, H, D/H)
        input_BSHd = op.Reshape(
            input_BSD,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["query_BSHd"],
        )
        # Transpose input from (B, S, H, D/H) to (B, H, S, D/H)
        input_BHSd = op.Transpose(input_BSHd, perm=[0, 2, 1, 3])
        # Apply rotary embedding on 4D input
        output = op.RotaryEmbedding(input_BHSd, position_ids, cos, sin, _domain="com.microsoft")

    def check(self, op, input_BSD, position_ids, cos, sin):
        # Check that input is a 3D tensor
        if input_BSD is None or input_BSD.shape is None or len(input_BSD.shape) != 3:
            return False
        # Check that position_ids is a 2D tensor
        if position_ids is None or position_ids.shape is None or len(position_ids.shape) != 2:
            return False
        # Check that cos and sin are 1D tensors
        if cos is None or cos.shape is None or len(cos.shape) != 1:
            return False
        if sin is None or sin.shape is None or len(sin.shape) != 1:
            return False
        return True