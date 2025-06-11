# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Union

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, pattern

Dim = Union[int, ir.SymbolicDim]


class SDPAPlus(pattern.RewriteRuleClassBase):
    def pattern(self, op, query, key, value, mask):
        # Pattern 1: Transpose 4D key directly
        key_transposed_1 = op.Transpose(key, perm=[0, 1, 3, 2])

        # Pattern 2: Transpose key after converting to 3D
        key_3d = op.Reshape(key, pattern.ANY_VALUE)
        key_3d_transposed = op.Transpose(key_3d, perm=[0, 2, 1])
        key_transposed_2 = op.Reshape(key_3d_transposed, pattern.ANY_VALUE)

        key_transposed = pattern.OrValue([key_transposed_1, key_transposed_2])
        return op.SDPA(
            query,
            key_transposed,
            value,
            mask,
            _domain="ai.onnxruntime.fusion",
            _outputs=["output"],
        )

    def rewrite(self, op, query, key, value, mask, **_):
        return op.SDPA(
            query, key, value, mask, transposed_key=False, _domain="ai.onnxruntime.fusion"
        )


sdpa_rule = pattern.RewriteRuleSet([SDPAPlus.rule()])

simplify_sdpa = _fusion_utils.apply_fusion_rules(sdpa_rule)
