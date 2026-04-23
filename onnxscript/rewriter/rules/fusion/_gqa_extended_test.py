# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for GQA fusion rule (rules/fusion variant).

Adds coverage for: mismatched group parameter (negative).
"""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir
from packaging import version

from onnxscript import FLOAT, optimizer, script, values
from onnxscript.rewriter.rules.fusion._gqa import fuse_gqa
from onnxscript.rewriter.testing import assert_numerically_equal

op = values.Opset("", 23)

# Config: H=8, Hkv=4, D=64, G=2
H = [8]
Hkv = [4]
D = [64]
G_CORRECT = [2]  # H / Hkv = 8 / 4 = 2
G_WRONG = [3]  # Wrong group count


@script(ir_version=10)
def _gqa_wrong_group(
    query_BHSD: FLOAT[2, 8, 4, 64],
    key_BHkvSD: FLOAT[2, 4, 4, 64],
    value_BHkvSD: FLOAT[2, 4, 4, 64],
    past_key_BHkvPD: FLOAT[2, 4, 8, 64],
    past_value_BHkvPD: FLOAT[2, 4, 8, 64],
) -> FLOAT[2, 8, 4, 64]:
    """GQA pattern with wrong group count — should NOT fuse."""
    present_key_BHkvStD = op.Concat(past_key_BHkvPD, key_BHkvSD, axis=-2)
    present_key_BHkv1StD = op.Unsqueeze(present_key_BHkvStD, 2)
    B = op.Shape(query_BHSD, start=0, end=1)
    T = op.Shape(present_key_BHkvStD, start=2, end=3)

    # Use G_WRONG instead of G_CORRECT — expand shape will be [B, Hkv, 3, S+P, D]
    expand_shape = op.Concat(B, Hkv, G_WRONG, T, D, axis=0)
    present_key_BHkvGStD = op.Expand(present_key_BHkv1StD, expand_shape)

    # Reshape target would be [B, Hkv*3, S+P, D] = [B, 12, ...] not [B, 8, ...]
    H_wrong = [12]  # Hkv * G_WRONG = 4 * 3 = 12
    reshape_shape = op.Concat(B, H_wrong, T, D, axis=0)
    present_key_BHStD = op.Reshape(present_key_BHkvGStD, reshape_shape)

    present_value_BHkvStD = op.Concat(past_value_BHkvPD, value_BHkvSD, axis=-2)
    present_value_BHkv1StD = op.Unsqueeze(present_value_BHkvStD, 2)
    present_value_BHkvGStD = op.Expand(present_value_BHkv1StD, expand_shape)
    present_value_BHStD = op.Reshape(present_value_BHkvGStD, reshape_shape)

    attention_BHSDh = op.Attention(
        query_BHSD,
        present_key_BHStD,
        present_value_BHStD,
    )
    return attention_BHSDh


class GQAFusionExtendedTest(unittest.TestCase):
    """Extended tests for GQA fusion."""

    def test_gqa_wrong_group_count_no_fusion(self):
        """Group count doesn't match H/Hkv — GQA fusion should fail."""
        model_proto = _gqa_wrong_group.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        optimizer.optimize(model)
        count = fuse_gqa(model)
        self.assertEqual(
            count, 0, "GQA fusion should NOT succeed with mismatched group count."
        )

    def test_basic_gqa_positive_with_different_head_config(self):
        """GQA with H=6, Hkv=2, G=3, D=32 — should still fuse."""
        H_cfg = [6]
        Hkv_cfg = [2]
        D_cfg = [32]
        G_cfg = [3]

        @script(ir_version=10)
        def gqa_alt(
            q: FLOAT[1, 6, 4, 32],
            k: FLOAT[1, 2, 4, 32],
            v: FLOAT[1, 2, 4, 32],
            pk: FLOAT[1, 2, 8, 32],
            pv: FLOAT[1, 2, 8, 32],
        ) -> FLOAT[1, 6, 4, 32]:
            pk_cat = op.Concat(pk, k, axis=-2)
            pk_unsq = op.Unsqueeze(pk_cat, 2)
            B = op.Shape(q, start=0, end=1)
            T = op.Shape(pk_cat, start=2, end=3)
            expand_shape = op.Concat(B, Hkv_cfg, G_cfg, T, D_cfg, axis=0)
            pk_exp = op.Expand(pk_unsq, expand_shape)
            reshape_shape = op.Concat(B, H_cfg, T, D_cfg, axis=0)
            pk_rsh = op.Reshape(pk_exp, reshape_shape)

            pv_cat = op.Concat(pv, v, axis=-2)
            pv_unsq = op.Unsqueeze(pv_cat, 2)
            pv_exp = op.Expand(pv_unsq, expand_shape)
            pv_rsh = op.Reshape(pv_exp, reshape_shape)

            attn = op.Attention(q, pk_rsh, pv_rsh)
            return attn

        model_proto = gqa_alt.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        optimizer.optimize(model)
        count = fuse_gqa(model)
        self.assertGreater(count, 0, "GQA fusion should succeed with H=6, Hkv=2, G=3, D=32.")

        # Verify numerical equivalence if onnx version supports it
        onnx_ver = version.parse(onnx.__version__)
        if onnx_ver >= version.parse("1.19.1") and not (
            onnx_ver.is_prerelease or onnx_ver.is_devrelease
        ):
            optimizer.remove_unused_nodes(model)
            rewritten_proto = ir.serde.serialize_model(model)
            assert_numerically_equal(model_proto, rewritten_proto, use_reference=True)


if __name__ == "__main__":
    unittest.main()
