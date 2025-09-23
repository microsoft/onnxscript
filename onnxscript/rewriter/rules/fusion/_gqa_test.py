# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import onnx
import onnx_ir as ir
from packaging import version

import onnxscript
import onnxscript.optimizer
import onnxscript.rewriter.testing
from onnxscript import FLOAT, script
from onnxscript.rewriter.rules.fusion._gqa import fuse_gqa

op = onnxscript.values.Opset("", 23)

H = [8]  # Number of attention heads
Hkv = [4]  # Number of key/value heads (H should be divisible by Hkv)
D = [64]  # Head size
G = [2]  # Number of groups


@script(ir_version=10)
def _gqa_script(
    query_BHSD: FLOAT[2, 8, 4, 64],  # B=2, H=8, S=4, D=64
    key_BHkvSD: FLOAT[2, 4, 4, 64],  # B=2, Hkv=4, S=4, D=64
    value_BHkvSD: FLOAT[2, 4, 4, 64],  # B=2, Hkv=4, S=4, D=64
    past_key_BHkvPD: FLOAT[2, 4, 8, 64],  # B=2, Hkv=4, P=8, D=64
    past_value_BHkvPD: FLOAT[2, 4, 8, 64],  # B=2, Hkv=4, P=8, D=64
) -> FLOAT[2, 8, 4, 64]:
    """Basic GQA pattern that should be fused into an Attention op."""

    # Concatenate past_key cache and current key
    present_key_BHkvStD = op.Concat(past_key_BHkvPD, key_BHkvSD, axis=-2)  # [B, Hkv, S+P, D]

    # Unsqueeze to add group dimension
    present_key_BHkv1StD = op.Unsqueeze(present_key_BHkvStD, 2)  # [B, Hkv, 1, S+P, D]

    # Calculate shapes dynamically
    B = op.Shape(query_BHSD, start=0, end=1)  # [B]
    T = op.Shape(present_key_BHkvStD, start=2, end=3)  # [S+P]

    # Create expand shape [B, Hkv, G, S+P, D]
    expand_shape = op.Concat(B, Hkv, G, T, D, axis=0)
    present_key_BHkvGStD = op.Expand(present_key_BHkv1StD, expand_shape)  # [B, Hkv, G, S+P, D]

    # Create reshape shape [B, H, S+P, D]
    reshape_shape = op.Concat(B, H, T, D, axis=0)
    present_key_BHStD = op.Reshape(present_key_BHkvGStD, reshape_shape)  # [B, H, S+P, D]

    # Same for value
    present_value_BHkvStD = op.Concat(
        past_value_BHkvPD, value_BHkvSD, axis=-2
    )  # [B, Hkv, S+P, D]
    present_value_BHkv1StD = op.Unsqueeze(present_value_BHkvStD, 2)  # [B, Hkv, 1, S+P, D]
    present_value_BHkvGStD = op.Expand(
        present_value_BHkv1StD, expand_shape
    )  # [B, Hkv, G, S+P, D]
    present_value_BHStD = op.Reshape(present_value_BHkvGStD, reshape_shape)  # [B, H, S+P, D]

    # Attention computation
    attention_BHSDh = op.Attention(
        query_BHSD,
        present_key_BHStD,
        present_value_BHStD,
    )

    return attention_BHSDh


class GQAFusionTest(unittest.TestCase):
    def test_basic_gqa_fusion(self):
        """Test basic GQA fusion pattern."""
        model_proto = _gqa_script.to_model_proto()

        # Apply GQA fusion
        model = ir.serde.deserialize_model(model_proto)
        onnxscript.optimizer.optimize(model)
        count = fuse_gqa(model)
        self.assertGreater(count, 0, "GQA fusion should have occurred")

        # We can't yet test numerical equivalence because of a bug in the op spec/implementation.
        onnx_ver = version.parse(onnx.__version__)
        if onnx_ver >= version.parse("1.19.1") and not (
            onnx_ver.is_prerelease or onnx_ver.is_devrelease
        ):
            # Only official releases >= 1.19.1
            onnxscript.optimizer.remove_unused_nodes(model)
            rewritten_model_proto = ir.serde.serialize_model(model)
            onnxscript.rewriter.testing.assert_numerically_equal(
                model_proto, rewritten_model_proto, use_reference=True
            )


if __name__ == "__main__":
    unittest.main()
