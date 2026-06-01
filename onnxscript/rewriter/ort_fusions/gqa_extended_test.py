# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for GQA ort_fusions fusion.

Adds coverage for: missing RotaryEmbedding (negative — no GQA fusion).
"""

from __future__ import annotations

import math
import unittest

import onnx
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference as shape_inference

from onnxscript import FLOAT, optimizer, script, values
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions.gqa import fuse_gqa
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa

msft_op = values.Opset("com.microsoft", 1)


class GQAOrtFusionExtendedTest(unittest.TestCase):
    """Extended negative test for GQA ort_fusion."""

    def test_no_rotary_embedding_no_gqa_fusion(self):
        """GQA source pattern without RotaryEmbedding → SDPA may fuse but GQA should NOT.

        The ort_fusions GQA rule requires RotaryEmbedding nodes to fuse the full GQA
        pattern (query/key rotary + kv-cache concat + expand + SDPA → GQA).
        Without rotary embedding, the GQA-specific fusion should not trigger.
        """
        head_size = 16
        num_heads = 20
        kv_num_heads = 10
        hidden_size = head_size * num_heads
        kv_hidden_size = head_size * kv_num_heads
        num_groups = num_heads // kv_num_heads

        H = [num_heads]
        Hkv = [kv_num_heads]
        Dh = [head_size]
        G = [num_groups]
        minus_1 = [-1]

        scale_factor = math.sqrt(math.sqrt(head_size))

        @script()
        def gqa_no_rotary(query, key, value, past_key, past_value):
            B = op.Shape(query, start=0, end=1)
            S = op.Shape(query, start=1, end=2)
            past_seq_length = op.Shape(past_key, start=2, end=3)
            total_seq_length = op.Add(past_seq_length, S)

            shape_BSHDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSHkvDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSD = op.Concat(B, S, minus_1, axis=0)
            shape_BHkvGSDh = op.Concat(B, Hkv, G, total_seq_length, Dh, axis=0)
            shape_BHSDh = op.Concat(B, H, total_seq_length, Dh, axis=0)

            query_BSHDh = op.Reshape(query, shape_BSHDh)
            query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

            key_BSHkvDh = op.Reshape(key, shape_BSHkvDh)
            key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])

            value_BSHkvDh = op.Reshape(value, shape_BSHkvDh)
            value_BHkvSDh = op.Transpose(value_BSHkvDh, perm=[0, 2, 1, 3])

            # NO RotaryEmbedding here — just use key/query directly
            key_seq = op.Concat(past_key, key_BHkvSDh, axis=-2)
            value_seq = op.Concat(past_value, value_BHkvSDh, axis=-2)

            key_unsq = op.Unsqueeze(key_seq, [2])
            key_exp = op.Expand(key_unsq, shape_BHkvGSDh)
            key_rsh = op.Reshape(key_exp, shape_BHSDh)

            value_unsq = op.Unsqueeze(value_seq, [2])
            value_exp = op.Expand(value_unsq, shape_BHkvGSDh)
            value_rsh = op.Reshape(value_exp, shape_BHSDh)

            # Attention
            key_t = op.Transpose(key_rsh, perm=[0, 1, 3, 2])
            divisor = op.Constant(value_float=scale_factor)
            scaled_q = op.Div(query_BHSDh, divisor)
            scaled_k = op.Div(key_t, divisor)
            score = op.MatMul(scaled_q, scaled_k)
            weight = op.Softmax(score, axis=-1)
            attn = op.MatMul(weight, value_rsh)

            attn_t = op.Transpose(attn, perm=[0, 2, 1, 3])
            attn_out = op.Reshape(attn_t, shape_BSD)

            return attn_out, key_seq, value_seq

        D = hidden_size
        Dkv_val = kv_hidden_size
        Dh_val = head_size
        Hkv_val = kv_num_heads

        input_types = (
            FLOAT["B", "S", D],
            FLOAT["B", "S", Dkv_val],
            FLOAT["B", "S", Dkv_val],
            FLOAT["B", Hkv_val, "P", Dh_val],
            FLOAT["B", Hkv_val, "P", Dh_val],
        )
        output_types = (
            FLOAT["B", "S", D],
            FLOAT["B", Hkv_val, "T", Dh_val],
            FLOAT["B", Hkv_val, "T", Dh_val],
        )

        source_model = gqa_no_rotary.to_model_proto(
            input_types=input_types,
            output_types=output_types,
        )

        # Add value_info for shapes needed by fusion
        query_BSHDh_vi = onnx.helper.make_tensor_value_info(
            "query_BSHDh", onnx.TensorProto.FLOAT, ["B", "S", num_heads, head_size]
        )
        key_BSHkvDh_vi = onnx.helper.make_tensor_value_info(
            "key_BSHkvDh", onnx.TensorProto.FLOAT, ["B", "S", kv_num_heads, head_size]
        )
        source_model.graph.value_info.extend([query_BSHDh_vi, key_BSHkvDh_vi])

        model = ir.serde.from_proto(source_model)
        inferred = shape_inference.infer_shapes(model)
        optimizer.optimize(inferred)

        # SDPA might fuse, but GQA should not (no RotaryEmbedding)
        fuse_sdpa(inferred, debug=False)
        count = fuse_gqa(inferred, debug=False)
        self.assertEqual(count, 0, "GQA fusion should NOT succeed without RotaryEmbedding.")


if __name__ == "__main__":
    unittest.main()
