# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for GQA packed QKV fusion.

Adds coverage for: misaligned slice boundaries (negative).
"""

from __future__ import annotations

import unittest

import onnx_ir as ir
import onnx_ir.passes.common.shape_inference as shape_inference

import onnxscript
import onnxscript.optimizer
from onnxscript import FLOAT, INT32, script
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions.gqa_packed_qkv import fuse_qkv_gqa

msft_op = onnxscript.values.Opset("com.microsoft", 1)


class PackedQKVExtendedTest(unittest.TestCase):
    """Extended tests for GQA packed QKV fusion."""

    def test_misaligned_slice_boundaries_no_fusion(self):
        """Slice boundaries don't align with head sizes → should NOT fuse.

        With q_num_heads=20, kv_num_heads=10, head_size=16:
          hidden_size = 16*(20 + 2*10) = 640
          q: [0, 320), k: [320, 480), v: [480, 640)
        We intentionally misalign k slice to [300, 460) instead.
        """
        Hq = 20
        Hkv = 10
        Dh = 16
        D = Dh * (Hq + 2 * Hkv)  # 640

        @script()
        def gqa_misaligned(
            packed_qkv,
            past_key,
            past_value,
            seqlens_k,
            total_sequence_length,
            cos,
            sin,
        ):
            # Correct q slice
            query = op.Slice(packed_qkv, [0], [320], [2], [1])
            # WRONG: misaligned key slice (should be [320, 480))
            key = op.Slice(packed_qkv, [300], [460], [2], [1])
            # Value slice based on wrong offset
            value = op.Slice(packed_qkv, [460], [640], [2], [1])

            attn, pk, pv = msft_op.GroupQueryAttention(
                query,
                key,
                value,
                past_key,
                past_value,
                seqlens_k,
                total_sequence_length,
                cos,
                sin,
                num_heads=Hq,
                kv_num_heads=Hkv,
                do_rotary=1,
                rotary_interleaved=0,
            )
            return attn, pk, pv

        input_types = (
            FLOAT["B", "S", D],
            FLOAT["B", Hkv, "P", Dh],
            FLOAT["B", Hkv, "P", Dh],
            INT32["B"],
            INT32[1],
            FLOAT["max_seqlen", Dh // 2],
            FLOAT["max_seqlen", Dh // 2],
        )
        output_types = (
            FLOAT["B", "S", D],
            FLOAT["B", Hkv, "T", Dh],
            FLOAT["B", Hkv, "T", Dh],
        )

        model_proto = gqa_misaligned.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.from_proto(model_proto)
        inferred = shape_inference.infer_shapes(model)
        onnxscript.optimizer.optimize(inferred)

        count = fuse_qkv_gqa(inferred, debug=False)
        self.assertEqual(count, 0, "Should NOT fuse with misaligned slice boundaries.")


if __name__ == "__main__":
    unittest.main()
