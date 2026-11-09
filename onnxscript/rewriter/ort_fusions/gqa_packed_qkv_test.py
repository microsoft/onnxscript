# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx.helper
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference as shape_inference
import onnxruntime as ort
from parameterized import parameterized

import onnxscript
import onnxscript.optimizer
from onnxscript import FLOAT, INT32, script
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose
from onnxscript.rewriter.ort_fusions.gqa_packed_qkv import fuse_qkv_gqa

msft_op = onnxscript.values.Opset("com.microsoft", 1)

# Test case for fusion of separate query, key and value inputs
# into a single packed QKV input for the GroupQueryAttention operator.


class PackedQKVforGQAFusionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Config parameters
        self.batchsize = 1
        self.seqlen = 8
        self.kv_seqlen = self.seqlen
        self.past_seqlen = 16
        self.head_size = 16
        self.q_num_heads = 20
        self.kv_num_heads = 10

        # Computed config parameters
        self.q_hidden_size = self.head_size * self.q_num_heads
        self.kv_hidden_size = self.head_size * self.kv_num_heads
        self.hidden_size = self.q_hidden_size + self.kv_hidden_size + self.kv_hidden_size

        # Abbreviations
        B = self.batchsize
        S = self.seqlen
        P = self.past_seqlen
        D = self.hidden_size
        Dh = self.head_size
        Hkv = self.kv_num_heads
        total_seqlen = S + P
        max_seqlen = total_seqlen

        self.input_types = (
            FLOAT["B", "S", D],  # packed_qkv
            FLOAT["B", Hkv, "P", Dh],  # past_key
            FLOAT["B", Hkv, "P", Dh],  # past_value
            INT32["B"],  # seqlens_k
            INT32[1],  # total_sequence_length
            FLOAT["max_seqlen", Dh // 2],  # cos
            FLOAT["max_seqlen", Dh // 2],  # sin
        )
        self.output_types = (
            FLOAT["B", "S", self.q_hidden_size],  # attention
            FLOAT["B", Hkv, "T", Dh],  # present_key
            FLOAT["B", Hkv, "T", Dh],  # present_value
        )

        self.inputs = {
            "packed_qkv": np.random.rand(B, S, D).astype(np.float32),
            "past_key": np.random.rand(B, Hkv, P, Dh).astype(np.float32),
            "past_value": np.random.rand(B, Hkv, P, Dh).astype(np.float32),
            "seqlens_k": np.full((B,), total_seqlen - 1, dtype=np.int32),
            "total_sequence_length": np.array([total_seqlen], dtype=np.int32),
            "cos": np.random.rand(max_seqlen, Dh // 2).astype(np.float32),
            "sin": np.random.rand(max_seqlen, Dh // 2).astype(np.float32),
        }

    def source_model_script(self):
        Hq = self.q_num_heads
        Hkv = self.kv_num_heads

        @script()
        def gqa(packed_qkv, past_key, past_value, seqlens_k, total_sequence_length, cos, sin):
            # Slice packed_qkv into query, key and value
            query_BSD = op.Slice(packed_qkv, [0], [320], [2], [1])
            key_BSDkv = op.Slice(packed_qkv, [320], [480], [2], [1])
            value_BSDkv = op.Slice(packed_qkv, [480], [640], [2], [1])

            attn, past_key, past_value = msft_op.GroupQueryAttention(
                query_BSD,
                key_BSDkv,
                value_BSDkv,
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
            return attn, past_key, past_value

        return gqa

    def test_invalid_head_counts_do_not_fuse_or_divide_by_zero(self):
        for query_heads, kv_heads in ((0, 10), (20, 0), (19, 10)):
            with self.subTest(query_heads=query_heads, kv_heads=kv_heads):
                proto = self.source_model_script().to_model_proto(
                    input_types=self.input_types, output_types=self.output_types
                )
                model = shape_inference.infer_shapes(ir.from_proto(proto))
                attention = next(
                    node for node in model.graph if node.op_type == "GroupQueryAttention"
                )
                attention.attributes["num_heads"] = ir.AttrInt64("num_heads", query_heads)
                attention.attributes["kv_num_heads"] = ir.AttrInt64("kv_num_heads", kv_heads)
                self.assertEqual(fuse_qkv_gqa(model), 0)

    @parameterized.expand([("slices", False), ("split", True)])
    def test_fuse_packed_qkv_for_gqa(self, _, canonicalize):
        """
        Test that fusion from query, key and value to a packed QKV for GQA
        is successful on source model and produces an equivalent model.
        """
        inputs = self.inputs

        source_model = self.source_model_script().to_model_proto(
            input_types=self.input_types,
            output_types=self.output_types,
        )
        source_model.graph.node[-1].attribute.append(
            onnx.helper.make_attribute("scale", 0.125)
        )
        session = ort.InferenceSession(
            source_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        source_model_outputs = session.run(None, inputs)

        source_model_ir = ir.serde.from_proto(source_model)
        inferred_model = shape_inference.infer_shapes(source_model_ir)
        if canonicalize:
            onnxscript.optimizer.optimize(inferred_model)

        count = fuse_qkv_gqa(inferred_model, debug=True)
        self.assertEqual(count, 1)
        attention = next(
            node for node in inferred_model.graph if node.op_type == "GroupQueryAttention"
        )
        self.assertEqual(attention.attributes["scale"].as_float(), 0.125)

        fused_model = ir.serde.to_proto(inferred_model)
        session = ort.InferenceSession(
            fused_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        fused_model_outputs = session.run(None, inputs)

        self.assertEqual(len(fused_model_outputs), len(source_model_outputs))
        assert_allclose(fused_model_outputs, source_model_outputs)


if __name__ == "__main__":
    unittest.main()
