# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose

# This is a basic test that verifies that a proposed expanded computation is equivalent to
# ORT's GQA (for the specific configuration considered).

# Simple GQA: no rotary embedding, no past key/value, no cos/sin cache, no seqlens/total_seqlen


class GQA1(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batchsize = 2
        self.seqlen = 8
        self.kv_seqlen = self.seqlen
        self.headsize = 16
        self.num_heads = 20
        self.kv_num_heads = 10
        self.hidden_size = self.headsize * self.num_heads
        self.kv_hidden_size = self.headsize * self.kv_num_heads
        self.num_groups = self.num_heads // self.kv_num_heads

    def random_inputs(self):
        B = self.batchsize
        S = self.seqlen
        D = self.hidden_size
        Dkv = self.kv_hidden_size
        query = np.random.rand(B, S, D).astype(np.float32)
        key = np.random.rand(B, S, Dkv).astype(np.float32)
        value = np.random.rand(B, S, Dkv).astype(np.float32)
        return {
            "query": query,
            "key": key,
            "value": value,
        }

    def fused_model(self):
        D = self.hidden_size
        Dkv = self.kv_hidden_size
        Dh = self.headsize
        H = self.num_heads
        Hkv = self.kv_num_heads
        return onnx.parser.parse_model(
            f"""
                <ir_version: 7, opset_import: [ "" : 18, "com.microsoft" : 1 ] >
                GQA (float[B, S, {D}] query, float[B, S, {Dkv}] key, float[B, S, {Dkv}] value)
                => (float[B, S, {D}] attn,
                    float[B, {Hkv}, S, {Dh}] past_key,
                    float[B, {Hkv}, S, {Dh}] past_value)
                {{
                    # Generate seqlens_k and total_seqlen inputs for GQA:
                    # In this test case, all batch elements have same sequence length.

                    total_seqlen = Shape <start=1, end=2> (query)
                    total_seqlen_int32 = Cast <to=6> (total_seqlen)
                    one = Constant <value = int32{{1}}> ()
                    total_seqlen_int32_minus_1 = Sub (total_seqlen_int32, one)
                    batchsize = Shape <start=0, end=1> (query)
                    seqlens_k = Tile (total_seqlen_int32_minus_1, batchsize)

                    attn, past_key, past_value = com.microsoft.GroupQueryAttention <num_heads = {H}, kv_num_heads = {Hkv}>
                        (query, key, value, , , seqlens_k, total_seqlen_int32)
                }}
            """
        )

    def expanded_model_script(self):
        scale_factor = math.sqrt(math.sqrt(self.headsize))
        minval = torch.finfo(torch.float32).min
        minval_tp = onnx.helper.make_tensor("minval", onnx.TensorProto.FLOAT, [1], [minval])
        H = [self.num_heads]
        Hkv = [self.kv_num_heads]
        Dh = [self.headsize]
        G = [self.num_groups]
        minus_1 = [-1]  # inferred dimension in Reshape op

        @script()
        def gqa(query, key, value):
            # Shapes used for Reshape ops. Note that we have a few different options on how shapes are
            # specified in an ONNX Reshape op (which supports special values 0 and -1 to propagate
            # existing dimension and one inferred dimension respectively). The following shapes are
            # based on what is observed in Phi models generated by the exporter.
            B = op.Shape(query, start=0, end=1)
            S = op.Shape(query, start=1, end=2)
            shape_BSHDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSHkvDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSD = op.Concat(B, S, minus_1, axis=0)
            shape_BHkvGSDh = op.Concat(B, Hkv, G, S, Dh, axis=0)
            shape_BHSDh = op.Concat(B, H, S, Dh, axis=0)
            shape_SS = op.Concat(S, S, axis=0)

            # First, get Q, K, V into right shapes. Inputs are 3D tensors in the BSD format.
            # D is different for Q and K/V (not reflected in the names, unfortunately).
            # We convert them into BHSDh (i.e., BHSd) format. In this version, we have only
            # one sequence length (S) for all Q, K, and V (with no cache).
            query_BSHDh = op.Reshape(query, shape_BSHDh)
            query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

            key_BSHkvDh = op.Reshape(key, shape_BSHkvDh)
            key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])
            key_BHkv1SDh = op.Unsqueeze(key_BHkvSDh, 2)
            key_BHkvGSDh = op.Expand(key_BHkv1SDh, shape_BHkvGSDh)
            key_BHSDh = op.Reshape(key_BHkvGSDh, shape_BHSDh)

            value_BSHkvDh = op.Reshape(value, shape_BSHkvDh)
            value_BHkvSDh = op.Transpose(value_BSHkvDh, perm=[0, 2, 1, 3])
            value_BHkv1SDh = op.Unsqueeze(value_BHkvSDh, 2)
            value_BHkvGSDh = op.Expand(value_BHkv1SDh, shape_BHkvGSDh)
            value_BHSDh = op.Reshape(value_BHkvGSDh, shape_BHSDh)

            # Generate a causal mask where every row looks like [0, 0, ..., /*diagonal=*/ 0, minval, minval, ...]
            all_min = op.ConstantOfShape(shape_SS, value=minval_tp)
            one = op.Constant(value_int=1)
            mask = op.Trilu(all_min, one, upper=1)

            # Now, compute attention:
            key_transposed = op.Transpose(key_BHSDh, perm=[0, 1, 3, 2])
            divisor = op.Constant(value_float=scale_factor)
            scaled_query = op.Div(query_BHSDh, divisor)
            scaled_key = op.Div(key_transposed, divisor)
            attn_score = op.MatMul(scaled_query, scaled_key)
            masked_attn_score = op.Add(attn_score, mask)
            attn_weight = op.Softmax(masked_attn_score, axis=-1)
            attention_BHSDh = op.MatMul(attn_weight, value_BHSDh)

            # Reshape back to original shape:
            attention_BSHDh = op.Transpose(attention_BHSDh, perm=[0, 2, 1, 3])
            attention_BSD = op.Reshape(attention_BSHDh, shape_BSD)
            return attention_BSD, key_BHkvSDh, value_BHkvSDh

        return gqa

    def expanded_model(self):
        D = self.hidden_size
        Dkv = self.kv_hidden_size
        Dh = self.headsize
        Hkv = self.kv_num_heads
        return self.expanded_model_script().to_model_proto(
            input_types=(FLOAT["B", "S", D], FLOAT["B", "S", Dkv], FLOAT["B", "S", Dkv]),
            output_types=(
                FLOAT["B", "S", D],
                FLOAT["B", Hkv, "S", Dh],
                FLOAT["B", Hkv, "S", Dh],
            ),
        )

    def test_equivalence(self):
        inputs = self.random_inputs()

        fused_model = self.fused_model()
        session = ort.InferenceSession(
            fused_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        outputs1 = session.run(None, inputs)

        expanded_model = self.expanded_model()
        session = ort.InferenceSession(
            expanded_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        outputs2 = session.run(None, inputs)

        self.assertEqual(len(outputs1), len(outputs2))
        assert_allclose(outputs1, outputs2)


# past_seqlen = 0
# total_seqlen = past_seqlen + S
# seqlens_k = np.array([total_seqlen-1], dtype=np.int32)
# total_seqlen_input = np.array(total_seqlen, dtype=np.int32)

if __name__ == "__main__":
    unittest.main()
