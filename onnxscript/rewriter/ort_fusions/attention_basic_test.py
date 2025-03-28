# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

import onnxscript
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose

msft_op = onnxscript.values.Opset("com.microsoft", 1)

# This is a basic test that verifies that a 
# proposed expanded computation using packed matmul and ORT's MHA
# is equivalent to ORT's Attention (for the specific configuration considered).

# Simple Attention: no rotary embedding, no past key/value, no cos/sin cache


class AttentionEquivalence(unittest.TestCase):
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)
        self.batchsize = 2
        self.seqlen = 8
        self.headsize = 16
        self.num_heads = 20
        self.input_hidden_size = self.headsize * self.num_heads
        self.q_hidden_size = 160
        self.k_hidden_size = 160
        self.v_hidden_size = 180
        #self.num_groups = self.num_heads // self.kv_num_heads

    def random_inputs(self):
        B = self.batchsize
        S = self.seqlen
        D = self.input_hidden_size
        D_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size
        input = np.random.rand(B, S, D).astype(np.float32)
        weight = np.random.rand(D, D_qkv).astype(np.float32)
        bias = np.random.rand(D_qkv).astype(np.float32)
        return {
            "input": input,
            "weight": weight,
            "bias": bias,
        }

    def fused_model_script(self):
        H = self.num_heads
        H_qkv = [self.q_hidden_size, self.k_hidden_size, self.v_hidden_size]

        @script()
        def attention(input, weight, bias):
            attn = msft_op.Attention(
                input,
                weight,
                bias,
                num_heads=H,
                qkv_hidden_sizes=H_qkv,
            )
            return attn

        return attention

    def expanded_model_script(self):
        Dh_q = self.q_hidden_size
        Dh_qk = self.q_hidden_size + self.k_hidden_size
        Dh_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size
        @script()
        def attention(input, weight, bias):
            QKV_no_bias = op.MatMul(input, weight)
            QKV = op.Add(QKV_no_bias, bias)

            query_BSDh = op.Slice(QKV, axes=[2], starts=[0], ends=Dh_q)
            key_BSDh = op.Slice(QKV, axes=[2], starts=Dh_q, ends=Dh_qk)
            value_BSDh = op.Slice(QKV, axes=[2], starts=Dh_qk, ends=Dh_qkv)

            mha = msft_op.MultiHeadAttention(
                query_BSDh,
                key_BSDh,
                value_BSDh,
                num_heads=self.num_heads,
            )
            return mha

        return attention

    def to_proto(self, model_script):
        D = self.input_hidden_size
        D_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size
        return model_script.to_model_proto(
            input_types=(FLOAT["B", "S", D], FLOAT[D, D_qkv], FLOAT[D_qkv]),
            output_types=(
                FLOAT["B", "S", self.v_hidden_size],
            ),
        )

    def test_equivalence(self):
        inputs = self.random_inputs()

        fused_model = self.to_proto(self.fused_model_script())  # self.fused_model()
        session = ort.InferenceSession(
            fused_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        outputs1 = session.run(None, inputs)

        expanded_model = self.to_proto(self.expanded_model_script())  # self.expanded_model()
        session = ort.InferenceSession(
            expanded_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        outputs2 = session.run(None, inputs)

        self.assertEqual(len(outputs1), len(outputs2))
        assert_allclose(outputs1, outputs2, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
