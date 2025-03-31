# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import packaging.version

import numpy as np
import onnxruntime as ort

import onnxscript
import onnxscript.ir as ir
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
import onnxscript.rewriter.ort_fusions._core as xformers
from onnxscript.rewriter.ort_fusions._test_utils import ORT_VERSION, assert_allclose, ort_run

msft_op = onnxscript.values.Opset("com.microsoft", 1)


class TestAttentionFusion(unittest.TestCase):
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
        # self.num_groups = self.num_heads // self.kv_num_heads

    def random_inputs(self):
        B = self.batchsize
        S = self.seqlen
        D = self.input_hidden_size
        N = self.num_heads
        H = self.headsize
        D_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size
        input = np.random.rand(B, S, D).astype(np.float32)
        weight = np.random.rand(D, D_qkv).astype(np.float32)
        bias = np.random.rand(D_qkv).astype(np.float32)
        return {
            "input": input,
            "weight": weight,
            "bias": bias,
        }

    def model_with_mha_script(self):
        D = self.input_hidden_size
        Dh_q = self.q_hidden_size
        Dh_qk = self.q_hidden_size + self.k_hidden_size
        Dh_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size

        @script()
        def model_with_mha(input, weight, bias):
            QKV_no_bias = op.MatMul(input, weight)
            QKV = op.Add(QKV_no_bias, bias)

            query_BSDh = op.Slice(QKV, axes=[2], starts=[0], ends=[160])
            key_BSDh = op.Slice(QKV, axes=[2], starts=[160], ends=[320])
            value_BSDh = op.Slice(QKV, axes=[2], starts=[320], ends=[500])

            mha = msft_op.MultiHeadAttention(
                query_BSDh,
                key_BSDh,
                value_BSDh,
                num_heads=self.num_heads,
            )
            return mha

        model_proto = model_with_mha.to_model_proto(
            input_types=(FLOAT["B", "S", D], FLOAT[D, Dh_qkv], FLOAT[Dh_qkv]),
            output_types=(FLOAT["B", "S", self.v_hidden_size],),
        )
        model = ir.serde.deserialize_model(model_proto)
        return model


    def test_model_with_mha(self):
        inputs = self.random_inputs()
        mha_model = self.model_with_mha_script()

        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            original_outputs = ort_run("original", mha_model, inputs)

        # Fuse Attention
        attention_count = xformers.fuse_attention(mha_model, debug=True)
        self.assertGreater(attention_count, 0)

        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", mha_model, inputs)
            assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
