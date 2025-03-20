# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SDPA fusion test cases."""

from __future__ import annotations

import math
import unittest

import numpy

import onnxscript.ir as ir
import onnxscript.optimizer
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa

batchsize = 2
num_heads = 4
seq_len = 8
headsize = 128
scale_factor = math.sqrt(headsize)
sqrt_scale_factor = math.sqrt(scale_factor)


@script()
def _masked_pre_div_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=sqrt_scale_factor)
    scaled_query = op.Div(query, divisor)
    scaled_key = op.Div(key_transposed, divisor)
    attn_score = op.MatMul(scaled_query, scaled_key)
    masked_attn_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


class _MaskedPreDivSDPATestCase:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            qkv_type = FLOAT[batchsize, num_heads, seq_len, headsize]
            mask_type = FLOAT[batchsize, num_heads, seq_len, seq_len]
            model_proto = _masked_pre_div_sdpa_script.to_model_proto(
                input_types=[qkv_type, qkv_type, qkv_type, mask_type], output_types=[qkv_type]
            )
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "query": numpy.random.rand(batchsize, num_heads, seq_len, headsize).astype(
                    numpy.float32
                ),
                "key": numpy.random.rand(batchsize, num_heads, seq_len, headsize).astype(
                    numpy.float32
                ),
                "value": numpy.random.rand(batchsize, num_heads, seq_len, headsize).astype(
                    numpy.float32
                ),
                "mask": numpy.random.rand(batchsize, num_heads, seq_len, seq_len).astype(
                    numpy.float32
                ),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


class TestSDPAFusion(unittest.TestCase):
    def test_sdpa_fusion(self):
        test = _MaskedPreDivSDPATestCase()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        # inputs = test.get_ort_inputs()
        # original_outputs = ort_run("original", model, inputs)

        count = fuse_sdpa(model)
        self.assertGreater(count, 0)

        # Check that the fusion was successful
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SDPA", op_types)

        # new_outputs = ort_run("optimized", model, inputs)
        # assert_allclose(new_outputs, original_outputs)
