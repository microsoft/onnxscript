# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SDPA fusion test cases."""

from __future__ import annotations

import math
import unittest

import numpy
import parameterized

import onnxscript.ir as ir
import onnxscript.optimizer
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa

B = 2  # batch size
N = 4  # number of heads
S = 8  # sequence length
H = 128  # head size
SCALE_FACTOR = math.sqrt(H)
MUL_SCALE_FACTOR = 1.0 / SCALE_FACTOR
SQRT_SCALE_FACTOR = math.sqrt(SCALE_FACTOR)
SQRT_MUL_SCALE_FACTOR = math.sqrt(MUL_SCALE_FACTOR)
CUSTOM_SCALE_FACTOR = 2.0


@script()
def _unmasked_pre_div_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=SQRT_SCALE_FACTOR)
    scaled_query = op.Div(query, divisor)
    scaled_key = op.Div(key_transposed, divisor)
    attn_score = op.MatMul(scaled_query, scaled_key)
    attn_weight = op.Softmax(attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _unmasked_pre_mul_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=SQRT_MUL_SCALE_FACTOR)
    scaled_query = op.Mul(query, multiplier)
    scaled_key = op.Mul(key_transposed, multiplier)
    attn_score = op.MatMul(scaled_query, scaled_key)
    attn_weight = op.Softmax(attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _unmasked_post_div_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Div(attn_score, divisor)
    attn_weight = op.Softmax(scaled_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _unmasked_post_mul_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=MUL_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Mul(attn_score, multiplier)
    attn_weight = op.Softmax(scaled_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_pre_div_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    scaled_query = op.Div(query, divisor)
    scaled_key = op.Div(key_transposed, divisor)
    attn_score = op.MatMul(scaled_query, scaled_key)
    attn_weight = op.Softmax(attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_pre_mul_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    scaled_query = op.Mul(query, multiplier)
    scaled_key = op.Mul(key_transposed, multiplier)
    attn_score = op.MatMul(scaled_query, scaled_key)
    attn_weight = op.Softmax(attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_multi_scale_pre_mul_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier_q = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    multiplier_k = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    scaled_query = op.Mul(query, multiplier_q)
    scaled_key = op.Mul(key_transposed, multiplier_k)
    attn_score = op.MatMul(scaled_query, scaled_key)
    attn_weight = op.Softmax(attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_post_div_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Div(attn_score, divisor)
    attn_weight = op.Softmax(scaled_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_post_mul_sdpa_script(query, key, value):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Mul(attn_score, multiplier)
    attn_weight = op.Softmax(scaled_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _masked_pre_div_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=SQRT_SCALE_FACTOR)
    scaled_query = op.Div(query, divisor)
    scaled_key = op.Div(key_transposed, divisor)
    attn_score = op.MatMul(scaled_query, scaled_key)
    masked_attn_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _masked_pre_mul_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=SQRT_MUL_SCALE_FACTOR)
    scaled_query = op.Mul(query, multiplier)
    scaled_key = op.Mul(key_transposed, multiplier)
    attn_score = op.MatMul(scaled_query, scaled_key)
    masked_attn_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _masked_post_div_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Div(attn_score, divisor)
    masked_attn_score = op.Add(scaled_attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _masked_post_mul_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=MUL_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Mul(attn_score, multiplier)
    masked_attn_score = op.Add(scaled_attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_pre_div_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    scaled_query = op.Div(query, divisor)
    scaled_key = op.Div(key_transposed, divisor)
    attn_score = op.MatMul(scaled_query, scaled_key)
    masked_attn_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_pre_mul_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    scaled_query = op.Mul(query, multiplier)
    scaled_key = op.Mul(key_transposed, multiplier)
    attn_score = op.MatMul(scaled_query, scaled_key)
    masked_attn_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_post_div_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    divisor = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Div(attn_score, divisor)
    masked_attn_score = op.Add(scaled_attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


@script()
def _custom_scale_post_mul_sdpa_script(query, key, value, mask):
    key_transposed = op.Transpose(key, perm=[0, 1, 3, 2])
    multiplier = op.Constant(value_float=CUSTOM_SCALE_FACTOR)
    attn_score = op.MatMul(query, key_transposed)
    scaled_attn_score = op.Mul(attn_score, multiplier)
    masked_attn_score = op.Add(scaled_attn_score, mask)
    attn_weight = op.Softmax(masked_attn_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


class SDPATestCase:
    def __init__(self, script_func):
        self.script_func = script_func

    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            qkv_type = FLOAT[B, N, S, H]
            mask_type = FLOAT[B, N, S, S]
            model_proto = self.script_func.to_model_proto(
                input_types=[qkv_type, qkv_type, qkv_type, mask_type], output_types=[qkv_type]
            )
            self._onnx_model = ir.serde.deserialize_model(model_proto)
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "query": numpy.random.rand(B, N, S, H).astype(numpy.float32),
                "key": numpy.random.rand(B, N, S, H).astype(numpy.float32),
                "value": numpy.random.rand(B, N, S, H).astype(numpy.float32),
                "mask": numpy.random.rand(B, N, S, S).astype(numpy.float32),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


class TestSDPAFusion(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("unmasked_pre_div", _unmasked_pre_div_sdpa_script),
            ("unmasked_pre_mul", _unmasked_pre_mul_sdpa_script),
            ("unmasked_post_div", _unmasked_post_div_sdpa_script),
            ("unmasked_post_mul", _unmasked_post_mul_sdpa_script),
            ("pre_div", _masked_pre_div_sdpa_script),
            ("pre_mul", _masked_pre_mul_sdpa_script),
            ("post_div", _masked_post_div_sdpa_script),
            ("post_mul", _masked_post_mul_sdpa_script),
            ("custom_scale_post_mul", _custom_scale_post_mul_sdpa_script),
            ("custom_scale_post_div", _custom_scale_post_div_sdpa_script),
            ("custom_scale_pre_mul", _custom_scale_pre_mul_sdpa_script),
            ("custom_scale_pre_div", _custom_scale_pre_div_sdpa_script),
            ("custom_scale_post_mul_masked", _custom_scale_post_mul_sdpa_script),
            ("custom_scale_post_div_masked", _custom_scale_post_div_sdpa_script),
            ("custom_scale_pre_mul_masked", _custom_scale_pre_mul_sdpa_script),
            ("custom_scale_pre_div_masked", _custom_scale_pre_div_sdpa_script),
            (
                "_custom_multi_scale_pre_mul_sdpa_script",
                _custom_multi_scale_pre_mul_sdpa_script,
            ),
        ]
    )
    def test_sdpa_fusion(self, name, script_func):
        test_case = SDPATestCase(script_func)
        model = test_case.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        # inputs = test_case.get_ort_inputs()
        # original_outputs = ort_run("original", model, inputs)

        count = fuse_sdpa(model, debug=True)
        self.assertGreater(count, 0)

        # Check that the fusion was successful
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SDPA", op_types)

        # Ensure that the scale of the SDPA node is set correctly
        sdpa_node = next(n for n in model.graph if n.op_type == "SDPA")
        self.assertEqual(sdpa_node.op_type, "SDPA")

        if "custom" in name:
            self.assertIsNotNone(sdpa_node.attributes.get("scale"))
            scale_factor = sdpa_node.attributes["scale"].value
            self.assertIsNotNone(scale_factor)
            if "pre" in name:
                self.assertEqual(scale_factor, CUSTOM_SCALE_FACTOR * CUSTOM_SCALE_FACTOR)
            elif "post" in name:
                self.assertEqual(scale_factor, CUSTOM_SCALE_FACTOR)
        else:
            # These tests are for the default scaling factors, no scale factor is passed to SDPA
            # pattern rewriting check functions should be sufficient to check if expected value
            # of scale_factor (is =default_scaling_factor)
            self.assertIsNone(sdpa_node.attributes.get("scale"))

        # new_outputs = ort_run("optimized", model, inputs)
        # assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
