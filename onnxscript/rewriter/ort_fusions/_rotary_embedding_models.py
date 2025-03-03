# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small test case models for rotary embedding."""

import numpy

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64


# x: [B, H, S, E]
# position_ids: [B, S]
@script()
def _test_case_1_script(x: FLOAT[1, 4, 8, 8], position_ids: INT64[1, 8]) -> FLOAT[1, 4, 8, 8]:
    inv_freq = op.Constant(value_floats=[1.0, 2.0, 3.0, 4.0])
    inv_freq_3d = op.Unsqueeze(inv_freq, [0, 2])
    position_ids_expanded = op.Unsqueeze(position_ids, [1])  # => [B, 1, S]
    position_ids_float = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
    freqs = op.MatMul(inv_freq_3d, position_ids_float)  # [B, E, S]
    freqs = op.Transpose(freqs, perm=[0, 2, 1])  # [B, S, E]
    emb = op.Concat(freqs, freqs, axis=-1)
    cos = op.Cos(emb)
    sin = op.Sin(emb)
    cos_4d = op.Unsqueeze(cos, 1)
    sin_4d = op.Unsqueeze(sin, 1)

    x1 = op.Slice(x, [0], [4], [3], [1])
    x2 = op.Slice(x, [4], [8], [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    rotary_embedding = op.Add(x * cos_4d, rotated_x * sin_4d)
    return rotary_embedding


class _TestCase1:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = _test_case_1_script.to_model_proto()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "x": numpy.random.rand(1, 4, 8, 8).astype(numpy.float32),
                "position_ids": numpy.arange(8, dtype=numpy.int64).reshape(1, 8),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def test_case_1():
    return _TestCase1()


# x: [B, H, S, E]
# position_ids: [S]
@script()
def _test_case_2_script(x: FLOAT[1, 4, 8, 8], position_ids: INT64[8]) -> FLOAT[1, 4, 8, 8]:
    inv_freq = op.Constant(value_floats=[1.0, 2.0, 3.0, 4.0])
    inv_freq_3d = op.Unsqueeze(inv_freq, [0, 2])
    position_ids_expanded = op.Unsqueeze(position_ids, [0, 1])  # => [1, 1, S]
    position_ids_float = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
    freqs = op.MatMul(inv_freq_3d, position_ids_float)  # [B, E, S]
    freqs = op.Transpose(freqs, perm=[0, 2, 1])  # [B, S, E]
    emb = op.Concat(freqs, freqs, axis=-1)
    cos = op.Cos(emb)
    sin = op.Sin(emb)
    cos_4d = op.Unsqueeze(cos, 1)
    sin_4d = op.Unsqueeze(sin, 1)

    x1 = op.Slice(x, [0], [4], [3], [1])
    x2 = op.Slice(x, [4], [8], [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    rotary_embedding = op.Add(x * cos_4d, rotated_x * sin_4d)
    return rotary_embedding


class _TestCase2:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = _test_case_2_script.to_model_proto()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "x": numpy.random.rand(1, 4, 8, 8).astype(numpy.float32),
                "position_ids": numpy.arange(8, dtype=numpy.int64).reshape(8),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def test_case_2():
    return _TestCase2()
