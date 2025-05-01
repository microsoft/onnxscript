# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small test case models for rotary embedding."""

import numpy

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64

# A simple rotary embedding example


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


# A simple rotary embedding example with 1D position_ids
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


# A partial rotary embedding example:

rotary_embedding_dim = 32  # Abbreviated as "rd" in shape descriptors below
half_rotary_embedding_dim = rotary_embedding_dim // 2
# A random inverse frequency tensor for the sake of this example.
inv_freqs_value = numpy.random.rand(1, half_rotary_embedding_dim, 1).astype(numpy.float32)


@script()
def _partial_rotary_script(position_ids, query):
    inv_freqs = op.Constant(value=inv_freqs_value)  # [1, rd/2, 1]
    position_ids_3d = op.Unsqueeze(position_ids, 1)  # [B, 1, S]
    position_ids_3d_float = op.Cast(position_ids_3d, to=1)
    matmul = op.MatMul(inv_freqs, position_ids_3d_float)  # [B, rd/2, S]
    transpose = op.Transpose(matmul, perm=[0, 2, 1])  # [B, S, rd/2]
    cat = op.Concat(transpose, transpose, axis=-1)  # [B, S, rd]
    cos_3d = op.Cos(cat)  # [B, S, rd]
    sin_3d = op.Sin(cat)  # [B, S, rd]
    # Split the query for partial embedding
    to_embed = op.Slice(query, [0], [32], [3], [1])
    unembedded = op.Slice(query, [32], [9223372036854775807], [3], [1])
    cos_4d = op.Unsqueeze(cos_3d, 1)  # [B, 1, S, rd]
    sin_4d = op.Unsqueeze(sin_3d, 1)  # [B, 1, S, rd]
    # Compute rotation of X as X * cos + rotate_half(X) * sin, where rotate_half(X)
    # essentially represents X rotated by 90 degrees
    to_embed_times_cos = op.Mul(to_embed, cos_4d)
    to_embed_x = op.Slice(to_embed, [0], [16], [3], [1])
    to_embed_y = op.Slice(to_embed, [16], [9223372036854775807], [3], [1])
    minus_to_embed_y = op.Neg(to_embed_y)
    to_embed_rotated_90 = op.Concat(minus_to_embed_y, to_embed_x, axis=-1)
    to_embed_rotated_90_times_sin = op.Mul(to_embed_rotated_90, sin_4d)
    embedded = op.Add(to_embed_times_cos, to_embed_rotated_90_times_sin)
    final = op.Concat(embedded, unembedded, axis=-1)
    return final


class _PartialRotaryTestCase:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = _partial_rotary_script.to_model_proto(
                input_types=(
                    INT64["Batchsize", "Sequence"],
                    FLOAT["Batchsize", 32, "Sequence", 80],
                ),
                output_types=(FLOAT["Batchsize", 32, "Sequence", 80],),
            )
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "query": numpy.random.rand(1, 32, 8, 80).astype(numpy.float32),
                "position_ids": numpy.arange(8, dtype=numpy.int64).reshape(1, 8),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def partial_rotary_test_case():
    return _PartialRotaryTestCase()
