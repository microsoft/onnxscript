# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for MultiHeadAttention fusion rules (mha.py).

The MHA rule matches the pattern:
    Q/K/V → Reshape → Transpose → [RotaryEmbedding] → [Concat past] → SDPA → Transpose → Reshape
and fuses it into a single MultiHeadAttention contrib op.

These are structural tests (no ORT run) because the pattern requires internal
SDPA nodes (ai.onnxruntime._fusion domain) which ORT cannot execute directly.
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx.helper
import onnx.shape_inference
import onnx_ir as ir

from onnxscript import FLOAT, INT64, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions._test_utils import ort_run
from onnxscript.rewriter.ort_fusions.mha import fuse_mha1, fuse_mha2

# Custom opsets
msft_op = values.Opset("com.microsoft", 1)
fusion_op = values.Opset("ai.onnxruntime._fusion", 1)

_B, _S, _H, _Dh = 2, 8, 4, 4
_D = _H * _Dh  # 16

_RESHAPE_Q = ir.tensor(np.array([0, 0, _H, _Dh], dtype=np.int64))
_RESHAPE_K = ir.tensor(np.array([0, 0, _H, _Dh], dtype=np.int64))
_RESHAPE_V = ir.tensor(np.array([0, 0, _H, _Dh], dtype=np.int64))
_RESHAPE_OUT = ir.tensor(np.array([0, 0, _D], dtype=np.int64))


# --- Simplest: no rotary, no past, key transposed ---


@script()
def _mha_basic_key_transposed(query_BSD, key_BSD, value_BSD):
    q_shape = op.Constant(value=_RESHAPE_Q)
    q_4d = op.Reshape(query_BSD, q_shape)
    q_BHSDh = op.Transpose(q_4d, perm=[0, 2, 1, 3])

    k_shape = op.Constant(value=_RESHAPE_K)
    k_4d = op.Reshape(key_BSD, k_shape)
    k_BHSDh = op.Transpose(k_4d, perm=[0, 2, 1, 3])

    v_shape = op.Constant(value=_RESHAPE_V)
    v_4d = op.Reshape(value_BSD, v_shape)
    v_BHSDh = op.Transpose(v_4d, perm=[0, 2, 1, 3])

    sdpa_out = fusion_op.SDPA(q_BHSDh, k_BHSDh, v_BHSDh, key_format="BHSd")

    att_transposed = op.Transpose(sdpa_out, perm=[0, 2, 1, 3])
    out_shape = op.Constant(value=_RESHAPE_OUT)
    return op.Reshape(att_transposed, out_shape)


# --- No rotary, no past, key NOT transposed ---


@script()
def _mha_basic_key_not_transposed(query_BSD, key_BSD, value_BSD):
    q_shape = op.Constant(value=_RESHAPE_Q)
    q_4d = op.Reshape(query_BSD, q_shape)
    q_BHSDh = op.Transpose(q_4d, perm=[0, 2, 1, 3])

    k_shape = op.Constant(value=_RESHAPE_K)
    k_4d = op.Reshape(key_BSD, k_shape)
    # Key is NOT transposed — stays in BSHd format

    v_shape = op.Constant(value=_RESHAPE_V)
    v_4d = op.Reshape(value_BSD, v_shape)
    v_BHSDh = op.Transpose(v_4d, perm=[0, 2, 1, 3])

    sdpa_out = fusion_op.SDPA(q_BHSDh, k_4d, v_BHSDh, key_format="BSHd")

    att_transposed = op.Transpose(sdpa_out, perm=[0, 2, 1, 3])
    out_shape = op.Constant(value=_RESHAPE_OUT)
    return op.Reshape(att_transposed, out_shape)


# --- With past key/value (has_past_present=True) ---


@script()
def _mha_with_past(query_BSD, key_BSD, value_BSD, past_key, past_value):
    q_shape = op.Constant(value=_RESHAPE_Q)
    q_4d = op.Reshape(query_BSD, q_shape)
    q_BHSDh = op.Transpose(q_4d, perm=[0, 2, 1, 3])

    k_shape = op.Constant(value=_RESHAPE_K)
    k_4d = op.Reshape(key_BSD, k_shape)
    k_BHSDh = op.Transpose(k_4d, perm=[0, 2, 1, 3])

    v_shape = op.Constant(value=_RESHAPE_V)
    v_4d = op.Reshape(value_BSD, v_shape)
    v_BHSDh = op.Transpose(v_4d, perm=[0, 2, 1, 3])

    # Concat with past
    key_seq = op.Concat(past_key, k_BHSDh, axis=-2)
    value_seq = op.Concat(past_value, v_BHSDh, axis=-2)

    sdpa_out = fusion_op.SDPA(q_BHSDh, key_seq, value_seq, key_format="BHSd")

    att_transposed = op.Transpose(sdpa_out, perm=[0, 2, 1, 3])
    out_shape = op.Constant(value=_RESHAPE_OUT)
    attention = op.Reshape(att_transposed, out_shape)
    return attention, key_seq, value_seq


# --- With rotary embedding (no past) ---


@script()
def _mha_with_rotary(query_BSD, key_BSD, value_BSD, position_ids, cos, sin):
    q_shape = op.Constant(value=_RESHAPE_Q)
    q_4d = op.Reshape(query_BSD, q_shape)
    q_BHSDh = op.Transpose(q_4d, perm=[0, 2, 1, 3])

    k_shape = op.Constant(value=_RESHAPE_K)
    k_4d = op.Reshape(key_BSD, k_shape)
    k_BHSDh = op.Transpose(k_4d, perm=[0, 2, 1, 3])

    v_shape = op.Constant(value=_RESHAPE_V)
    v_4d = op.Reshape(value_BSD, v_shape)
    v_BHSDh = op.Transpose(v_4d, perm=[0, 2, 1, 3])

    q_emb = msft_op.RotaryEmbedding(q_BHSDh, position_ids, cos, sin)
    k_emb = msft_op.RotaryEmbedding(k_BHSDh, position_ids, cos, sin)

    sdpa_out = fusion_op.SDPA(q_emb, k_emb, v_BHSDh, key_format="BHSd")

    att_transposed = op.Transpose(sdpa_out, perm=[0, 2, 1, 3])
    out_shape = op.Constant(value=_RESHAPE_OUT)
    return op.Reshape(att_transposed, out_shape)


class MultiHeadAttentionFusionTest(unittest.TestCase):
    """Structural unit tests for MultiHeadAttention fusion rules."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _apply(self, model: ir.Model) -> int:
        count = fuse_mha1(model)
        count += fuse_mha2(model)
        return count

    def _count_op(self, model: ir.Model, op_type: str, domain: str = "") -> int:
        return sum(1 for n in model.graph if n.op_type == op_type and n.domain == domain)

    def _get_mha_node(self, model: ir.Model) -> ir.Node | None:
        for node in model.graph:
            if node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft":
                return node
        return None

    _3D = (FLOAT["B", "S", _D],) * 3
    _OUT_1 = (FLOAT["B", "S", _D],)

    def test_packed_rank_two_unit_axis_layout(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 18, "ai.onnxruntime._fusion" : 1]>
            graph (float[1,16] q, float[1,16] k, float[1,16] v) => (float[1,16] y)
            <int64[4] heads = {1,4,1,4}, int64[2] packed = {1,16}> {
                q4 = Reshape(q,heads)
                k4 = Reshape(k,heads)
                v4 = Reshape(v,heads)
                attention = ai.onnxruntime._fusion.SDPA<key_format="BHSd">(q4,k4,v4)
                y = Reshape(attention,packed)
            }
            """
        )
        model = ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))
        # Custom SDPA shape inference does not populate this intermediate.
        model.graph[-2].outputs[0].shape = ir.Shape([1, 4, 1, 4])
        self.assertEqual(self._apply(model), 1)
        self.assertIsNotNone(self._get_mha_node(model))

    def test_rotary_attributes_are_not_dropped(self):
        for attribute, value in (
            ("interleaved", 2),
            ("rotary_embedding_dim", 3),
            ("rotary_embedding_dim", 6),
            ("num_heads", 2),
            ("is_packed_batching", 1),
        ):
            with self.subTest(attribute=attribute):
                model = self._build(
                    _mha_with_rotary,
                    (
                        *self._3D,
                        INT64["B", "S"],
                        FLOAT[32, _Dh // 2],
                        FLOAT[32, _Dh // 2],
                    ),
                    self._OUT_1,
                )
                rotary = next(
                    node for node in model.graph if node.op_type == "RotaryEmbedding"
                )
                rotary.attributes[attribute] = ir.AttrInt64(attribute, value)
                self.assertEqual(self._apply(model), 0)

    def test_explicit_zero_or_nonfinite_scale_is_not_fused(self):
        for scale in (0.0, -0.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                model = self._build(_mha_basic_key_transposed, self._3D, self._OUT_1)
                sdpa = next(node for node in model.graph if node.op_type == "SDPA")
                sdpa.attributes["scale"] = ir.AttrFloat32("scale", scale)
                self.assertEqual(self._apply(model), 0)

    def test_partial_rotary_mha_preserves_past_caches_numerically(self):
        for interleaved in (0, 1):
            with self.subTest(interleaved=interleaved):
                model = ir.from_onnx_text(
                    f"""
                    <ir_version: 8, opset_import: ["" : 18, "com.microsoft" : 1,
                        "ai.onnxruntime._fusion" : 1]>
                    graph (float[1,2,16] q, float[1,2,16] k, float[1,2,16] v,
                        float[1,4,3,4] past_key, float[1,4,3,4] past_value,
                        int64[1,2] positions, float[5,1] cosine, float[5,1] sine)
                        => (float[1,2,16] y, float[1,4,5,4] present_key,
                            float[1,4,5,4] present_value)
                    <int64[4] heads = {{1,2,4,4}}, int64[3] packed = {{1,2,16}}> {{
                        qr = Reshape(q,heads)
                        kr = Reshape(k,heads)
                        vr = Reshape(v,heads)
                        q4 = Transpose<perm=[0,2,1,3]>(qr)
                        k4 = Transpose<perm=[0,2,1,3]>(kr)
                        v4 = Transpose<perm=[0,2,1,3]>(vr)
                        rotary_q = com.microsoft.RotaryEmbedding<num_heads=4,
                            rotary_embedding_dim=2, interleaved={interleaved}>
                            (q4,positions,cosine,sine)
                        rotary_k = com.microsoft.RotaryEmbedding<num_heads=4,
                            rotary_embedding_dim=2, interleaved={interleaved}>
                            (k4,positions,cosine,sine)
                        present_key = Concat<axis=-2>(past_key,rotary_k)
                        present_value = Concat<axis=-2>(past_value,v4)
                        attention = ai.onnxruntime._fusion.SDPA<key_format="BHSd">
                            (rotary_q,present_key,present_value)
                        transposed = Transpose<perm=[0,2,1,3]>(attention)
                        y = Reshape(transposed,packed)
                    }}
                    """
                )
                proto = ir.to_proto(model)
                proto.functions.append(
                    onnx.helper.make_function(
                        "ai.onnxruntime._fusion",
                        "SDPA",
                        ["q", "k", "v"],
                        ["y"],
                        [
                            onnx.helper.make_node(
                                "Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]
                            ),
                            onnx.helper.make_node("MatMul", ["q", "kt"], ["scores"]),
                            onnx.helper.make_node("Constant", [], ["scale"], value_float=0.5),
                            onnx.helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
                            onnx.helper.make_node(
                                "Softmax", ["scaled"], ["probabilities"], axis=-1
                            ),
                            onnx.helper.make_node("MatMul", ["probabilities", "v"], ["y"]),
                        ],
                        [onnx.helper.make_opsetid("", 18)],
                        attributes=["key_format"],
                    )
                )
                model = ir.from_proto(onnx.shape_inference.infer_shapes(proto))
                rng = np.random.default_rng(42)
                feeds = {
                    name: rng.normal(size=shape).astype(np.float32)
                    for name, shape in (
                        ("q", (1, 2, 16)),
                        ("k", (1, 2, 16)),
                        ("v", (1, 2, 16)),
                        ("past_key", (1, 4, 3, 4)),
                        ("past_value", (1, 4, 3, 4)),
                    )
                }
                angles = np.arange(5, dtype=np.float32).reshape(5, 1)
                feeds.update(
                    positions=np.array([[3, 4]], np.int64),
                    cosine=np.cos(angles),
                    sine=np.sin(angles),
                )
                expected = ort_run("partial_rotary_with_past", model, feeds)
                self.assertEqual(self._apply(model), 1)
                self.assertEqual(len(self._get_mha_node(model).inputs), 8)
                actual = ort_run("fused_partial_rotary_with_past", model, feeds)
                for left, right in zip(expected, actual, strict=True):
                    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)
                for node in model.graph:
                    if node.op_type == "RotaryEmbedding":
                        self.assertEqual(node.attributes["num_heads"].as_int(), 4)
                        self.assertEqual(node.attributes["rotary_embedding_dim"].as_int(), 2)
                        self.assertEqual(node.attributes["interleaved"].as_int(), interleaved)

    # --- Positive tests ---

    def test_decoding_unit_layout_preserves_fusion(self):
        model = self._build(
            _mha_basic_key_transposed,
            (FLOAT[2, 1, _D],) * 3,
            (FLOAT[2, 1, _D],),
        )
        self.assertEqual(self._apply(model), 1)
        self.assertEqual(self._get_mha_node(model).attributes.get_int("num_heads"), _H)

    def test_unit_layout_requires_an_equivalent_axis_order(self):
        from onnxscript.rewriter.ort_fusions.mha import _is_unit_layout

        source = ir.Value(shape=ir.Shape([2, 3, 16]))
        incorrect = ir.Value(shape=ir.Shape([2, 4, 3, 4]))
        self.assertFalse(_is_unit_layout(source, incorrect))
        unknown = ir.Value(shape=ir.Shape([2, None, 16]))
        self.assertFalse(_is_unit_layout(unknown, incorrect))
        singleton = ir.Value(shape=ir.Shape([2, 1, 16]))
        equivalent = ir.Value(shape=ir.Shape([2, 4, 1, 4]))
        self.assertTrue(_is_unit_layout(singleton, equivalent))
        flattened = ir.Value(shape=ir.Shape([1, 16]))
        self.assertTrue(_is_unit_layout(flattened, ir.Value(shape=ir.Shape([1, 4, 1, 4]))))
        self.assertFalse(_is_unit_layout(flattened, equivalent))

    def test_basic_key_transposed(self):
        """Simplest MHA: no rotary, no past, key transposed → fuses."""
        model = self._build(_mha_basic_key_transposed, self._3D, self._OUT_1)
        count = self._apply(model)
        self.assertEqual(count, 1)
        self.assertEqual(self._count_op(model, "MultiHeadAttention", "com.microsoft"), 1)
        self.assertEqual(self._count_op(model, "SDPA", "ai.onnxruntime._fusion"), 0)
        mha = self._get_mha_node(model)
        self.assertIsNotNone(mha)
        self.assertEqual(mha.attributes.get_int("num_heads", 0), _H)

    def test_basic_key_not_transposed(self):
        """Key not transposed (BSHd format) → still fuses."""
        model = self._build(_mha_basic_key_not_transposed, self._3D, self._OUT_1)
        count = self._apply(model)
        self.assertEqual(count, 1)
        self.assertEqual(self._count_op(model, "MultiHeadAttention", "com.microsoft"), 1)

    def test_with_past_key_value(self):
        """Past key/value Concats → fuses with 3 outputs (attention, present_k, present_v)."""
        model = self._build(
            _mha_with_past,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", _H, "Spast", _Dh],
                FLOAT["B", _H, "Spast", _Dh],
            ],
            output_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", _H, "St", _Dh],
                FLOAT["B", _H, "St", _Dh],
            ],
        )
        count = self._apply(model)
        self.assertEqual(count, 1)
        mha = self._get_mha_node(model)
        self.assertIsNotNone(mha)
        self.assertEqual(len(mha.outputs), 3)
        # past_key and past_value should be connected (inputs 6, 7)
        self.assertIsNotNone(mha.inputs[6])
        self.assertIsNotNone(mha.inputs[7])

    def test_with_rotary_embedding(self):
        """RotaryEmbedding on Q and K before SDPA → fuses."""
        model = self._build(
            _mha_with_rotary,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S"],  # position_ids
                FLOAT["S", _Dh],  # cos
                FLOAT["S", _Dh],  # sin
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 1)
        mha = self._get_mha_node(model)
        self.assertIsNotNone(mha)
        # Rotary should be moved to operate on BSD-format inputs in the rewrite
        rotary_count = self._count_op(model, "RotaryEmbedding", "com.microsoft")
        self.assertGreater(rotary_count, 0)

    # --- Negative test ---

    def test_rank2_query_no_fusion(self):
        """Query with rank 2 [S, D] instead of [B, S, D] → shape check rejects."""
        model = self._build(
            _mha_basic_key_transposed,
            input_types=[
                FLOAT["S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)

    def test_wrong_query_transpose_perm_no_fusion(self):
        """Corrupted query Transpose perm → pattern should not match."""
        model = self._build(_mha_basic_key_transposed, self._3D, self._OUT_1)
        for node in model.graph:
            if node.op_type == "Transpose":
                perm = node.attributes.get_ints("perm")
                if perm == (0, 2, 1, 3):
                    node.attributes["perm"] = ir.AttrInt64s("perm", [0, 1, 2, 3])
                    break
        count = self._apply(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
