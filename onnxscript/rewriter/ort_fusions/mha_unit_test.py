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
import onnx_ir as ir

from onnxscript import FLOAT, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions.mha import fuse_mha1, fuse_mha2

# Custom opsets
msft_op = values.Opset("com.microsoft", 1)
fusion_op = values.Opset("ai.onnxruntime._fusion", 1)

_B, _S, _H, _Dh = 2, 8, 4, 4
_D = _H * _Dh  # 16
_Skv = 8
_Spast = 4

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

    # --- Positive tests ---

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


if __name__ == "__main__":
    unittest.main()
