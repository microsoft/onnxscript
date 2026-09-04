# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for RotaryEmbeddingFusion and PartialRotaryEmbeddingFusion rules.

RotaryEmbeddingFusion matches: x * cos + rotate_half(x) * sin
and rewrites to RotaryEmbedding(x, cos, sin, ...).

PartialRotaryEmbeddingFusion matches: Concat(RotaryEmbedding(Slice(x)), Slice(x))
and adds rotary_embedding_dim attribute to the RotaryEmbedding op.
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript import FLOAT, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions.rotary_embedding import (
    fuse_partial_rotary_embedding,
    fuse_rotary_embedding,
)

fusion_op = values.Opset("ai.onnxruntime._fusion", 1)
msft_op = values.Opset("com.microsoft", 1)

_B, _H, _S, _Dh = 2, 4, 8, 8
_HALF = _Dh // 2

# Constants for slice boundaries
_ZERO = ir.tensor(np.array([0], dtype=np.int64))
_HALF_TENSOR = ir.tensor(np.array([_HALF], dtype=np.int64))
_HEAD_SIZE_TENSOR = ir.tensor(np.array([_Dh], dtype=np.int64))
_MAX_INT = ir.tensor(np.array([9223372036854775807], dtype=np.int64))


# --- Full rotary embedding pattern ---


@script()
def _rotary_full(x, cos, sin):
    """X * cos + rotate_half(x) * sin — standard non-interleaved pattern."""
    start_0 = op.Constant(value=_ZERO)
    end_half = op.Constant(value=_HALF_TENSOR)
    start_half = op.Constant(value=_HALF_TENSOR)
    end_full = op.Constant(value=_HEAD_SIZE_TENSOR)
    # rotate_half: concat(-x2, x1)
    x1 = op.Slice(x, start_0, end_half, [3], [1])
    x2 = op.Slice(x, start_half, end_full, [3], [1])
    neg_x2 = op.Neg(x2)
    rotated = op.Concat(neg_x2, x1, axis=-1)
    return op.Add(op.Mul(x, cos), op.Mul(rotated, sin))


# --- Partial rotary embedding pattern ---


@script()
def _partial_rotary(x, cos, sin, position_ids):
    """Slice → RotaryEmbedding on first half, concat with second half."""
    end1 = op.Constant(value=_HALF_TENSOR)
    start2 = op.Constant(value=_HALF_TENSOR)
    max_end = op.Constant(value=_MAX_INT)
    x_part1 = op.Slice(x, [0], end1, [3], [1])
    x_part2 = op.Slice(x, start2, max_end, [3], [1])
    x_part1_rope = msft_op.RotaryEmbedding(x_part1, position_ids, cos, sin, interleaved=0)
    return op.Concat(x_part1_rope, x_part2, axis=-1)


# --- Negative: 3D input instead of 4D ---


@script()
def _rotary_3d_input(x, cos, sin):
    """3D input — should fail the 4D check."""
    start_0 = op.Constant(value=_ZERO)
    end_half = op.Constant(value=_HALF_TENSOR)
    start_half = op.Constant(value=_HALF_TENSOR)
    end_full = op.Constant(value=_HEAD_SIZE_TENSOR)
    x1 = op.Slice(x, start_0, end_half, [3], [1])
    x2 = op.Slice(x, start_half, end_full, [3], [1])
    neg_x2 = op.Neg(x2)
    rotated = op.Concat(neg_x2, x1, axis=-1)
    return op.Add(op.Mul(x, cos), op.Mul(rotated, sin))


class RotaryEmbeddingFusionTest(unittest.TestCase):
    """Unit tests for RotaryEmbeddingFusion rule."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _count_op(self, model: ir.Model, op_type: str, domain: str = "") -> int:
        return sum(1 for n in model.graph if n.op_type == op_type and n.domain == domain)

    # --- Positive tests ---

    def test_full_rotary_fuses(self):
        """Standard full rotary embedding pattern → fuses to RotaryEmbedding op."""
        model = self._build(
            _rotary_full,
            input_types=[
                FLOAT[_B, _H, "S", _Dh],
                FLOAT[_B, _H, "S", _Dh],
                FLOAT[_B, _H, "S", _Dh],
            ],
            output_types=[FLOAT[_B, _H, "S", _Dh]],
        )
        count = fuse_rotary_embedding(model)
        self.assertEqual(count, 1)
        self.assertEqual(self._count_op(model, "RotaryEmbedding", "ai.onnxruntime._fusion"), 1)
        # Pattern ops should be consumed
        self.assertEqual(self._count_op(model, "Neg"), 0)

    def test_partial_rotary_fuses(self):
        """Partial rotary: Slice → RotaryEmbedding on first part, concat with rest."""
        model = self._build(
            _partial_rotary,
            input_types=[
                FLOAT[_B, _H, "S", _Dh],
                FLOAT["S", _HALF],
                FLOAT["S", _HALF],
                FLOAT[_B, "S"],
            ],
            output_types=[FLOAT[_B, _H, "S", _Dh]],
        )
        count = fuse_partial_rotary_embedding(model)
        self.assertEqual(count, 1)
        # RotaryEmbedding should still exist but now with rotary_embedding_dim
        rope_nodes = [
            n
            for n in model.graph
            if n.op_type == "RotaryEmbedding" and n.domain == "com.microsoft"
        ]
        self.assertEqual(len(rope_nodes), 1)
        self.assertIn("rotary_embedding_dim", rope_nodes[0].attributes)
        self.assertEqual(rope_nodes[0].attributes["rotary_embedding_dim"].value, _HALF)

    # --- Negative tests ---

    def test_3d_input_no_fusion(self):
        """3D input (missing batch or head dim) → check rejects."""
        model_proto = _rotary_3d_input.to_model_proto(
            input_types=[
                FLOAT[_H, "S", _Dh],
                FLOAT[_H, "S", _Dh],
                FLOAT[_H, "S", _Dh],
            ],
            output_types=[FLOAT[_H, "S", _Dh]],
        )
        model = ir.serde.deserialize_model(model_proto)
        # Skip optimize — 3D shapes cause shape inference errors which is expected
        count = fuse_rotary_embedding(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
