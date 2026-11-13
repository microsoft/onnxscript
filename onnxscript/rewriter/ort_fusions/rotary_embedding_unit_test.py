# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for RotaryEmbeddingFusion and PartialRotaryEmbeddingFusion rules.

RotaryEmbeddingFusion matches: x * cos + rotate_half(x) * sin
and rewrites to RotaryEmbedding(x, cos, sin, ...).

PartialRotaryEmbeddingFusion matches: Concat(RotaryEmbedding(Slice(x)), Slice(x))
and sets rotary_embedding_dim and num_heads on the RotaryEmbedding op.
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

from onnxscript import FLOAT, INT64, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions import _test_utils, optimize_for_ort
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

    @parameterized.parameterized.expand(
        [
            (False, _H, None),
            (True, _H, None),
            (False, "heads", None),
            (True, "heads", None),
            (False, _H, 0),
            (False, _H, _H),
        ]
    )
    def test_partial_rotary_numerical(self, full_pipeline, heads, original_heads):
        model = self._build(
            _partial_rotary,
            input_types=[
                FLOAT["B", heads, "S", _Dh],
                FLOAT["S", _HALF // 2],
                FLOAT["S", _HALF // 2],
                INT64["B", "S"],
            ],
            output_types=[FLOAT["B", heads, "S", _Dh]],
        )
        if original_heads is not None:
            rope = next(n for n in model.graph if n.op_type == "RotaryEmbedding")
            rope.attributes["num_heads"] = ir.AttrInt64("num_heads", original_heads)
        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.standard_normal((_B, _H, _S, _Dh)).astype(np.float32),
            "cos": rng.standard_normal((_S, _HALF // 2)).astype(np.float32),
            "sin": rng.standard_normal((_S, _HALF // 2)).astype(np.float32),
            "position_ids": np.tile(np.arange(_S, dtype=np.int64), (_B, 1)),
        }
        original_outputs = _test_utils.ort_run("original", model, inputs)
        if full_pipeline:
            model, counts = optimize_for_ort(model)
            count = counts["partial_rotary_embedding"]
        else:
            count = fuse_partial_rotary_embedding(model)
        self.assertEqual(count, int(isinstance(heads, int)))
        rope = next(n for n in model.graph if n.op_type == "RotaryEmbedding")
        if count:
            self.assertEqual(rope.attributes["num_heads"].as_int(), _H)
            self.assertEqual(rope.attributes["rotary_embedding_dim"].as_int(), _HALF)
        _test_utils.assert_allclose(
            _test_utils.ort_run("optimized", model, inputs), original_outputs
        )

    @parameterized.parameterized.expand(
        [
            ("unknown_shape", None),
            ("unknown_heads", [_B, None, _S, _Dh]),
            ("zero_heads", [_B, 0, _S, _Dh]),
            ("unknown_head_size", [_B, _H, _S, "head_size"]),
            ("short_head", [_B, _H, _S, _HALF - 1]),
        ]
    )
    def test_partial_rotary_invalid_shape(self, _, shape):
        model = self._build(
            _partial_rotary,
            input_types=[
                FLOAT[_B, _H, _S, _Dh],
                FLOAT[_S, _HALF // 2],
                FLOAT[_S, _HALF // 2],
                INT64[_B, _S],
            ],
            output_types=[FLOAT[_B, _H, _S, _Dh]],
        )
        model.graph.inputs[0].shape = ir.Shape(shape) if shape is not None else None
        self.assertEqual(fuse_partial_rotary_embedding(model), 0)

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
