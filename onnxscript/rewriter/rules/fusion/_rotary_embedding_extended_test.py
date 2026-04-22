# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for RotaryEmbedding and PartialRotaryEmbedding fusion rules.

Adds coverage for: PartialRotaryEmbedding23 positive test, boundary mismatch
negative, and "rotary_embedding_dim already set" negative.
"""

from __future__ import annotations

import unittest

import numpy
import onnx
import onnx_ir as ir
from packaging.version import Version

import onnxscript
import onnxscript.rewriter.testing
from onnxscript.rewriter.models import _rotary_embedding_models
from onnxscript.rewriter.rules.fusion import _rotary_embedding

_ROTARY_DIM = 32

_INPUT_TYPES = (
    onnxscript.INT64["Batchsize", "Sequence"],
    onnxscript.FLOAT["Batchsize", 32, "Sequence", 80],
)
_OUTPUT_TYPES = (onnxscript.FLOAT["Batchsize", 32, "Sequence", 80],)


class PartialRotaryEmbeddingExtendedTest(unittest.TestCase):
    """Extended tests for PartialRotaryEmbedding23 fusion rule."""

    def _get_partial_model(self, *, mismatched: bool = False) -> ir.Model:
        """Get a fresh partial rotary embedding model."""
        script_fn = _rotary_embedding_models._make_partial_rotary_script(mismatched=mismatched)
        model_proto = script_fn.to_model_proto(
            input_types=_INPUT_TYPES, output_types=_OUTPUT_TYPES
        )
        return ir.serde.deserialize_model(model_proto)

    def test_partial_rotary_embedding_fused(self):
        """Full rotary embedding + partial concat → fuse into RotaryEmbedding with rotary_embedding_dim."""
        model = self._get_partial_model()
        model.graph.opset_imports[""] = 23

        original_proto = ir.serde.serialize_model(model)

        onnxscript.optimizer.optimize(model)
        # First fuse the base rotary embedding
        count = _rotary_embedding.fuse_rotary_embedding(model)
        self.assertGreater(count, 0, "Base RotaryEmbedding fusion should succeed first.")

        # Then fuse partial rotary embedding
        count_partial = _rotary_embedding.fuse_partial_rotary_embedding(model)
        self.assertGreater(count_partial, 0, "PartialRotaryEmbedding fusion should succeed.")

        # Verify RotaryEmbedding has rotary_embedding_dim attribute
        rope_nodes = [n for n in model.graph if n.op_type == "RotaryEmbedding"]
        self.assertTrue(len(rope_nodes) > 0, "Should have RotaryEmbedding node.")
        rope_node = rope_nodes[0]
        self.assertIn("rotary_embedding_dim", rope_node.attributes)
        self.assertEqual(rope_node.attributes["rotary_embedding_dim"].value, _ROTARY_DIM)

        # Numerical validation via reference implementation (if onnx version supports it)
        rewritten_proto = ir.serde.serialize_model(model)
        onnx_ver = Version(onnx.__version__)
        if onnx_ver >= Version("1.19.1") and not (
            onnx_ver.is_devrelease or onnx_ver.is_prerelease
        ):
            inputs = {
                "query": numpy.random.rand(1, 32, 8, 80).astype(numpy.float32),
                "position_ids": numpy.arange(8, dtype=numpy.int64).reshape(1, 8),
            }
            onnxscript.rewriter.testing.assert_numerically_equal(
                original_proto, rewritten_proto, args=inputs, use_reference=True
            )

    def test_partial_rotary_mismatched_boundaries_no_fusion(self):
        """When end1 != start2 in partial slice, PartialRotaryEmbedding should NOT fuse."""
        model = self._get_partial_model(mismatched=True)
        model.graph.opset_imports[""] = 23

        onnxscript.optimizer.optimize(model)
        # First fuse the base rotary embedding
        count = _rotary_embedding.fuse_rotary_embedding(model)
        self.assertGreater(count, 0)

        # Partial fusion should fail because end1=32 but start2=33
        count_partial = _rotary_embedding.fuse_partial_rotary_embedding(model)
        self.assertEqual(count_partial, 0, "Should NOT fuse with mismatched boundaries.")

    def test_partial_rotary_already_has_dim_attr_no_fusion(self):
        """If RotaryEmbedding already has rotary_embedding_dim, partial fusion should NOT apply."""
        model = self._get_partial_model()
        model.graph.opset_imports[""] = 23

        onnxscript.optimizer.optimize(model)
        # Fuse base rotary embedding
        count = _rotary_embedding.fuse_rotary_embedding(model)
        self.assertGreater(count, 0)

        # Add rotary_embedding_dim attribute to the RotaryEmbedding node
        for node in model.graph:
            if node.op_type == "RotaryEmbedding":
                node.attributes["rotary_embedding_dim"] = ir.AttrInt64(
                    "rotary_embedding_dim", 16
                )
                break

        # Partial fusion should refuse to fuse
        count_partial = _rotary_embedding.fuse_partial_rotary_embedding(model)
        self.assertEqual(
            count_partial, 0, "Should NOT fuse when rotary_embedding_dim already set."
        )


if __name__ == "__main__":
    unittest.main()
