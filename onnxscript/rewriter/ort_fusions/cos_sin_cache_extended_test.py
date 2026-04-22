# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for cos_sin_cache fusion.

Adds coverage for: non-constant inv_freq (negative — dynamic inv_freq prevents cache precomputation).
"""

from __future__ import annotations

import unittest

import numpy
import onnx_ir as ir

from onnxscript import FLOAT, INT64, optimizer, script, values
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions._test_utils import ort_run
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.rotary_embedding import fuse_rotary_embedding

msft_op = values.Opset("com.microsoft", 1)


class CosSinCacheExtendedTest(unittest.TestCase):
    """Extended tests for cos_sin_cache fusion."""

    def test_non_constant_inv_freq_no_cache_fusion(self):
        """When inv_freq is a graph input (not constant), cos_sin_cache should not fuse.

        The cos_sin_cache fusion relies on inv_freq being a constant to precompute
        the cos/sin lookup table. When inv_freq is dynamic, no fusion should apply.
        """

        @script()
        def model_with_dynamic_inv_freq(
            x: FLOAT[1, 4, 8, 8],
            position_ids: INT64[1, 8],
            inv_freq: FLOAT[1, 4, 1],
        ) -> FLOAT[1, 4, 8, 8]:
            # inv_freq is a graph input, not a constant
            position_ids_expanded = op.Unsqueeze(position_ids, [1])
            position_ids_float = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
            freqs = op.MatMul(inv_freq, position_ids_float)
            freqs_t = op.Transpose(freqs, perm=[0, 2, 1])
            emb = op.Concat(freqs_t, freqs_t, axis=-1)
            cos = op.Cos(emb)
            sin = op.Sin(emb)
            cos_4d = op.Unsqueeze(cos, [1])
            sin_4d = op.Unsqueeze(sin, [1])

            x1 = op.Slice(x, [0], [4], [3], [1])
            x2 = op.Slice(x, [4], [8], [3], [1])
            minus_x2 = op.Neg(x2)
            rotated_x = op.Concat(minus_x2, x1, axis=-1)
            result = op.Add(x * cos_4d, rotated_x * sin_4d)
            return result

        model_proto = model_with_dynamic_inv_freq.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        optimizer.optimize(model)

        # Rotary embedding fusion should still work
        re_count = fuse_rotary_embedding(model)
        self.assertGreater(re_count, 0, "RotaryEmbedding fusion should succeed.")

        # cos_sin_cache fusion should NOT work because inv_freq is not constant
        cache_count = fuse_cos_sin_cache(model)
        self.assertEqual(
            cache_count, 0, "cos_sin_cache should NOT fuse with dynamic inv_freq."
        )

    def test_constant_inv_freq_does_fuse(self):
        """Sanity check: constant inv_freq allows cos_sin_cache fusion."""

        @script()
        def model_with_const_inv_freq(
            x: FLOAT[1, 4, 8, 8], position_ids: INT64[1, 8]
        ) -> FLOAT[1, 4, 8, 8]:
            inv_freq = op.Constant(value_floats=[1.0, 2.0, 3.0, 4.0])
            inv_freq_3d = op.Unsqueeze(inv_freq, [0, 2])
            position_ids_expanded = op.Unsqueeze(position_ids, [1])
            position_ids_float = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
            freqs = op.MatMul(inv_freq_3d, position_ids_float)
            freqs_t = op.Transpose(freqs, perm=[0, 2, 1])
            emb = op.Concat(freqs_t, freqs_t, axis=-1)
            cos = op.Cos(emb)
            sin = op.Sin(emb)
            cos_4d = op.Unsqueeze(cos, [1])
            sin_4d = op.Unsqueeze(sin, [1])

            x1 = op.Slice(x, [0], [4], [3], [1])
            x2 = op.Slice(x, [4], [8], [3], [1])
            minus_x2 = op.Neg(x2)
            rotated_x = op.Concat(minus_x2, x1, axis=-1)
            result = op.Add(x * cos_4d, rotated_x * sin_4d)
            return result

        model_proto = model_with_const_inv_freq.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        optimizer.optimize(model)

        inputs = {
            "x": numpy.random.rand(1, 4, 8, 8).astype(numpy.float32),
            "position_ids": numpy.arange(8, dtype=numpy.int64).reshape(1, 8),
        }
        original_outputs = ort_run("original", model, inputs)

        re_count = fuse_rotary_embedding(model)
        self.assertGreater(re_count, 0)
        cache_count = fuse_cos_sin_cache(model)
        self.assertGreater(cache_count, 0, "cos_sin_cache should fuse with constant inv_freq.")

        # Numerical validation
        fused_outputs = ort_run("fused", model, inputs)
        numpy.testing.assert_allclose(
            original_outputs[0], fused_outputs[0], rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
