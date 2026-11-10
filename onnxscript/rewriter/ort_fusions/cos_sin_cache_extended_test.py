# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Boundary and numerical regressions for ORT rotary cache conversion."""

from __future__ import annotations

import unittest

import numpy
import onnx
import onnx_ir as ir
from parameterized import parameterized

from onnxscript import FLOAT, INT64, optimizer, script, values
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions._test_utils import ort_run
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.rotary_embedding import fuse_rotary_embedding

msft_op = values.Opset("com.microsoft", 1)


class CosSinCacheExtendedTest(unittest.TestCase):
    """Extended tests for cos_sin_cache fusion."""

    def _cache_model(
        self,
        *,
        batch=2,
        sequence=3,
        constant_frequencies=False,
        count=1,
        half=False,
        concat_after_trig=False,
        expand_frequency=False,
    ):
        declarations = [
            "int64[1] axis = {1}",
            "int64[2] halves = {2,2}",
        ]
        body = []
        for index in range(count):
            if constant_frequencies:
                frequency = f"frequency{index}"
                declarations.append(
                    f"float[1,2,1] {frequency} = {{{index + 1}.0,{index + 2}.0}}"
                )
            else:
                declarations.append(
                    f"float[1,2,1] inverse{index} = {{{index + 1}.0,{index + 2}.0}}"
                )
                frequency = f"frequency{index}"
                inverse = f"inverse{index}"
                if expand_frequency:
                    declarations.append(f"int64[3] expanded_shape{index} = {{{batch},2,1}}")
                    body.append(f"expanded{index}=Expand({inverse},expanded_shape{index})")
                    inverse = f"expanded{index}"
                body.append(
                    f"""
                    p{index} = Unsqueeze(position_ids,axis)
                    fp{index} = Cast<to=1>(p{index})
                    {frequency} = MatMul({inverse},fp{index})
                    """
                )
            cosine, sine = f"c{index}", f"s{index}"
            angles = f"f{index}" if concat_after_trig else f"repeated{index}"
            trig = [
                f"{cosine}=Cos({angles})",
                f"{sine}=Sin({angles})",
            ]
            if half:
                trig.extend(
                    [
                        f"half_cosine{index}=Cast<to=10>({cosine})",
                        f"half_sine{index}=Cast<to=10>({sine})",
                    ]
                )
                cosine, sine = f"half_cosine{index}", f"half_sine{index}"
            if concat_after_trig:
                trig.extend(
                    [
                        f"joined_cosine{index}=Concat<axis=-1>({cosine},{cosine})",
                        f"joined_sine{index}=Concat<axis=-1>({sine},{sine})",
                    ]
                )
                cosine, sine = f"joined_cosine{index}", f"joined_sine{index}"
            body.append(
                f"""
                f{index} = Transpose<perm=[0,2,1]>({frequency})
                repeated{index} = Concat<axis=-1>(f{index},f{index})
                {" ".join(trig)}
                cosine{index} = Unsqueeze({cosine},axis)
                sine{index} = Unsqueeze({sine},axis)
                left{index},right{index} = Split<axis=3>(x,halves)
                negative{index} = Neg(right{index})
                rotated{index} = Concat<axis=-1>(negative{index},left{index})
                real{index} = Mul(x,cosine{index})
                imaginary{index} = Mul(rotated{index},sine{index})
                output{index} = Add(real{index},imaginary{index})
                """
            )
        dtype = "float16" if half else "float"
        inputs = f"{dtype}[{batch},2,{sequence},4] x"
        if not constant_frequencies:
            inputs += f", int64[1,{sequence}] position_ids"
        outputs = ",".join(
            f"{dtype}[{batch},2,{sequence},4] output{index}" for index in range(count)
        )
        model = ir.from_onnx_text(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph ({inputs}) => ({outputs}) <{",".join(declarations)}> {{
                {"".join(body)}
            }}
            """
        )
        return ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))

    @parameterized.expand(
        [
            (half, concat, expand)
            for half in (False, True)
            for concat in (False, True)
            for expand in (False, True)
        ]
    )
    def test_cache_cast_concat_and_frequency_expansion_variants(self, half, concat, expand):
        model = self._cache_model(half=half, concat_after_trig=concat, expand_frequency=expand)
        feeds = {
            "x": numpy.random.default_rng(42)
            .normal(size=(2, 2, 3, 4))
            .astype(numpy.float16 if half else numpy.float32),
            "position_ids": numpy.array([[-3, 0, 7]], numpy.int64),
        }
        expected = ort_run("original_cache_variant", model, feeds)
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 1)
        actual = ort_run("fused_cache_variant", model, feeds)
        numpy.testing.assert_allclose(
            actual[0],
            expected[0],
            rtol=2e-3 if half else 1e-5,
            atol=2e-3 if half else 1e-5,
        )
        inferred = ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))
        rotary = next(node for node in inferred.graph if node.op_type == "RotaryEmbedding")
        self.assertEqual(rotary.domain, "com.microsoft")
        self.assertEqual(
            rotary.inputs[2].dtype, ir.DataType.FLOAT16 if half else ir.DataType.FLOAT
        )

    @parameterized.expand(
        [
            ("negative", [-7, -1, 0]),
            ("sparse", [0, 1000000000, 3]),
            ("int64_extremes", [-(2**63), 0, 2**63 - 1]),
        ]
    )
    def test_cache_uses_actual_positions(self, _, positions):
        model = self._cache_model()
        feeds = {
            "x": numpy.random.default_rng(42).normal(size=(2, 2, 3, 4)).astype(numpy.float32),
            "position_ids": numpy.array([positions], numpy.int64),
        }
        expected = ort_run("original", model, feeds)
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 1)
        self.assertLess(len(model.graph), 40)
        actual = ort_run("actual_position_cache", model, feeds)
        numpy.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)

    def test_distinct_constant_frequencies_do_not_share_cache(self):
        model = self._cache_model(constant_frequencies=True, count=2)
        feeds = {
            "x": numpy.random.default_rng(43).normal(size=(2, 2, 3, 4)).astype(numpy.float32)
        }
        expected = ort_run("original", model, feeds)
        self.assertEqual(fuse_rotary_embedding(model), 2)
        self.assertEqual(fuse_cos_sin_cache(model), 2)
        actual = ort_run("distinct_frequency_caches", model, feeds)
        for left, right in zip(expected, actual, strict=True):
            numpy.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)

    def test_overridable_inverse_frequency_is_not_constant(self):
        model = self._cache_model()
        model.graph.inputs.append(model.graph.initializers["inverse0"])
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 0)

    def test_constant_frequency_defaults_are_not_treated_as_immutable(self):
        model = self._cache_model(constant_frequencies=True)
        model.graph.inputs.append(model.graph.initializers["frequency0"])
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 0)

    def test_cache_casts_must_preserve_matching_attributes(self):
        model = self._cache_model(half=True)
        model.opset_imports[""] = 19
        self.assertEqual(fuse_rotary_embedding(model), 1)
        for node in model.graph:
            if node.op_type == "Cast" and node.inputs[0].producer().op_type == "Sin":
                node.attributes["saturate"] = ir.AttrInt64("saturate", 0)
        self.assertEqual(fuse_cos_sin_cache(model), 0)

    def test_cache_conversion_requires_bounded_shape_opset(self):
        model = self._cache_model()
        self.assertEqual(fuse_rotary_embedding(model), 1)
        model.opset_imports[""] = 14
        self.assertEqual(fuse_cos_sin_cache(model), 0)

    def test_positions_must_not_grow_rotary_data_batch(self):
        model = self._cache_model(batch=1)
        model.graph.inputs[1].shape = ir.Shape([2, 3])
        model.graph.outputs[0].shape = ir.Shape([2, 2, 3, 4])
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 0)

    @parameterized.expand([("batch", 0, 3), ("sequence", 2, 0)])
    def test_empty_rotary_cache(self, _, batch, sequence):
        model = self._cache_model(batch=batch, sequence=sequence)
        feeds = {
            "x": numpy.empty((batch, 2, sequence, 4), numpy.float32),
            "position_ids": numpy.zeros((1, sequence), numpy.int64),
        }
        expected = ort_run("original", model, feeds)
        self.assertEqual(fuse_rotary_embedding(model), 1)
        self.assertEqual(fuse_cos_sin_cache(model), 1)
        actual = ort_run("empty_cache", model, feeds)
        numpy.testing.assert_array_equal(actual[0], expected[0])

    def test_non_constant_inv_freq_no_cache_fusion(self):
        """When inv_freq is a graph input (not constant), cos_sin_cache should not fuse.

        Cache conversion currently restricts inverse-frequency parameters to
        immutable constants. Dynamic frequency parameters remain unfused.
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
