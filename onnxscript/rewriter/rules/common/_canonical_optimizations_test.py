# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _canonical_optimizations


class CanonicalOptimizationsTest(unittest.TestCase):
    def _apply(self, model_text: str) -> tuple[ir.Model, ir.Model, int]:
        original = ir.from_onnx_text(model_text)
        rewritten = ir.serde.deserialize_model(ir.serde.serialize_model(original))
        count = _canonical_optimizations.rules.apply_to_model(rewritten)
        return original, rewritten, count

    def test_pad_pad_fuses_full_and_axes_forms(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x) => (float[5,10] y)
            <int64[4] p1 = {1, 2, 0, 3},
             int64[2] p2 = {4, 1},
             int64[1] axes = {-1},
             float value = {2.0}>
            {
                a = Pad<mode="constant">(x, p1, value)
                y = Pad<mode="constant">(a, p2, value, axes)
            }
            """
        )
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Pad"])
        np.testing.assert_array_equal(
            rewritten.graph[0].inputs[1].const_value.numpy(), [1, 6, 0, 4]
        )
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_pad_pad_fuses_default_fill_values(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x) => (float[5,7] y)
            <int64[4] p1 = {1, 2, 0, 0}, int64[4] p2 = {0, 1, 2, 1}>
            {
                a = Pad(x, p1)
                y = Pad(a, p2)
            }
            """
        )
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Pad"])
        self.assertEqual(len(rewritten.graph[0].inputs), 2)
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_pad_pad_rejects_shared_first_pad(self):
        _, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2] x) => (float[4] y, float[3] saved)
            <int64[2] p1 = {1, 0}, int64[2] p2 = {0, 1}>
            {
                saved = Pad(x, p1)
                y = Pad(saved, p2)
            }
            """
        )
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Pad", "Pad"])
        self.assertIs(rewritten.graph.outputs[1].producer(), rewritten.graph[0])

    def test_pad_pad_rejects_negative_or_different_values(self):
        for pads, first_value, second_value in [
            ("{-1, 0}", "2.0", "2.0"),
            ("{1, 0}", "2.0", "3.0"),
        ]:
            with self.subTest(pads=pads, second_value=second_value):
                _, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (float[2] x) => (float[2] y)
                    <int64[2] p1 = {pads}, int64[2] p2 = {{0, 0}},
                     float v1 = {{{first_value}}}, float v2 = {{{second_value}}}>
                    {{
                        a = Pad(x, p1, v1)
                        y = Pad(a, p2, v2)
                    }}
                    """
                )
                self.assertEqual(count, 0)
                self.assertEqual(len(rewritten.graph), 2)

    def test_pad_pad_rejects_unsupported_modes_axes_and_overflow(self):
        cases = [
            (
                18,
                'mode="reflect"',
                "",
                "int64[2] p1 = {1, 0}, int64[2] p2 = {0, 1}",
            ),
            (
                17,
                'mode="constant"',
                ", zero, axes",
                (
                    "int64[2] p1 = {1, 0}, int64[2] p2 = {0, 1}, "
                    "int64[1] axes = {0}, float zero = {0.0}"
                ),
            ),
            (
                18,
                'mode="constant"',
                "",
                "int64[2] p1 = {9223372036854775807, 0}, int64[2] p2 = {1, 0}",
            ),
        ]
        for opset, mode, second_extra, initializers in cases:
            with self.subTest(opset=opset, mode=mode, second_extra=second_extra):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : {opset}]>
                    graph (float[2] x) => (float[] y)
                    <{initializers}>
                    {{
                        a = Pad<{mode}>(x, p1)
                        y = Pad(a, p2{second_extra})
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_pad_pad_rejects_explicit_negative_zero_against_default(self):
        for first_inputs, second_inputs in [
            ("x, p1", "a, p2, negative_zero"),
            ("x, p1, negative_zero", "a, p2"),
        ]:
            with self.subTest(first_inputs=first_inputs):
                _, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (float[2] x) => (float[4] y)
                    <int64[2] p1 = {{1, 0}}, int64[2] p2 = {{0, 1}},
                     float negative_zero = {{-0.0}}>
                    {{
                        a = Pad({first_inputs})
                        y = Pad({second_inputs})
                    }}
                    """
                )
                self.assertEqual(count, 0)
                self.assertEqual(len(rewritten.graph), 2)

    def test_all_zero_reshape_is_identity(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x) => (float[2,3] y)
            <int64[2] shape = {0, 0}>
            {
                y = Reshape(x, shape)
            }
            """
        )
        self.assertEqual(count, 1)
        self.assertEqual(rewritten.graph[0].op_type, "Identity")
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_pad_pad_distinguishes_bfloat16_signed_zero(self):
        for first_value, second_value in (
            ("", ", negative"),
            (", negative", ""),
            (", positive", ", negative"),
            (", negative", ", positive"),
        ):
            with self.subTest(first_value=first_value, second_value=second_value):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (bfloat16[2] x) => (bfloat16[4] y)
                    <int64[2] p1 = {{1, 0}}, int64[2] p2 = {{0, 1}},
                     bfloat16 negative = {{32768}}, bfloat16 positive = {{0}}>
                    {{
                        a = Pad(x, p1{first_value})
                        y = Pad(a, p2{second_value})
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_reshape_allowzero_is_not_identity(self):
        _, _, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[0,3] x) => (float[0,3] y)
            <int64[2] shape = {0, 3}>
            {
                y = Reshape<allowzero=1>(x, shape)
            }
            """
        )
        self.assertEqual(count, 0)

    def test_all_zero_reshape_rejects_rank_change(self):
        _, _, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x) => (float[2,3,1] y)
            <int64[3] shape = {0, 0, 0}>
            {
                y = Reshape(x, shape)
            }
            """
        )
        self.assertEqual(count, 0)

    def test_semantically_inspected_reference_attributes_are_rejected(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18, "local" : 1]>
            graph (float[2,3] x) => (float[2,3] y)
            {
                y = local.reshape <az = 0> (x)
            }
            <domain: "local", opset_import: ["" : 18]>
            reshape <az> (x) => (y)
            {
                shape = Constant <value = int64[2] {0, 0}> ()
                y = Reshape <allowzero: int = @az> (x, shape)
            }
            """
        )
        count = _canonical_optimizations.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        function = model.functions[("local", "reshape", "")]
        self.assertTrue(function[-1].attributes["allowzero"].is_ref())

    def test_neutral_vectors_are_removed_without_losing_broadcast_shape(self):
        cases = [
            ("int32", "Add", "x, zero", "int32[3] zero = {0, 0, 0}"),
            ("int32", "Add", "zero, x", "int32[3] zero = {0, 0, 0}"),
            ("int32", "Sub", "x, zero", "int32[3] zero = {0, 0, 0}"),
            ("float", "Mul", "x, one", "float[3] one = {1, 1, 1}"),
            ("float", "Mul", "one, x", "float[3] one = {1, 1, 1}"),
            ("float", "Div", "x, one", "float[3] one = {1, 1, 1}"),
        ]
        for dtype, op_type, inputs, initializer in cases:
            with self.subTest(op_type=op_type, inputs=inputs):
                _, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph ({dtype}[2,3] x) => ({dtype}[2,3] y)
                    <{initializer}>
                    {{
                        y = {op_type}({inputs})
                    }}
                    """
                )
                self.assertEqual(count, 1)
                self.assertEqual(rewritten.graph[0].op_type, "Identity")

    def test_neutral_vector_rejects_broadcast_and_float_signed_zero_change(self):
        cases = [
            ("float[2,3]", "float[3] zero = {0, 0, 0}", "Add(x, zero)"),
            ("int32[2,3]", "int32[1] one = {1}", "Mul(x, one)"),
            ("int32[2,3]", "int32[3] zero = {0, 0, 0}", "Sub(zero, x)"),
        ]
        for dtype, initializer, expression in cases:
            with self.subTest(expression=expression):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph ({dtype} x) => ({dtype} y)
                    <{initializer}>
                    {{
                        y = {expression}
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_integer_reduce_arg_pairs_fuse_to_topk_with_ties(self):
        cases = [
            ("Max", 1, "[2,1]", "1"),
            ("Min", 0, "[2]", "0"),
        ]
        inputs = np.array([[7, 7, 3, 5], [-2, 4, 4, 9]], dtype=np.int32)
        for suffix, largest, output_shape, keepdims in cases:
            with self.subTest(suffix=suffix):
                original, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (int32[2,4] x) =>
                        (int32{output_shape} values, int64{output_shape} indices)
                    <int64[1] axes = {{1}}>
                    {{
                        values = Reduce{suffix}<keepdims={keepdims}>(x, axes)
                        indices = Arg{suffix}<axis=1, keepdims={keepdims},
                            select_last_index=0>(x)
                    }}
                    """
                )
                self.assertEqual(count, 1)
                self.assertEqual(rewritten.graph[0].op_type, "TopK")
                self.assertEqual(rewritten.graph[0].attributes["largest"].as_int(), largest)
                testing.assert_numerically_equal(
                    original,
                    rewritten,
                    {"x": inputs},
                    rtol=0,
                    atol=0,
                    use_reference=True,
                )
                expected_indices = (
                    np.array([[0], [3]], dtype=np.int64)
                    if suffix == "Max"
                    else np.array([2, 0], dtype=np.int64)
                )
                model = ir.serde.serialize_model(rewritten)
                import onnx.reference

                actual_indices = onnx.reference.ReferenceEvaluator(model).run(
                    None, {"x": inputs}
                )[1]
                np.testing.assert_array_equal(actual_indices, expected_indices)

    def test_reduce_arg_topk_rejects_last_index_and_floats(self):
        cases = [
            (18, "int32", "select_last_index=1"),
            (18, "float", "select_last_index=0"),
            (17, "int32", "select_last_index=0"),
        ]
        for opset, dtype, select_last in cases:
            with self.subTest(opset=opset, dtype=dtype, select_last=select_last):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : {opset}]>
                    graph ({dtype}[2,4] x) =>
                        ({dtype}[2,1] values, int64[2,1] indices)
                    <int64[1] axes = {{1}}>
                    {{
                        values = ReduceMax<keepdims=1>(x, axes)
                        indices = ArgMax<axis=1, keepdims=1, {select_last}>(x)
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_reduce_arg_topk_supports_int16_and_default_axis(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (int16[3,2] x) =>
                (int16[1,2] values, int64[1,2] indices)
            <int64[1] axes = {0}>
            {
                values = ReduceMax<keepdims=1>(x, axes)
                indices = ArgMax<keepdims=1>(x)
            }
            """
        )
        self.assertEqual(count, 1)
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.array([[5, 1], [5, 7], [3, 7]], dtype=np.int16)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_reduce_arg_topk_normalizes_matching_negative_axes(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (uint32[2,4] x) =>
                (uint32[2] values, int64[2] indices)
            <int64[1] axes = {-1}>
            {
                values = ReduceMin<keepdims=0>(x, axes)
                indices = ArgMin<axis=-1, keepdims=0>(x)
            }
            """
        )
        self.assertEqual(count, 1)
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.array([[5, 1, 1, 3], [4, 9, 2, 2]], dtype=np.uint32)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_reduce_arg_topk_rejects_axis_and_attribute_mismatches(self):
        cases = [
            ("{1}", "axis=0, keepdims=1", "keepdims=1"),
            ("{2}", "axis=2, keepdims=1", "keepdims=1"),
            ("{1}", "axis=1, keepdims=0", "keepdims=1"),
            ("{1}", "axis=1, keepdims=1", "keepdims=1, noop_with_empty_axes=1"),
        ]
        for axes, arg_attributes, reduce_attributes in cases:
            with self.subTest(axes=axes, arg_attributes=arg_attributes):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (int32[2,4] x) =>
                        (int32[2,1] values, int64[2,1] indices)
                    <int64[1] axes = {axes}>
                    {{
                        values = ReduceMax<{reduce_attributes}>(x, axes)
                        indices = ArgMax<{arg_attributes}>(x)
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_reduce_arg_topk_uses_function_local_opset(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["" : 18, "local" : 1]>
            graph (int32[2,4] x) =>
                (int32[2,1] values, int64[2,1] indices)
            {
                values, indices = local.reduce_arg(x)
            }
            <domain: "local", opset_import: ["" : 17]>
            reduce_arg (x) => (values, indices)
            {
                axes = Constant<value = int64[1] {1}>()
                values = ReduceMax<keepdims=1>(x, axes)
                indices = ArgMax<axis=1, keepdims=1, select_last_index=0>(x)
            }
            """
        )
        count = _canonical_optimizations.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        function = model.functions[("local", "reduce_arg", "")]
        self.assertEqual(
            [node.op_type for node in function], ["Constant", "ReduceMax", "ArgMax"]
        )

    def test_checked_integer_range_offset_moves_into_bounds(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph () => (int32[4] y)
            <int32 start = {-3}, int32 limit = {5}, int32 delta = {2},
             int32[1] offset = {7}>
            {
                r = Range(start, limit, delta)
                y = Add(r, offset)
            }
            """
        )
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Range"])
        testing.assert_numerically_equal(
            original, rewritten, {}, rtol=0, atol=0, use_reference=True
        )

    def test_range_offset_rejects_overflow_and_shared_range(self):
        models = [
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph () => (int32[1] y)
            <int32 start = {2147483646}, int32 limit = {2147483647},
             int32 delta = {1}, int32 offset = {1}>
            {
                r = Range(start, limit, delta)
                y = Add(r, offset)
            }
            """,
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph () => (int32[3] y, int32[3] saved)
            <int32 start = {0}, int32 limit = {3}, int32 delta = {1},
             int32 offset = {2}>
            {
                saved = Range(start, limit, delta)
                y = Add(saved, offset)
            }
            """,
        ]
        for model_text in models:
            with self.subTest(model_text=model_text):
                _, _, count = self._apply(model_text)
                self.assertEqual(count, 0)

    def test_range_offset_rejects_rank_growth(self):
        _, _, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph () => (int32[1,3] y)
            <int32 start = {0}, int32 limit = {3}, int32 delta = {1},
             int32[1,1] offset = {2}>
            {
                r = Range(start, limit, delta)
                y = Add(r, offset)
            }
            """
        )
        self.assertEqual(count, 0)

    def test_range_offset_rejects_zero_delta_and_floating_range(self):
        cases = [
            ("int32", "0", "3", "0", "2"),
            ("float", "0.0", "3.0", "1.0", "2.0"),
        ]
        for dtype, start, limit, delta, offset in cases:
            with self.subTest(dtype=dtype, delta=delta):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph () => ({dtype}[3] y)
                    <{dtype} start = {{{start}}}, {dtype} limit = {{{limit}}},
                     {dtype} delta = {{{delta}}}, {dtype} offset = {{{offset}}}>
                    {{
                        r = Range(start, limit, delta)
                        y = Add(r, offset)
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_range_offset_rejects_old_opset_and_non_scalar_bounds(self):
        cases = [
            (10, "int32 start = {0}", "start"),
            (18, "int32[1] start = {0}", "start"),
        ]
        for opset, start_initializer, start_name in cases:
            with self.subTest(opset=opset, start_initializer=start_initializer):
                _, _, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : {opset}]>
                    graph () => (int32[3] y)
                    <{start_initializer}, int32 limit = {{3}},
                     int32 delta = {{1}}, int32 offset = {{2}}>
                    {{
                        r = Range({start_name}, limit, delta)
                        y = Add(r, offset)
                    }}
                    """
                )
                self.assertEqual(count, 0)

    def test_cast_moves_before_transpose_and_preserves_attributes(self):
        original, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 19]>
            graph (float[2,3] x) => (int32[3,2] y)
            {
                t = Transpose<perm=[1, 0]>(x)
                y = Cast<to=6>(t)
            }
            """
        )
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Cast", "Transpose"])
        self.assertEqual(rewritten.graph[0].attributes["to"].as_int(), 6)
        self.assertEqual(rewritten.graph[1].attributes["perm"].as_ints(), (1, 0))
        testing.assert_numerically_equal(
            original,
            rewritten,
            {"x": np.array([[1.2, -2.8, 3.0], [4.1, 5.9, -6.2]], dtype=np.float32)},
            rtol=0,
            atol=0,
            use_reference=True,
        )

    def test_cast_does_not_duplicate_shared_transpose(self):
        _, rewritten, count = self._apply(
            """
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x) => (int32[3,2] y, float[3,2] saved)
            {
                saved = Transpose<perm=[1, 0]>(x)
                y = Cast<to=6>(saved)
            }
            """
        )
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Transpose", "Cast"])

    def test_cast_does_not_duplicate_other_shared_layouts(self):
        cases = [
            (
                "int64[2] parameter = {3, 2}",
                "saved = Reshape(x, parameter)",
                "float[3,2]",
            ),
            (
                "int64[1] parameter = {0}",
                "saved = Squeeze(x, parameter)",
                "float[2,3]",
            ),
            (
                "int64[1] parameter = {0}",
                "saved = Unsqueeze(x, parameter)",
                "float[1,2,3]",
            ),
        ]
        for initializer, expression, saved_type in cases:
            with self.subTest(expression=expression):
                input_shape = "[1,2,3]" if "Squeeze" in expression else "[2,3]"
                _, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (float{input_shape} x) => (int32 y, {saved_type} saved)
                    <{initializer}>
                    {{
                        {expression}
                        y = Cast<to=6>(saved)
                    }}
                    """
                )
                self.assertEqual(count, 0)
                self.assertEqual(rewritten.graph[-1].op_type, "Cast")

    def test_cast_moves_before_other_layout_ops(self):
        cases = [
            (
                "Reshape",
                "int64[2] parameter = {3, 2}",
                "layout = Reshape<allowzero=0>(x, parameter)",
                "allowzero",
            ),
            (
                "Squeeze",
                "int64[1] parameter = {0}",
                "layout = Squeeze(x, parameter)",
                None,
            ),
            (
                "Unsqueeze",
                "int64[1] parameter = {0}",
                "layout = Unsqueeze(x, parameter)",
                None,
            ),
        ]
        for op_type, initializer, expression, attribute in cases:
            with self.subTest(op_type=op_type):
                input_shape = "[1,2,3]" if op_type == "Squeeze" else "[2,3]"
                output_shape = (
                    "[2,3]"
                    if op_type == "Squeeze"
                    else ("[1,2,3]" if op_type == "Unsqueeze" else "[3,2]")
                )
                _, rewritten, count = self._apply(
                    f"""
                    <ir_version: 9, opset_import: ["" : 18]>
                    graph (float{input_shape} x) => (int32{output_shape} y)
                    <{initializer}>
                    {{
                        {expression}
                        y = Cast<to=6>(layout)
                    }}
                    """
                )
                self.assertEqual(count, 1)
                self.assertEqual([node.op_type for node in rewritten.graph], ["Cast", op_type])
                if attribute is not None:
                    self.assertIn(attribute, rewritten.graph[1].attributes)

    def test_registered_rule_names_describe_the_transformations(self):
        self.assertEqual(
            [rule.name for rule in _canonical_optimizations.rules.rules],
            [
                "FuseConsecutiveConstantPads",
                "RemoveIdentityAllZeroReshape",
                "FuseReduceMaxArgMaxIntoTopK",
                "FuseReduceMinArgMinIntoTopK",
                "RemoveRightAddZeroVector",
                "RemoveLeftAddZeroVector",
                "RemoveRightSubZeroVector",
                "RemoveRightMulOneVector",
                "RemoveLeftMulOneVector",
                "RemoveRightDivOneVector",
                "MoveCastBeforeTranspose",
                "MoveCastBeforeReshape",
                "MoveCastBeforeSqueeze",
                "MoveCastBeforeUnsqueeze",
                "MoveIntegerRangeOffsetIntoBounds",
            ],
        )


if __name__ == "__main__":
    unittest.main()
