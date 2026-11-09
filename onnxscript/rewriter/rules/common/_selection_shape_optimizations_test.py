# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.reference
import onnx_ir as ir
import parameterized

from onnxscript import optimizer, rewriter
from onnxscript.rewriter.rules.common import (
    _boolean_optimizations as boolean,
)
from onnxscript.rewriter.rules.common import (
    _label_encoder,
)
from onnxscript.rewriter.rules.common import (
    _shape_optimizations as shape,
)


class SelectionShapeOptimizationsTest(unittest.TestCase):
    def _check(self, text, feeds, rules, count=1):
        model = ir.from_onnx_text(text)
        original = ir.serde.serialize_model(model)
        actual = rules.apply_to_model(model)
        self.assertEqual(actual, count)
        rewritten = ir.serde.serialize_model(model)
        onnx.checker.check_model(rewritten)
        expected = onnx.reference.ReferenceEvaluator(original).run(None, feeds)
        outputs = onnx.reference.ReferenceEvaluator(rewritten).run(None, feeds)
        for before, after in zip(expected, outputs):
            np.testing.assert_array_equal(before, after)
            if before.dtype.kind == "f":
                self.assertEqual(before.tobytes(), after.tobytes())
        return model

    @parameterized.parameterized.expand([(False,), (True,)])
    def test_not_not_shared_output(self, shared):
        outputs = "bool[N] y, bool[N] intermediate" if shared else "bool[N] y"
        model = self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[N] x) => ({outputs}) {{
                intermediate = Not(x)
                y = Not(intermediate)
            }}
            """,
            {"x": np.array([True, False])},
            ir_rule_set(boolean.not_not_rule),
        )
        self.assertEqual(sum(n.op_type == "Not" for n in model.graph), int(shared))

    def test_not_where_broadcast_and_special_values(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[2,1] c, float[1,3] x, float[2,3] y)
                => (float[2,3] z, bool[2,1] negated) {
                negated = Not(c)
                z = Where(negated, x, y)
            }
            """,
            {
                "c": np.array([[True], [False]]),
                "x": np.array([[np.nan, -0.0, np.inf]], np.float32),
                "y": np.array([[0.0, -0.0, -np.inf], [1, 2, 3]], np.float32),
            },
            ir_rule_set(boolean.not_where_rule),
        )

    @parameterized.parameterized.expand([("And", "1"), ("Or", "0")])
    def test_boolean_identities(self, op_type, neutral):
        for order in ("x, c", "c, x"):
            for shape_spec, literal in (
                ("", neutral),
                ("[1]", neutral),
                ("[2]", f"{neutral},{neutral}"),
            ):
                with self.subTest(order=order, shape=shape_spec):
                    self._check(
                        f"""
                        <ir_version: 8, opset_import: ["" : 18]>
                        graph (bool[2] x) => (bool[2] y)
                        <bool{shape_spec} c = {{{literal}}}> {{
                            y = {op_type}({order})
                        }}
                        """,
                        {"x": np.array([True, False])},
                        boolean.rules,
                    )

    def test_boolean_identity_does_not_remove_broadcast(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[1] x) => (bool[2] y)
            <bool[2] c = {1,1}> { y = And(x,c) }
            """,
            {"x": np.array([False])},
            boolean.rules,
            count=0,
        )

    def test_boolean_identity_preserves_overridable_initializer(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[2] x, bool[2] c) => (bool[2] y)
            <bool[2] c = {1,1}> { y = And(x,c) }
            """,
            {"x": np.array([True, True]), "c": np.array([False, False])},
            boolean.rules,
            count=0,
        )

    @parameterized.parameterized.expand([(False,), (True,)])
    def test_where_add_preserves_broadcast_and_operand_order(self, common_first):
        lhs, rhs = ("common, x", "common, y") if common_first else ("x, common", "y, common")
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[2,1] c, float[2,1] common, float[3] x, float[1,3] y)
                => (float[2,3] z) {{
                a = Add({lhs})
                b = Add({rhs})
                z = Where(c,a,b)
            }}
            """,
            {
                "c": np.array([[True], [False]]),
                "common": np.array([[np.inf], [-0.0]], np.float32),
                "x": np.array([-np.inf, np.nan, -0.0], np.float32),
                "y": np.array([[0.0, -0.0, np.inf]], np.float32),
            },
            boolean.rules,
        )

    def test_where_add_rejects_shared_arithmetic(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[N] c, float[N] common, float[N] x, float[N] y)
                => (float[N] z, float[N] a) {
                a = Add(common,x)
                b = Add(common,y)
                z = Where(c,a,b)
            }
            """,
            {
                "c": np.array([True]),
                "common": np.array([1.0], np.float32),
                "x": np.array([2.0], np.float32),
                "y": np.array([3.0], np.float32),
            },
            boolean.rules,
            count=0,
        )

    def test_where_add_preserves_subgraph_captures(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[N] c, bool decision, float[N] common, float[N] x, float[N] y)
                => (float[N] z, float[N] captured) {
                a = Add(common,x)
                b = Add(common,y)
                z = Where(c,a,b)
                captured = If<
                    then_branch = then_graph () => (float[N] t) { t = Identity(a) },
                    else_branch = else_graph () => (float[N] f) { f = Identity(b) }
                >(decision)
            }
            """,
            {
                "c": np.array([True, False]),
                "decision": np.array(True),
                "common": np.array([1.0, -0.0], np.float32),
                "x": np.array([2.0, 0.0], np.float32),
                "y": np.array([3.0, -0.0], np.float32),
            },
            boolean.rules,
            count=0,
        )

    def test_boolean_identities_preserve_empty_broadcasts(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[0,3] x) => (bool[0,3] y)
            <bool[1,3] c = {1,1,1}> { y = And(x,c) }
            """,
            {"x": np.empty((0, 3), np.bool_)},
            boolean.rules,
        )

    def test_unsqueeze_equal(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N] x, float[N] y) => (bool[N,1] z)
            <int64[1] a = {-1}, int64[1] b = {1}> {
                u = Unsqueeze(x,a)
                v = Unsqueeze(y,b)
                z = Equal(u,v)
            }
            """,
            {
                "x": np.array([0, np.nan], np.float32),
                "y": np.array([-0.0, np.nan], np.float32),
            },
            ir_rule_set(boolean.unsqueeze_equal_rule),
        )

    def test_reuse_unsqueeze(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N] x) => (float[1,N] u, bool[1,N] z)
            <int64[1] axes = {0}, float c = {0.0}> {
                u = Unsqueeze(x,axes)
                e = Equal(x,c)
                z = Unsqueeze(e,axes)
            }
            """,
            {"x": np.array([0, 1], np.float32)},
            ir_rule_set(boolean.reuse_unsqueezed_comparison_rule),
        )

    def test_reuse_unsqueeze_does_not_create_forward_reference(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N] x) => (bool[1,N] z, float[1,N] u)
            <int64[1] axes = {0}, float c = {0.0}> {
                e = Equal(x,c)
                z = Unsqueeze(e,axes)
                u = Unsqueeze(x,axes)
            }
            """,
            {"x": np.array([0, 1], np.float32)},
            ir_rule_set(boolean.reuse_unsqueezed_comparison_rule),
            count=0,
        )

    def test_unsqueeze_equal_rejects_different_input_ranks(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N] x, float[1,N] y) => (bool[1,N,N] z)
            <int64[1] a = {-1}, int64[1] b = {-1}> {
                u = Unsqueeze(x,a)
                v = Unsqueeze(y,b)
                z = Equal(u,v)
            }
            """,
            {"x": np.array([0, 1], np.float32), "y": np.array([[0, 1]], np.float32)},
            ir_rule_set(boolean.unsqueeze_equal_rule),
            count=0,
        )

    def test_not_not_rejects_custom_domain(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 18, "custom" : 1]>
            graph (bool[N] x) => (bool[N] y) {
                t = custom.Not(x)
                y = Not(t)
            }
            """
        )
        self.assertEqual(boolean.rules.apply_to_model(model), 0)

    @parameterized.parameterized.expand(
        [("",), ("start=1",), ("start=-3,end=-1",), ("start=8,end=10",)]
    )
    def test_shape_cast_preserves_bounds_and_shared_cast(self, attributes):
        attributes = f"<{attributes}>" if attributes else ""
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B,C] x) => (int64[K] y, double[A,B,C] converted) {{
                converted = Cast<to=11>(x)
                y = Shape{attributes}(converted)
            }}
            """,
            {"x": np.ones((2, 0, 3), np.float32)},
            ir_rule_set(shape.shape_cast_rule),
        )

    @parameterized.parameterized.expand(
        [
            ("perm=[2,0,1]", ""),
            ("", "start=-2,end=9"),
            ("perm=[2,0,1]", "start=2,end=1"),
            ("perm=[2,0,1]", "start=-99,end=-1"),
        ]
    )
    def test_shape_transpose(self, permutation, bounds):
        permutation = f"<{permutation}>" if permutation else ""
        bounds = f"<{bounds}>" if bounds else ""
        model = self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B,C] x) => (int64[K] y) {{
                t = Transpose{permutation}(x)
                y = Shape{bounds}(t)
            }}
            """,
            {"x": np.ones((2, 0, 3), np.float32)},
            ir_rule_set(shape.shape_transpose_rule),
        )
        self.assertNotIn("Transpose", [n.op_type for n in model.graph])

    @parameterized.parameterized.expand(
        [
            ("[2]", "0,1", "int64[2]", "", 18, 1),
            ("", "-1", "int64", "start=1,end=-1", 18, 1),
            ("[0]", "", "int64[0]", "", 18, 1),
            ("[2]", "0,2", "int64[2]", "", 18, 0),
            ("[1]", "0", "int64[1]", "", 14, 0),
        ]
    )
    def test_gather_shape(self, index_shape, values, output_type, bounds, opset, count):
        bounds = f"<{bounds}>" if bounds else ""
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : {opset}]>
            graph (float[A,B,C,D] x) => ({output_type} y)
            <int64{index_shape} indices = {{{values}}}> {{
                s = Shape{bounds}(x)
                y = Gather<axis=-1>(s,indices)
            }}
            """,
            {"x": np.ones((2, 0, 3, 4), np.float32)},
            ir_rule_set(shape.gather_shape_rule),
            count=count,
        )

    @parameterized.parameterized.expand(
        [
            ("0,-1", ""),
            ("1,3", "start=-4,end=-1"),
            ("0,1", "start=0,end=2"),
            ("1,2", "start=8"),
        ]
    )
    def test_shape_unsqueeze(self, axes, bounds):
        bounds = f"<{bounds}>" if bounds else ""
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B] x) => (int64[K] y)
            <int64[2] axes = {{{axes}}}> {{
                u = Unsqueeze(x,axes)
                y = Shape{bounds}(u)
            }}
            """,
            {"x": np.ones((0, 3), np.float32)},
            ir_rule_set(shape.shape_unsqueeze_rule),
        )

    def test_shape_unsqueeze_requires_constant_axes(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B] x, int64[1] axes) => (int64[3] y) {
                u = Unsqueeze(x,axes)
                y = Shape(u)
            }
            """,
            {"x": np.ones((0, 3), np.float32), "axes": np.array([-1], np.int64)},
            ir_rule_set(shape.shape_unsqueeze_rule),
            count=0,
        )

    def test_gather_shape_requires_constant_indices(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B] x, int64[1] indices) => (int64[1] y) {
                s = Shape(x)
                y = Gather(s,indices)
            }
            """,
            {"x": np.ones((0, 3), np.float32), "indices": np.array([-1], np.int64)},
            ir_rule_set(shape.gather_shape_rule),
            count=0,
        )

    def test_gather_shape_rejects_out_of_range_indices(self):
        for index in (-3, 2):
            with self.subTest(index=index):
                model = ir.from_onnx_text(
                    f"""
                    <ir_version: 8, opset_import: ["" : 18]>
                    graph (float[A,B] x) => (int64 y)
                    <int64 index = {{{index}}}> {{
                        s = Shape(x)
                        y = Gather(s,index)
                    }}
                    """
                )
                self.assertEqual(ir_rule_set(shape.gather_shape_rule).apply_to_model(model), 0)

    def test_gather_shape_preserves_int32_scalar_and_scalar_input_shape(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B] x) => (int64 y)
            <int32 index = {-1}> { s = Shape(x) y = Gather(s,index) }
            """,
            {"x": np.empty((2, 0), np.float32)},
            ir_rule_set(shape.gather_shape_rule),
        )
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float x) => (int64[0] y) {
                transposed = Transpose(x)
                y = Shape(transposed)
            }
            """,
            {"x": np.array(-0.0, np.float32)},
            ir_rule_set(shape.shape_transpose_rule),
        )

    def test_shape_attribute_references_are_not_guessed(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18, "local" : 1]>
            graph (float[2,3] x) => (int64[1] y) {
                y = local.query<begin=1>(x)
            }
            <domain: "local", opset_import: ["" : 18]>
            query<begin>(x) => (y) {
                c = Cast<to=11>(x)
                y = Shape<start: int = @begin>(c)
            }
            """
        )
        self.assertEqual(shape.rules.apply_to_model(model), 0)

    def test_shape_transpose_preserves_shared_data(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[A,B] x) => (float[B,A] t, int64[2] y) {
                t = Transpose(x)
                y = Shape(t)
            }
            """,
            {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
            ir_rule_set(shape.shape_transpose_rule),
        )

    @parameterized.parameterized.expand([("Expand",), ("Reshape",)])
    def test_equal_shape_children(self, op_type):
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N,M] x, int64[2] first, int64[2] second)
                => (float[M,N] a, float[M,N] b) {{
                a = {op_type}(x,first)
                b = {op_type}(x,second)
            }}
            """,
            {
                "x": np.arange(4, dtype=np.float32).reshape(2, 2),
                "first": np.array([2, 2], np.int64),
                "second": np.array([2, 2], np.int64),
            },
            ir_rule_set(
                shape.same_shape_expand_rule
                if op_type == "Expand"
                else shape.same_shape_reshape_rule
            ),
        )

    def test_equal_shape_children_rejects_distinct_symbols(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N,M] x, int64[2] first, int64[2] second)
                => (float[M,N] a, float[K,N] b) {
                a = Reshape(x,first)
                b = Reshape(x,second)
            }
            """,
            {
                "x": np.arange(4, dtype=np.float32).reshape(2, 2),
                "first": np.array([2, 2], np.int64),
                "second": np.array([2, 2], np.int64),
            },
            ir_rule_set(shape.same_shape_reshape_rule),
            count=0,
        )

    @parameterized.parameterized.expand([("Expand",), ("Reshape",)])
    def test_same_shape_reuse_rejects_anonymous_dimensions(self, op_type):
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[1] x, int64[2] first, int64[2] second)
                => (float[?,?] a, float[?,?] b) {{
                a = {op_type}(x,first)
                b = {op_type}(x,second)
            }}
            """,
            {
                "x": np.ones(1, np.float32),
                "first": np.array([1, 1], np.int64),
                "second": np.array([1, 1], np.int64),
            },
            ir_rule_set(
                shape.same_shape_expand_rule
                if op_type == "Expand"
                else shape.same_shape_reshape_rule
            ),
            count=0,
        )

    @parameterized.parameterized.expand([(False,), (True,), (False, 0)])
    def test_simplify_expand_shape(self, shared, rows=2):
        outputs = "float[N,3] y, int64[2] original" if shared else "float[N,3] y"
        model = self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N,1] x, int64[1] first, int64[1] second)
                => ({outputs}) {{
                original = Concat<axis=0>(first,second)
                y = Expand(x,original)
            }}
            """,
            {
                "x": np.ones((rows, 1), np.float32),
                "first": np.array([rows], np.int64),
                "second": np.array([3], np.int64),
            },
            ir_rule_set(shape.simplify_expand_shape_rule),
        )
        self.assertEqual(
            sum(node.op_type == "Concat" for node in model.graph), 1 + int(shared)
        )
        self.assertEqual(
            ir_rule_set(shape.simplify_expand_shape_rule).apply_to_model(model), 0
        )

    @parameterized.parameterized.expand([("?,1", "?,3"), ("1,1", "2,3"), ("1", "2,3")])
    def test_simplify_expand_shape_requires_dimension_proof(self, input_shape, output_shape):
        self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[{input_shape}] x, int64[1] first, int64[1] second)
                => (float[{output_shape}] y) {{
                target = Concat<axis=0>(first,second)
                y = Expand(x,target)
            }}
            """,
            {
                "x": np.ones((2, 1) if input_shape == "?,1" else (1, 1), np.float32)
                if "," in input_shape
                else np.ones(1, np.float32),
                "first": np.array([2], np.int64),
                "second": np.array([3], np.int64),
            },
            ir_rule_set(shape.simplify_expand_shape_rule),
            count=0,
        )

    def test_label_encoder_composes_defaults(self):
        model = self._check(
            """
            <ir_version: 8, opset_import: ["" : 18, "ai.onnx.ml" : 4]>
            graph (string[N] x) => (string[N] y, int64[N] intermediate) {
                intermediate = ai.onnx.ml.LabelEncoder<
                    keys_strings=["a","b"], values_int64s=[1,2], default_int64=3
                >(x)
                y = ai.onnx.ml.LabelEncoder<
                    keys_int64s=[1,3], values_strings=["one","default"]
                >(intermediate)
            }
            """,
            {"x": np.array(["a", "b", "missing"])},
            _label_encoder.rules,
        )
        self.assertEqual(sum(n.op_type == "LabelEncoder" for n in model.graph), 2)

    def test_label_encoder_rejects_float_mappings(self):
        self._check(
            """
            <ir_version: 8, opset_import: ["" : 18, "ai.onnx.ml" : 4]>
            graph (float[N] x) => (int64[N] y) {
                t = ai.onnx.ml.LabelEncoder<keys_floats=[1.0], values_int64s=[2]>(x)
                y = ai.onnx.ml.LabelEncoder<keys_int64s=[2], values_int64s=[3]>(t)
            }
            """,
            {"x": np.array([1.0, np.nan], np.float32)},
            _label_encoder.rules,
            count=0,
        )

    def test_label_encoder_preserves_typed_empty_mapping(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 18, "ai.onnx.ml" : 4]>
            graph (string[N] x) => (int64[N] y) {
                t = ai.onnx.ml.LabelEncoder<
                    keys_strings=["a"], values_int64s=[1], default_int64=7
                >(x)
                y = ai.onnx.ml.LabelEncoder<
                    keys_int64s=[7], values_int64s=[9], default_int64=3
                >(t)
            }
            """
        )
        model.graph[0].attributes["keys_strings"] = ir.AttrStrings("keys_strings", [])
        model.graph[0].attributes["values_int64s"] = ir.AttrInt64s("values_int64s", [])
        self.assertEqual(_label_encoder.rules.apply_to_model(model), 1)
        rewritten = ir.to_proto(model)
        onnx.checker.check_model(rewritten)
        self.assertEqual(
            model.graph[-1].attributes["keys_strings"].type, ir.AttributeType.STRINGS
        )
        self.assertEqual(
            model.graph[-1].attributes["values_int64s"].type, ir.AttributeType.INTS
        )
        self.assertEqual(model.graph[-1].attributes["default_int64"].as_int(), 9)

    @parameterized.parameterized.expand(
        [
            (2, "int64", "int64s", "[1,2]", "[3,4]", "[3]", "[7]", [1, 2, 9]),
            (
                4,
                "string",
                "strings",
                '["a","b"]',
                '["c","d"]',
                '["c"]',
                '["e"]',
                ["a", "b", "z"],
            ),
        ]
    )
    def test_label_encoder_unshared_and_implicit_defaults(
        self, opset, dtype, suffix, keys, values, outer_keys, outer_values, inputs
    ):
        model = self._check(
            f"""
            <ir_version: 8, opset_import: ["" : 18, "ai.onnx.ml" : {opset}]>
            graph ({dtype}[N] x) => ({dtype}[N] y) {{
                t = ai.onnx.ml.LabelEncoder<keys_{suffix}={keys},values_{suffix}={values}>(x)
                y = ai.onnx.ml.LabelEncoder<
                    keys_{suffix}={outer_keys},values_{suffix}={outer_values}
                >(t)
            }}
            """,
            {"x": np.array(inputs, np.int64 if dtype == "int64" else str)},
            _label_encoder.rules,
        )
        self.assertEqual(sum(n.op_type == "LabelEncoder" for n in model.graph), 1)

    @parameterized.parameterized.expand([("rewrite",), ("optimize",)])
    def test_default_surfaces_include_new_rules(self, surface):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            graph (bool[N] condition, float[A,B] x)
                => (bool[N] y, int64[2] s) {
                n = Not(condition)
                y = Not(n)
                c = Cast<to=11>(x)
                s = Shape(c)
            }
            """
        )
        original = ir.serde.serialize_model(model)
        if surface == "rewrite":
            rewriter.rewrite(model)
        else:
            optimizer.optimize(model)
        rewritten = ir.serde.serialize_model(model)
        onnx.checker.check_model(rewritten)
        self.assertNotIn("Not", [n.op_type for n in model.graph])
        self.assertNotIn("Cast", [n.op_type for n in model.graph])
        feeds = {
            "condition": np.array([True, False]),
            "x": np.ones((0, 3), np.float32),
        }
        before = onnx.reference.ReferenceEvaluator(original).run(None, feeds)
        after = onnx.reference.ReferenceEvaluator(rewritten).run(None, feeds)
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(expected, actual)


def ir_rule_set(rule):
    from onnxscript.rewriter import RewriteRuleSet

    return RewriteRuleSet([rule])


if __name__ == "__main__":
    unittest.main()
