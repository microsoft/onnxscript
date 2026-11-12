# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import ast
import importlib.util
import os
import tempfile
import unittest
import unittest.mock

import onnx.parser
import onnx_ir as ir

from onnxscript.rewriter import pattern, pattern_codegen

_HAS_LIBCST = importlib.util.find_spec("libcst") is not None


def _load_model(text: str) -> ir.Model:
    """Parse a model and name its nodes after their op types (Add, Add_1, ...)."""
    model = ir.from_proto(onnx.parser.parse_model(text))
    counts: dict[str, int] = {}
    for node in model.graph:
        n = counts.get(node.op_type, 0)
        counts[node.op_type] = n + 1
        node.name = node.op_type if n == 0 else f"{node.op_type}_{n}"
    return model


_MODEL = """
    <ir_version: 8, opset_import: ["" : 18]>
    agraph (float[N] x, float[N] y) => (float[N] z)
    {
        a = Add(x, y)
        b = LeakyRelu<alpha=0.1>(a)
        c, d = Split<axis=-1, num_outputs=2>(b)
        z = Concat<axis=0>(c, d)
    }
"""


def _names(values) -> list[str | None]:
    return [v.name for v in values]


class SelectSubgraphTest(unittest.TestCase):
    def test_inputs_and_outputs_are_the_selection_boundary(self):
        model = _load_model(_MODEL)
        view = pattern_codegen.select_subgraph(model, ["LeakyRelu", "Split"])
        self.assertEqual([n.name for n in view], ["LeakyRelu", "Split"])
        self.assertEqual(_names(view.inputs), ["a"])
        self.assertEqual(_names(view.outputs), ["c", "d"])

    def test_intermediate_values_are_not_outputs(self):
        model = _load_model(_MODEL)
        view = pattern_codegen.select_subgraph(model, ["Add", "LeakyRelu"])
        self.assertEqual(_names(view.inputs), ["x", "y"])
        self.assertEqual(_names(view.outputs), ["b"])

    def test_value_used_inside_and_outside_selection_is_an_output(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                b = Neg(a)
                z = Add(a, b)
            }
            """
        )
        view = pattern_codegen.select_subgraph(model, ["Relu", "Neg"])
        self.assertEqual(_names(view.outputs), ["a", "b"])

    def test_graph_output_is_an_output(self):
        model = _load_model(_MODEL)
        view = pattern_codegen.select_subgraph(model, ["Concat"])
        self.assertEqual(_names(view.outputs), ["z"])

    def test_selection_order_follows_the_graph(self):
        model = _load_model(_MODEL)
        view = pattern_codegen.select_subgraph(model, ["LeakyRelu", "Add"])
        self.assertEqual([n.name for n in view], ["Add", "LeakyRelu"])

    def test_unknown_name_suggests_close_matches(self):
        model = _load_model(_MODEL)
        with self.assertRaisesRegex(ValueError, r"'Spilt'.*did you mean.*Split"):
            pattern_codegen.select_subgraph(model, ["Spilt"])

    def test_empty_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            pattern_codegen.select_subgraph(_load_model(_MODEL), [])

    def test_duplicate_node_names_are_rejected(self):
        model = _load_model(_MODEL)
        for node in model.graph:
            node.name = "same"
        with self.assertRaisesRegex(ValueError, "not unique"):
            pattern_codegen.select_subgraph(model, ["same"])

    def test_non_convex_selection_is_rejected(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                b = Neg(a)
                z = Add(a, b)
            }
            """
        )
        with self.assertRaisesRegex(ValueError, "not convex"):
            pattern_codegen.select_subgraph(model, ["Relu", "Add"])


@unittest.skipUnless(_HAS_LIBCST, "libcst is not installed")
class GenerateTest(unittest.TestCase):
    def test_generated_source(self):
        model = _load_model(_MODEL)
        source = pattern_codegen.generate_rewrite_rule(model, ["LeakyRelu", "Split"])
        expected = '''\
from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter import pattern


class LeakyReluSplitRule(pattern.RewriteRuleClassBase):
    """Rewrite rule generated from nodes: LeakyRelu, Split"""

    def pattern(self, op, a):
        b = op.LeakyRelu(a, alpha=0.10000000149011612)
        c, d = op.Split(b, axis=-1, num_outputs=2, _outputs=2)
        return c, d

    def rewrite(self, op, a: ir.Value):
        raise NotImplementedError("Write the replacement here.")
'''
        self.assertEqual(source, expected)

    def test_class_name(self):
        model = _load_model(_MODEL)
        source = pattern_codegen.generate_rewrite_rule(model, ["Add"], class_name="MyRule")
        self.assertIn("class MyRule(", source)
        with self.assertRaisesRegex(ValueError, "Invalid class name"):
            pattern_codegen.generate_rewrite_rule(model, ["Add"], class_name="not valid")

    def test_long_selection_gets_a_generic_class_name(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                b = Relu(a)
                c = Relu(b)
                d = Relu(c)
                z = Relu(d)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(
            model, ["Relu", "Relu_1", "Relu_2", "Relu_3", "Relu_4"]
        )
        self.assertIn("class GeneratedRule(", source)

    def test_identifier_sanitization(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                z = Neg(a)
            }
            """
        )
        relu, neg = list(model.graph)
        relu.inputs[0].name = "/model/layer.0/input"
        relu.outputs[0].name = "class"
        neg.outputs[0].name = None
        source = pattern_codegen.generate_rewrite_rule(model, ["Relu", "Neg"])
        ast.parse(source)
        self.assertIn("def pattern(self, op, model_layer_0_input):", source)
        self.assertIn("class_ = op.Relu(model_layer_0_input)", source)
        self.assertIn("v1 = op.Neg(class_)", source)

    def test_value_names_that_collide_with_reserved_names_or_each_other(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                z = Neg(a)
            }
            """
        )
        relu, neg = list(model.graph)
        relu.inputs[0].name = "op"
        relu.outputs[0].name = "op_1"
        neg.outputs[0].name = "op"
        source = pattern_codegen.generate_rewrite_rule(model, ["Relu", "Neg"])
        self.assertIn("def pattern(self, op, op_1):", source)
        self.assertIn("op_1_1 = op.Relu(op_1)", source)
        self.assertIn("op_2 = op.Neg(op_1_1)", source)

    def test_value_used_twice_is_one_pattern_input(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Mul(x, x)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Mul"])
        self.assertIn("def pattern(self, op, x):", source)
        self.assertIn("z = op.Mul(x, x)", source)

    def test_optional_inputs(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x, float[N] lo) => (float[N] z, float[N] w)
            {
                z = Clip(x, lo)
                w = Clip(x, , lo)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Clip", "Clip_1"])
        self.assertIn("z = op.Clip(x, lo)", source)
        self.assertIn("w = op.Clip(x, None, lo)", source)

    def test_trailing_absent_optional_input_is_dropped(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Clip(x, , )
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Clip"])
        self.assertIn("z = op.Clip(x)", source)

    def test_custom_domain(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18, "com.microsoft" : 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = com.microsoft.Gelu(x)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Gelu"])
        self.assertIn('z = op.Gelu(x, _domain="com.microsoft")', source)

    def test_attribute_kinds(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Foo<i=-3, f=0.5, s="a\\"b", ints=[1, 2], floats=[0.5], strings=["u", "v"]>(x)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Foo"])
        self.assertIn(
            'op.Foo(x, i=-3, f=0.5, s="a\\"b", ints=[1, 2], floats=[0.5], strings=["u", "v"])',
            source,
        )

    def test_unsupported_attribute_is_skipped_with_a_comment(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph () => (float[2] z)
            {
                z = Constant<value = float[2] {1.0, 2.0}>()
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Constant"])
        self.assertIn("# Attribute 'value' of Constant (Constant) is not matched", source)
        self.assertIn("z = op.Constant()", source)
        ast.parse(source)

    def test_requires_libcst_message(self):
        # The message is exercised by mocking the import; see _import_libcst.
        with unittest.mock.patch.dict("sys.modules", {"libcst": None}):
            self.assertRaisesRegex(
                ImportError,
                "pip install libcst",
                pattern_codegen._import_libcst,  # pylint: disable=protected-access
            )

    def _apply(self, model: ir.Model, source: str, replacement: str) -> None:
        """Execute the generated rule, with its rewrite stub replaced, on the model."""
        stub = 'raise NotImplementedError("Write the replacement here.")'
        self.assertIn(stub, source)
        namespace: dict = {}
        exec(compile(source.replace(stub, replacement), "<generated>", "exec"), namespace)
        rule_classes = [
            v
            for v in namespace.values()
            if isinstance(v, type)
            and issubclass(v, pattern.RewriteRuleClassBase)
            and v is not pattern.RewriteRuleClassBase
        ]
        self.assertEqual(len(rule_classes), 1)
        count = rule_classes[0].rule().apply_to_model(model)
        self.assertEqual(count, 1)

    def test_generated_rule_matches_and_rewrites_the_source_model(self):
        model = _load_model(_MODEL)
        source = pattern_codegen.generate_rewrite_rule(model, ["LeakyRelu", "Split"])
        self._apply(
            model, source, "return op.Split(op.Relu(a), axis=-1, num_outputs=2, _outputs=2)"
        )
        self.assertEqual([n.op_type for n in model.graph], ["Add", "Relu", "Split", "Concat"])

    def test_generated_rule_with_multiple_boundary_values(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            {
                a = Relu(x)
                b = Neg(a)
                z = Add(a, b)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Relu", "Neg"])
        self.assertIn("return a, b", source)
        self._apply(model, source, "return op.Identity(x), op.Neg(x)")
        self.assertEqual([n.op_type for n in model.graph], ["Identity", "Neg", "Add"])

    def test_initializer_inputs_become_pattern_inputs(self):
        model = _load_model(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            agraph (float[N] x) => (float[N] z)
            <float[1] w = {2.0}>
            {
                z = Mul(x, w)
            }
            """
        )
        source = pattern_codegen.generate_rewrite_rule(model, ["Mul"])
        self.assertIn("def pattern(self, op, x, w):", source)
        self._apply(model, source, "return op.Add(x, w)")
        self.assertEqual([n.op_type for n in model.graph], ["Add"])


@unittest.skipUnless(_HAS_LIBCST, "libcst is not installed")
class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = os.path.join(self.tmp.name, "model.onnx")
        ir.save(_load_model(_MODEL), self.model_path)

    def test_writes_output_file(self):
        out = os.path.join(self.tmp.name, "rule.py")
        code = pattern_codegen.main(
            [
                self.model_path,
                "--nodes",
                "Add",
                "LeakyRelu",
                "--class-name",
                "AddLeaky",
                "-o",
                out,
            ]
        )
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            source = f.read()
        self.assertIn("class AddLeaky(pattern.RewriteRuleClassBase):", source)

    def test_unknown_node_is_an_error(self):
        code = pattern_codegen.main([self.model_path, "--nodes", "Nope"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
