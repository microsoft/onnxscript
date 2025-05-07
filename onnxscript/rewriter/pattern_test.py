# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import contextlib
import io
import logging
import unittest

import numpy as np
import onnx.checker
import onnx.parser

import onnxscript.optimizer
from onnxscript import FLOAT, ir, script
from onnxscript import opset17 as op
from onnxscript.rewriter import cast_constant_of_shape, pattern

logger = logging.getLogger(__name__)


class ReciprocalMulTest(unittest.TestCase):
    def rule(self) -> pattern.RewriteRule:
        def reciprocal_mul_pattern(op, x, y):
            return (1 / x) * y

        def div(op, x, y):
            return op.Div(y, x)

        return pattern.RewriteRule(reciprocal_mul_pattern, div)

    def test_single_match(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                c1 = Constant<value_float = 1.0>()
                t1 = Div(c1, x)
                z1 = Mul(t1, y)
                z = Identity(z1)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = self.rule().apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 3)

    def test_failed_match(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                c1 = Constant<value_float = 0.9>()
                t1 = Div(c1, x)
                z1 = Mul(t1, y)
                z = Identity(z1)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = self.rule().apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 4)

        # Test verbose output produces something:
        # TODO(rama): Need a better way to test this.
        # Well-defined error-codes and messages would be helpful.

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.rule().apply_to_model(model, verbose=5)
        out = buffer.getvalue()
        self.assertIn("Match failed", out)

    def test_multiple_matches(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                # {c1, t1, z1} is a valid match
                # {c2, t2, z2} is a valid match
                # {c3, t3, z3} is a match, but cannot be replaced since t3 has other-uses.
                c1 = Constant<value_float = 1.0>()
                c2 = Constant<value_float = 1.0>()
                t2 = Div(c2, y)
                t1 = Div(c1, x)
                z1 = Mul(t1, y)
                z2 = Mul(t2, z1)

                c3 = Constant<value_float = 1.0>()
                t3 = Div(c3, x)
                z3 = Mul(t3, y)
                reuse_t3 = Div(t3, x)
                z = Add(z2, reuse_t3)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = self.rule().apply_to_model(model)
        self.assertEqual(count, 2)
        self.assertEqual(len(model.graph), 9)


class FastGeluTest(unittest.TestCase):
    def rule(self) -> pattern.RewriteRule:
        def fast_gelu_pattern1(op, x):
            b = 0.044715
            c = 0.79788
            tanh = op.Tanh(c * (x + (x**3) * b))
            return (1.0 + tanh) * (0.5 * x)

        def fast_gelu(op, x):
            return op.FastGelu(x, _domain="com.microsoft")

        return pattern.RewriteRule(fast_gelu_pattern1, fast_gelu)

    def long_form_rule(self) -> pattern.RewriteRule:
        def fast_gelu_pattern1_long(op, x):
            three = pattern.Constant(3)
            x_cube = op.Pow(x, three)
            b = pattern.Constant(0.044715)
            x_cube_mul_b = op.Mul(x_cube, b)  # support OR op.Mul(B, x_cube)
            sum_ = op.Add(x, x_cube_mul_b)
            c = pattern.Constant(0.79788)
            c_times_sum = op.Mul(c, sum_)
            tanh = op.Tanh(c_times_sum)
            one = pattern.Constant(1.0)
            one_plus_tanh = op.Add(one, tanh)
            half = pattern.Constant(0.5)
            half_x = op.Mul(half, x)
            return op.Mul(one_plus_tanh, half_x)

        def fast_gelu(op, x):
            return op.FastGelu(x, _domain="com.microsoft")

        return pattern.RewriteRule(fast_gelu_pattern1_long, fast_gelu)

    def _check(self, rule):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                three = Constant <value_int=3>()
                x_cube = Pow(x, three)
                B = Constant <value_float=0.044715>()
                x_cube_mul_B = Mul(x_cube, B)
                sum = Add(x, x_cube_mul_B)
                C = Constant <value_float=0.79788>()
                C_times_sum = Mul(C, sum)
                tanh = Tanh(C_times_sum)
                one = Constant <value_float=1.0> ()
                one_plus_tanh = Add(one, tanh)
                half = Constant <value_float=0.5> ()
                half_x = Mul(half, x)
                z = Mul(one_plus_tanh, half_x)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)
        # 5 Constant nodes and 1 FastGelu node
        self.assertEqual(len(model.graph), 6)

    def test_short_rule(self):
        self._check(self.rule())

    def test_long_rule(self):
        self._check(self.long_form_rule())


class ConcatTest(unittest.TestCase):
    def rule(self) -> pattern.RewriteRule:
        def concat_pattern(op, x, y, axis):
            seq = op.SequenceConstruct(x, y)
            return op.ConcatFromSequence(seq, axis=axis)

        def concat(op, x, y, axis):
            return op.Concat(x, y, axis=axis)

        return pattern.RewriteRule(concat_pattern, concat)

    def test_concat(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[M] z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = self.rule().apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

    def test_concat_in_function(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[M] y) => (float[Z] z)
            {
                z = afunction (x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x, y) => (z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = self.rule().apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[("pkg.custom", "afunction", "")]), 1)
        self.assertEqual(model.functions[("pkg.custom", "afunction", "")][0].op_type, "Concat")


class RewriteRuleTest(unittest.TestCase):
    def test_commute(self):
        def add_0(op, x):
            return x + 0

        def identity(op, x):
            return op.Identity(x)

        add_0_rule = pattern.RewriteRule(add_0, identity)

        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[M] z)
            {
                zero = Constant <value_float=0.0> ()
                z = Add (zero, x)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = pattern.RewriteRuleSet([add_0_rule], commute=True).apply_to_model(model)
        optimized_model = ir.serde.serialize_model(model)
        self.assertEqual(count, 1)
        nodes = optimized_model.graph.node
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[1].op_type, "Identity")

    def test_const_value(self):
        def reshape(op, x, newshape):
            return op.Reshape(x, newshape)

        def identity(op, x, newshape):
            del newshape  # Unused
            return op.Identity(x)

        def check_for_redundant_reshape(context, x, newshape):
            oldshape = x.shape
            newshape_const_value = newshape.const_value
            if newshape_const_value is None:
                return False

            newshape = newshape_const_value.numpy()
            newshape = newshape.tolist()

            if len(oldshape) != len(newshape):
                return False
            return all(not (d1 != d2 and d2 != -1) for d1, d2 in zip(oldshape, newshape))  # pylint: disable=consider-using-in

        rule = pattern.RewriteRule(reshape, identity, check_for_redundant_reshape)

        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[10, 20, 30] x) => (float[10, 20, 30] z)
            {
                shape = Constant <value_ints=[10, 20, 30]> ()
                z = Reshape (x, shape)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = pattern.RewriteRuleSet([rule]).apply_to_model(model)
        optimized_model = ir.serde.serialize_model(model)
        self.assertEqual(count, 1)
        nodes = optimized_model.graph.node
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[1].op_type, "Identity")

    def test_delayed_run_provides_correct_bindings_for_multiple_matches(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output, float[1, 4] output2)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 10> (constant)
                constant2 = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output2 = Cast <to = 1> (constant2)
            }
            """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 2)
        self.assertEqual(len(model.graph), 2)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 10)
        self.assertEqual(model.graph[1].attributes["value"].value.dtype, 1)

    def test_opset_import(self):
        def add_same(op, x):
            return x + x

        def double(op, x):
            return op.Double(x, _domain="custom.domain", _version=10)

        rule = pattern.RewriteRule(add_same, double)

        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[M] z)
            {
                y = Add (x, x)
                z = Relu (y)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = pattern.RewriteRuleSet([rule], commute=True).apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(model.graph.opset_imports["custom.domain"], 10)

    def test_opset_import_in_function(self):
        def add_same(op, x):
            return x + x

        def double(op, x):
            return op.Double(x, _domain="custom.domain", _version=10)

        rule = pattern.RewriteRule(add_same, double)

        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x) => (float[M] z)
            {
                z = pkg.custom.afunction (x)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x) => (z)
            {
                y = Add (x, x)
                z = Relu (y)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = pattern.RewriteRuleSet([rule], commute=True).apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(model.graph.opset_imports["custom.domain"], 10)
        self.assertEqual(
            model.functions[("pkg.custom", "afunction", "")].opset_imports["custom.domain"], 10
        )
        onnx.checker.check_model(ir.serde.serialize_model(model))

    def test_optional_attribute(self):
        """Test rules with optional attributes."""

        def concat_pattern(op, x, y):
            seq = op.SequenceConstruct(x, y)
            result = op.ConcatFromSequence(seq, _outputs=["result"])
            return result

        def concat(op, x, y, result: ir.Value):
            node = result.producer()
            assert node is not None
            axis = node.attributes.get("axis", None)
            return op.Concat(x, y, axis=axis)

        rule = pattern.RewriteRule(concat_pattern, concat)

        # Case 1: a model with attribute axis present
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[M] z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].op_type, "Concat")
        self.assertEqual(model.graph[0].attributes["axis"].value, 0)

        # Case 2: a model with attribute axis absent
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[M] z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence (t)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].op_type, "Concat")
        self.assertNotIn("axis", model.graph[0].attributes)

    def test_match_none_input(self):
        def none_pattern(op, x):
            # match against a call to Original where the first input is None
            return op.Original(None, x)

        def replacement(op, x):
            return op.Replaced(x)

        rule = pattern.RewriteRule(none_pattern, replacement)

        @script()
        def test_model(x: FLOAT[1024]) -> FLOAT[1024]:
            # Pattern should match following call
            t1 = op.Original(None, x)
            # Pattern should not match following call
            z = op.Original(t1, x)
            return z

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)

        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 2)
        self.assertEqual(model.graph.node(0).op_type, "Replaced")
        self.assertEqual(model.graph.node(1).op_type, "Original")

    def test_match_optional_input(self):
        def none_pattern(op, optional_input, x):
            # match against a call to Original where the first input may or may not be None
            return op.Original(optional_input, x)

        def replacement(op, optional_input, x):
            if optional_input is None:
                return op.ReplacedNone(x)
            return op.ReplacedNotNone(x)

        rule = pattern.RewriteRule(none_pattern, replacement)

        @script()
        def test_model(x: FLOAT[1024]) -> FLOAT[1024]:
            # Pattern should match following call
            t1 = op.Original(None, x)
            # as well as this one
            z = op.Original(t1, x)
            return z

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)

        count = rule.apply_to_model(model)
        self.assertEqual(count, 2)
        self.assertEqual(len(model.graph), 2)
        self.assertEqual(model.graph.node(0).op_type, "ReplacedNone")
        self.assertEqual(model.graph.node(1).op_type, "ReplacedNotNone")

    def test_graph_visitor(self):
        class ReplaceFoo(pattern.RewriteRuleClassBase):
            def __init__(self):
                super().__init__()
                self.replacement = None

            def pattern(self, op):
                return op.Foo()

            def rewrite(self, op):
                if self.replacement is None:
                    self.replacement = op.Bar()
                return self.replacement

        rule = ReplaceFoo.rule()

        @script()
        def test_model(x: FLOAT[1024]) -> FLOAT[1024]:
            # Pattern should match following call
            t1 = op.Foo()
            # as well as this one
            t2 = op.Foo()
            z = op.Add(t1, t2)
            return z

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)

        count = rule.apply_to_model(model)
        self.assertEqual(count, 2)
        self.assertEqual(len(model.graph), 2)
        self.assertEqual(model.graph.node(0).op_type, "Bar")
        self.assertEqual(model.graph.node(1).op_type, "Add")

    def test_debug_mode(self):
        def source_pattern(op, x):
            t1 = op.Abs(x)
            t2 = op.Neg(t1)
            t3 = op.Exp(t2)
            return t3

        def replacement(op, x):
            return op.Something(x)

        rule = pattern.RewriteRule(source_pattern, replacement)

        @script()
        def test_model(x: FLOAT[1024]) -> FLOAT[1024]:
            a2 = op.Abs(x)  # match-1 fails here
            a3 = op.Exp(a2)  # match-1 starts here
            b1 = op.Neg(a3)  # match-2 fails here
            b2 = op.Neg(b1)  # match-2 (partially) succeeds here
            b3 = op.Exp(b2)  # match-2 starts here
            return b3

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)

        tracer = pattern.MatchingTracer()
        count = rule.apply_to_model(model, tracer=tracer)
        self.assertEqual(count, 0)
        best_matches = tracer.best_matches_map[rule]
        self.assertEqual(len(best_matches), 1)
        best_match = best_matches[0]
        self.assertEqual(best_match.status.value, pattern.MatchStatus.NO_MATCH)
        self.assertIn("OpType mismatch: expected Abs, got Neg", best_match.match_result.reason)

    def test_new_initializer(self):
        def source_pattern(op, x, y):
            return op.Gemm(x, op.Transpose(y))

        def check(context, x, y):
            return y.const_value is not None

        def replacement(op, x, y):
            tensor = y.const_value
            name = y.name + "_transposed"
            transposed = ir.tensor(tensor.numpy().T, name=name)
            initializer = op.initializer(transposed)
            return op.Gemm(x, initializer)

        rule = pattern.RewriteRule(source_pattern, replacement, check)

        y_value = np.random.rand(8, 4).astype(np.float32)

        @script()
        def test_model(x: FLOAT[16, 8]) -> FLOAT[16, 4]:
            y = op.Constant(value=y_value)
            return op.Gemm(x, op.Transpose(y))

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual(len(model.graph.initializers), 1)
        last_node = model.graph[-1]
        self.assertEqual(len(last_node.inputs), 2)
        init_name = last_node.inputs[1].name
        self.assertIn(init_name, model.graph.initializers)
        self.assertIs(last_node.inputs[1], model.graph.initializers[init_name])

    def test_extract_function(self):
        def source_pattern(op, x, y, z):
            sum = op.Add(x, y)
            return op.Mul(sum, z)

        def replacement(op, x, y, z):
            return op.AddMul(x, y, z, _domain="some.domain")

        rule = pattern.RewriteRule(source_pattern, replacement, as_function=True)

        @script()
        def test_model(x: FLOAT[1024], y: FLOAT[1024], z: FLOAT[1024]) -> FLOAT[1024]:
            return op.Mul(op.Add(x, y), z)

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.graph), 1)
        call_node = model.graph.node(0)
        self.assertEqual(call_node.domain, "some.domain")
        self.assertEqual(call_node.op_type, "AddMul")
        function_id = call_node.op_identifier()
        self.assertIn(function_id, model.functions)
        function = model.functions[function_id]
        self.assertEqual([x.op_type for x in function], ["Add", "Mul"])
        onnxscript.optimizer.inline(model)
        self.assertEqual([x.op_type for x in model.graph], ["Add", "Mul"])

    def test_extract_function_with_attr(self):
        def source_pattern(op, x, y):
            sum = op.Add(x, y)
            return op.Transpose(sum, perm=[1, 0])

        def replacement(op, x, y):
            return op.AddTranspose(x, y, _domain="some.domain")

        rule = pattern.RewriteRule(source_pattern, replacement, as_function=True)

        @script()
        def test_model(x: FLOAT[1024, 512], y: FLOAT[1024, 512]) -> FLOAT[512, 1024]:
            return op.Transpose(op.Add(x, y), perm=[1, 0])

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.graph), 1)
        call_node = model.graph.node(0)
        self.assertEqual(call_node.domain, "some.domain")
        self.assertEqual(call_node.op_type, "AddTranspose")
        function_id = call_node.op_identifier()
        self.assertIn(function_id, model.functions)
        function = model.functions[function_id]
        self.assertEqual([x.op_type for x in function], ["Add", "Transpose"])
        transpose_node = function[1]
        self.assertEqual(transpose_node.attributes["perm"].value, [1, 0])
        onnxscript.optimizer.inline(model)
        self.assertEqual([x.op_type for x in model.graph], ["Add", "Transpose"])

    def test_extract_repeated_function(self):
        def source_pattern(op, x, y, z):
            sum = op.Add(x, y)
            return op.Mul(sum, z)

        def replacement(op, x, y, z):
            return op.AddMul(x, y, z, _domain="some.domain")

        rule = pattern.RewriteRule(source_pattern, replacement, as_function=True)

        @script()
        def test_model(x: FLOAT[1024], y: FLOAT[1024], z: FLOAT[1024]) -> FLOAT[1024]:
            t1 = op.Mul(op.Add(x, y), z)
            t2 = op.Mul(op.Add(t1, y), z)
            return t2

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual(len(model.functions), 2)
        self.assertEqual(len(model.graph), 2)
        for call_node in model.graph:
            self.assertEqual(call_node.domain, "some.domain")
            self.assertEqual(call_node.op_type, "AddMul")
            function_id = call_node.op_identifier()
            self.assertIn(function_id, model.functions)
        onnxscript.optimizer.inline(model)
        self.assertEqual([x.op_type for x in model.graph], ["Add", "Mul", "Add", "Mul"])

    def test_any_value(self):
        def source_pattern(op, x):
            return op.Add(x, op.Mul(0, pattern.ANY_VALUE))

        def replacement(op, x):
            return op.Identity(x)

        rule = pattern.RewriteRule(source_pattern, replacement)

        @script()
        def test_model(x: FLOAT[1024], y: FLOAT[1024]) -> FLOAT[1024]:
            zero = op.Constant(value_float=0.0)
            return op.Add(x, op.Mul(zero, y))

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual([x.op_type for x in model.graph], ["Constant", "Mul", "Add"])
        rule.apply_to_model(model)
        self.assertEqual(len(model.graph), 2)
        self.assertEqual([x.op_type for x in model.graph], ["Constant", "Identity"])

    def test_or_pattern(self):
        def source_pattern(op, x, y, bias):
            t1 = op.MatMul(x, y)
            t2 = op.Add(t1, bias)
            t1_or_t2 = pattern.OrValue([t1, t2], tag_var="has_bias", tag_values=[False, True])
            return op.Relu(t1_or_t2)

        def replacement(op, x, y, bias, has_bias):
            if has_bias:
                return op.WithBias(x, y, bias)
            else:
                return op.WithoutBias(x, y)

        rule = pattern.RewriteRule(source_pattern, replacement)

        @script()
        def test_model1(x: FLOAT[16, 32], y: FLOAT[32, 16]) -> FLOAT[16, 16]:
            return op.Relu(op.MatMul(x, y))

        model_proto = test_model1.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual([x.op_type for x in model.graph], ["WithoutBias"])

        @script()
        def test_model2(x: FLOAT[16, 32], y: FLOAT[32, 16], bias: FLOAT[16]) -> FLOAT[16, 16]:
            return op.Relu(op.Add(op.MatMul(x, y), bias))

        model_proto = test_model2.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual([x.op_type for x in model.graph], ["WithBias"])

    def test_backtracking_pattern(self):
        def source_pattern(op, x, y, bias):
            t1 = op.MatMul(x, y)
            choice1 = op.Add(t1, bias)
            choice2 = op.Add(bias, t1)
            t2 = pattern.OrValue([choice1, choice2])
            return op.Relu(t2)

        def replacement(op, x, y, bias):
            return op.GemmRelu(x, y, bias)

        rule = pattern.RewriteRule(source_pattern, replacement)

        @script()
        def test_model1(x: FLOAT[16, 32], y: FLOAT[32, 16], bias: FLOAT[16]) -> FLOAT[16, 16]:
            return op.Relu(op.Add(op.MatMul(x, y), bias))

        model_proto = test_model1.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual([x.op_type for x in model.graph], ["GemmRelu"])
        self.assertEqual([x.name for x in model.graph.node(0).inputs], ["x", "y", "bias"])

        @script()
        def test_model2(x: FLOAT[16, 32], y: FLOAT[32, 16], bias: FLOAT[16]) -> FLOAT[16, 16]:
            return op.Relu(op.Add(bias, op.MatMul(x, y)))

        model_proto = test_model2.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual([x.op_type for x in model.graph], ["GemmRelu"])
        self.assertEqual([x.name for x in model.graph.node(0).inputs], ["x", "y", "bias"])

    def test_or_pattern_return_value(self):
        """Test that an OrValue can be used as a return value from the source pattern."""

        def source_pattern(op, x, y):
            choice1 = op.Add(x, y)
            choice2 = op.Mul(x, y)
            t = pattern.OrValue([choice1, choice2])
            z = op.Relu(t)
            return z, t

        def replacement(op, x, y):
            z, t = op.ReluPlus(x, y, _outputs=2)
            return z, t

        rule = pattern.RewriteRule(source_pattern, replacement)

        @script()
        def test_model1(x: FLOAT[16, 32], y: FLOAT[16, 32]) -> FLOAT[16, 32]:
            return op.Relu(op.Add(x, y))

        model_proto = test_model1.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        self.assertEqual([x.op_type for x in model.graph], ["ReluPlus"])


class PatternBuilderTest(unittest.TestCase):
    def test_pattern_builder_context(self):
        builder = pattern.OpsetPatternBuilder("", True)
        with pattern.pattern_builder(builder):
            x = builder.Op1()
            y = builder.Op2(x)
            z = x + y
            w = builder.Op3(z)
            _ = z * w
        ops = [x.op_type for x in builder.nodes()]
        self.assertEqual(ops, ["Op1", "Op2", "Add", "Op3", "Mul"])


if __name__ == "__main__":
    unittest.main()
