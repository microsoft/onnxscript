# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for @script functions with control flow called via op.call()."""

import unittest
from typing import Tuple

import onnx
import onnx_ir as ir

import onnxscript
from onnxscript import script
from onnxscript._internal.builder import OpBuilder, TypeSpec, _resolve_type_spec
from onnxscript.onnx_types import BOOL, FLOAT, INT64


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_graph_and_builder(
    inputs: list[Tuple[str, TypeSpec]],
    opset_version: int = 15,
) -> Tuple[OpBuilder, ...]:
    """Create a graph with the given inputs and return (op, *input_values)."""
    graph = ir.Graph(
        name="test",
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": opset_version},
    )
    for name, type_spec in inputs:
        ts = _resolve_type_spec(type_spec)
        graph.inputs.append(ir.Value(name=name, type=ts.type, shape=ts.shape))
    gb = onnxscript.GraphBuilder(graph)
    return (gb.op, *graph.inputs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ScriptControlFlowViaCallTest(unittest.TestCase):
    """Test that @script functions with Loop / If can be inlined via op.call()."""

    def test_if_then_else(self):
        """Call a script function containing an If node."""
        op, A, B = _make_graph_and_builder([("A", FLOAT[4]), ("B", FLOAT[4])])
        graph = op.builder.graph

        @script(default_opset=op)
        def _maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            """If-then-else: return the input whose element sum is larger."""
            sum1 = op.ReduceSum(A)
            sum2 = op.ReduceSum(B)
            if sum1 < sum2:
                result = op.Identity(B)
            else:
                result = op.Identity(A)
            return result

        result = op.call(_maxsum, A, B)
        graph.outputs.append(result)

        op_types = [n.op_type for n in graph]
        self.assertIn("If", op_types)

        # The If node should have then_branch and else_branch graph attrs
        if_node = next(n for n in graph if n.op_type == "If")
        self.assertIn("then_branch", if_node.attributes)
        self.assertIn("else_branch", if_node.attributes)
        then_graph = if_node.attributes["then_branch"].value
        else_graph = if_node.attributes["else_branch"].value
        self.assertIsInstance(then_graph, ir.Graph)
        self.assertIsInstance(else_graph, ir.Graph)

    def test_for_loop(self):
        """Call a script function containing a Loop node."""
        op, x, N = _make_graph_and_builder([("x", FLOAT[4]), ("N", INT64)])
        graph = op.builder.graph

        @script(default_opset=op)
        def _sumprod(x, N):
            """For-loop: accumulate sum and product over N iterations."""
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod

        result = op.call(_sumprod, x, N)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        graph.outputs.extend(result)

        op_types = [n.op_type for n in graph]
        self.assertIn("Loop", op_types)

        # The Loop node should have a body graph attr
        loop_node = next(n for n in graph if n.op_type == "Loop")
        self.assertIn("body", loop_node.attributes)
        body = loop_node.attributes["body"].value
        self.assertIsInstance(body, ir.Graph)

    def test_loop_body_references_outer_scope_value(self):
        """A loop body where a function parameter is used inside the loop.

        In onnxscript's @script model, 'alpha' is a formal parameter. When
        compiled, it becomes an outer-scope reference in the Loop body -- the
        body's Mul node references 'alpha' directly without it appearing in
        the Loop's formal inputs or loop-carried dependencies.

        After inlining via op.call(), this outer-scope reference should be
        correctly wired to the caller's value.
        """
        op, x, N, alpha = _make_graph_and_builder([("x", FLOAT[4]), ("N", INT64), ("alpha", FLOAT)])
        graph = op.builder.graph

        @script(default_opset=op)
        def _loop_with_alpha(x, N, alpha):
            """Loop whose body references a function parameter (outer-scope ref)."""
            result = op.Identity(x)
            for _ in range(N):
                result = result * alpha
            return result

        result = op.call(_loop_with_alpha, x, N, alpha)
        graph.outputs.append(result)

        op_types = [n.op_type for n in graph]
        self.assertIn("Loop", op_types)

        loop_node = next(n for n in graph if n.op_type == "Loop")
        body = loop_node.attributes["body"].value
        self.assertIsInstance(body, ir.Graph)

        # alpha is an outer-scope reference: it should appear as an input to the
        # Mul node inside the body, but NOT as a body input.
        body_input_names = {v.name for v in body.inputs}
        self.assertNotIn("alpha", body_input_names)

        # The Mul node in the body should reference alpha
        mul_nodes = [n for n in body if n.op_type == "Mul"]
        self.assertTrue(len(mul_nodes) > 0)
        mul_input_names = [v.name if v else None for v in mul_nodes[0].inputs]
        self.assertIn("alpha", mul_input_names)

    def test_if_referencing_outer_value(self):
        """An if-then-else where the branches reference a value defined before the if.

        In onnxscript @script, values defined before an if/else are accessible
        inside the branches (outer scope reference). After inlining, these should
        be properly wired.
        """
        op, X, Y, flag = _make_graph_and_builder([("X", FLOAT[4]), ("Y", FLOAT[4]), ("flag", BOOL)])
        graph = op.builder.graph

        @script(default_opset=op)
        def _conditional_add_or_mul(X, Y, flag):
            """If-then-else whose branches reference a value defined before the if."""
            Z = X + Y
            if flag:
                result = Z + X
            else:
                result = Z * X
            return result

        result = op.call(_conditional_add_or_mul, X, Y, flag)
        graph.outputs.append(result)

        op_types = [n.op_type for n in graph]
        self.assertIn("Add", op_types)  # Z = X + Y
        self.assertIn("If", op_types)

        if_node = next(n for n in graph if n.op_type == "If")
        then_graph = if_node.attributes["then_branch"].value
        else_graph = if_node.attributes["else_branch"].value

        # The then_branch should contain an Add node that references Z (outer scope)
        then_ops = [n.op_type for n in then_graph]
        self.assertIn("Add", then_ops)

        # The else_branch should contain a Mul node that references Z (outer scope)
        else_ops = [n.op_type for n in else_graph]
        self.assertIn("Mul", else_ops)

        # Z (from the outer Add) should be referenced inside the branches
        add_node = next(n for n in graph if n.op_type == "Add")
        z_value = add_node.outputs[0]
        then_add = next(n for n in then_graph if n.op_type == "Add")
        self.assertIn(z_value, list(then_add.inputs))

    def test_model_validation_with_if(self):
        """Build a complete model with an inlined if-then-else and validate it."""
        op, A, B = _make_graph_and_builder([("A", FLOAT[4]), ("B", FLOAT[4])])
        graph = op.builder.graph

        @script(default_opset=op)
        def _maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            """If-then-else: return the input whose element sum is larger."""
            sum1 = op.ReduceSum(A)
            sum2 = op.ReduceSum(B)
            if sum1 < sum2:
                result = op.Identity(B)
            else:
                result = op.Identity(A)
            return result

        result = op.call(_maxsum, A, B)
        graph.outputs.append(result)

        model = ir.Model(graph=graph, ir_version=8)
        proto = ir.serde.serialize_model(model)

        # If-then-else models should pass validation
        onnx.checker.check_model(proto)

    def test_model_validation_with_loop_missing_types(self):
        """Loop model from untyped @script function fails validation due to missing types.

        The _sumprod function is defined without type annotations, so the loop
        body's inputs/outputs lack type information. The ONNX checker requires
        these to be present.
        """
        op, x, N = _make_graph_and_builder([("x", FLOAT[4]), ("N", INT64)])
        graph = op.builder.graph

        @script(default_opset=op)
        def _sumprod(x, N):
            """For-loop: accumulate sum and product over N iterations."""
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod

        result = op.call(_sumprod, x, N)
        if isinstance(result, list):
            graph.outputs.extend(result)
        else:
            graph.outputs.append(result)

        model = ir.Model(graph=graph, ir_version=8)
        proto = ir.serde.serialize_model(model)

        # This fails because the loop body inputs/outputs lack type info
        # (the @script function was defined without type annotations)
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(proto)


if __name__ == "__main__":
    unittest.main()
