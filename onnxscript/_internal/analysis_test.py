from __future__ import annotations

import ast
import unittest
from typing import Any

from onnxscript._internal import analysis, ast_utils
from onnxscript.onnx_opset import opset15 as op
from onnxscript.sourceinfo import formatter


class AnalysisResultsVisitor(ast.NodeVisitor):
    """Visitor class to flatten the results of liveness analysis in a pre-order traversal."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[Any] = []

    def generic_visit(self, node):
        if hasattr(node, "live_in"):
            self.results.append(node.live_in)
        ast.NodeVisitor.generic_visit(self, node)
        if isinstance(node, (ast.For, ast.While)):
            last = node.body[-1]
            self.results.append(last.live_out)  # type: ignore


class TestLivenessAnalysis(unittest.TestCase):
    def analyze(self, fun):
        source, parse_tree = ast_utils.get_src_and_ast(fun)
        analysis.do_liveness_analysis(parse_tree, formatter(source))
        visitor = AnalysisResultsVisitor()
        visitor.visit(parse_tree)
        return visitor.results

    def assertLiveness(self, fun, expected):
        self.assertEqual(self.analyze(fun), [set(x) for x in expected])

    def test_basic1(self):
        def basic_eg(x):
            # live = {x}
            y = x + 1
            # live = {y}
            x = 1
            # live = {y}
            return y + 1

        self.assertLiveness(basic_eg, [["x"], ["y"], ["y"]])

    def test_doc_string(self):
        def basic_eg(x):
            # live = {x}
            """This is a docstring."""
            # live = {x}
            y = x + 1
            # live = {y}
            x = 1
            # live = {y}
            return y + 1

        self.assertLiveness(basic_eg, [["x"], ["x"], ["y"], ["y"]])

    def test_for_loop(self):
        def loop_eg():
            # live = {}
            sum = 0.0
            # live = {sum}
            x = 0.0
            # live = {x, sum}
            for i in range(10):
                # live = {x, sum, i}
                sum = sum + i
                # live = {x, sum}
                x = x + sum * sum
                # live = {x, sum}
            # live = {x}
            return x

        self.assertLiveness(
            loop_eg,
            [
                [],
                ["sum"],
                ["x", "sum"],
                ["x", "sum", "i"],
                ["x", "sum"],
                ["x", "sum"],
                ["x"],
            ],
        )

    def test_while_loop(self):
        def while_eg(x):
            # live = {x}
            cond = x < 100
            # live = {x, cond}
            while cond:
                # live = {x}
                x = x + 2
                # live = {x}
                cond = x < 100
                # live = {x, cond}
            # live = {x}
            return x

        self.assertLiveness(
            while_eg, [["x"], ["x", "cond"], ["x"], ["x"], ["x", "cond"], ["x"]]
        )


class TestExposedUses(unittest.TestCase):
    def assertUses(self, f, expected):
        source, parse_tree = ast_utils.get_src_and_ast(f)
        result = analysis.exposed_uses(parse_tree.body, formatter(source))
        self.assertEqual(result, set(expected))

    def test_basic(self):
        def f(x):
            x = x + 10
            y = 20
            z = x + y
            x = 30 + z

        self.assertUses(f, {"x"})

    def test_if(self):
        def f(x, y, z):
            if x:
                c = 10
                result = y + c
            else:
                c = 20
                result = z + c
            c30 = 30
            return result + c30

        self.assertUses(f, {"x", "y", "z"})

    def test_for_loop(self):
        def f(x, y):
            for i in range(x):
                y = y + i
            tmp = 10
            result = y + tmp
            return result

        self.assertUses(f, {"x", "y"})

    def test_while_loop(self):
        def f(x, y):
            i = 1
            while (i < 10) and (x):
                y = y + i
            tmp = y * 2
            return tmp

        self.assertUses(f, {"x", "y"})

    def test_called_function(self):
        def f(x, y):
            def nested():  # pylint: disable=unused-variable
                return y

            return op.Dummy(x, body=nested)

        self.assertUses(f, {"x", "y"})

    def test_uncalled_function(self):
        def f(x, y):
            def nested():  # pylint: disable=unused-variable
                return y

            return op.Dummy(x)

        self.assertUses(f, {"x"})

    def test_doc_string(self):
        def f(x):
            """This is a docstring."""
            x = x + 10
            y = 20
            z = x + y
            x = 30 + z

        self.assertUses(f, {"x"})


class TestAssignedVarAnalysis(unittest.TestCase):
    def assert_assigned_vars(self, f, expected: set[str]):
        source, parse_tree = ast_utils.get_src_and_ast(f)
        result = analysis.assigned_vars(parse_tree.body, formatter(source))
        self.assertEqual(result, expected)

    def test_basic_defs(self):
        def f(x):
            x = x + 1
            y = x + 2
            return y

        self.assert_assigned_vars(f, {"x", "y"})

    def test_if_defs(self):
        def f(x):
            if x > 1:
                y = x + 1
                z = 2 * y
            else:
                t = x + 2
                z = 3 * t
            return z

        self.assert_assigned_vars(f, {"z", "y", "t"})

    def test_loop_defs(self):
        def f(x):
            sum = 0
            while x > 0:
                x = x - 1
                square = x * x
                sum = sum + square
            return sum

        self.assert_assigned_vars(f, {"sum", "x", "square"})

    def test_if_loop_defs(self):
        def f(x):
            if x > 0:
                sum = 0
                while x > 0:
                    x = x - 1
                    square = x * x
                    sum = sum + square
            else:
                sum = 0
            return sum

        self.assert_assigned_vars(f, {"sum", "x", "square"})

    def test_doc_string(self):
        def f(x):
            """This is a docstring."""
            x = x + 1
            y = x + 2
            return y

        self.assert_assigned_vars(f, {"x", "y"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
