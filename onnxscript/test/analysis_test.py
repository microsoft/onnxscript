import ast
import unittest

from onnxscript.analysis import do_liveness_analysis, exposed_uses
from onnxscript.converter import Converter
from onnxscript.main import get_ast


class AnalysisResultsVisitor(ast.NodeVisitor):
    """
    Visitor class to flatten the results of liveness analysis in a pre-order traversal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results = []

    def generic_visit(self, node):
        if hasattr(node, "live_in"):
            self.results.append(node.live_in)
        ast.NodeVisitor.generic_visit(self, node)
        if isinstance(node, (ast.For, ast.While)):
            last = node.body[-1]
            self.results.append(last.live_out)


class TestLivenessAnalysis(unittest.TestCase):
    def analyze(self, fun):
        ast = get_ast(fun) # pylint: disable=redefined-outer-name
        do_liveness_analysis(ast, Converter())
        visitor = AnalysisResultsVisitor()
        visitor.visit(ast)
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
        ast = get_ast(f) # pylint: disable=redefined-outer-name
        result = exposed_uses(ast.body, Converter())
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
