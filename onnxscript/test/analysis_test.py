import ast
import unittest

from onnxscript.main import get_ast
from onnxscript.analysis import do_liveness_analysis
from onnxscript.converter import Converter
from onnxscript.onnx_opset import opset15 as op


class AnalysisResultsVisitor(ast.NodeVisitor):
    '''
    Visitor class to flatten the results of liveness analysis in a pre-order traversal.
    '''

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


class TestAnalysis(unittest.TestCase):
    def analyze(self, fun):
        ast = get_ast(fun)
        do_liveness_analysis(ast, Converter())
        visitor = AnalysisResultsVisitor()
        visitor.visit(ast)
        return visitor.results

    def assertLiveness(self, fun, expected):
        self.assertEqual(self.analyze(fun), [set(x) for x in expected])

    def test_basic1(self):
        def basic_eg(x):
            # live = {x}
            y = x+1
            # live = {y}
            x = 1
            # live = {y}
            return y+1
        self.assertLiveness(basic_eg, [
            ["x"],
            ["y"],
            ["y"]
        ])

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
                x = x + sum*sum
                # live = {x, sum}
            # live = {x}
            return x

        self.assertLiveness(loop_eg, [
            [],
            ["sum"],
            ["x", "sum"],
            ["x", "sum", "i"],
            ["x", "sum"],
            ["x", "sum"],
            ["x"]
        ])

    def test_while_loop(self):
        def while_eg(x):
            # live = {x}
            cond = (x < 100)
            # live = {x, cond}
            while cond:
                # live = {x}
                x = x + 2
                # live = {x}
                cond = (x < 100)
                # live = {x, cond}
            # live = {x}
            return x
        self.assertLiveness(while_eg, [
            ["x"],
            ["x", "cond"],
            ["x"],
            ["x"],
            ["x", "cond"],
            ["x"]
        ])


if __name__ == '__main__':
    unittest.main(verbosity=2)
