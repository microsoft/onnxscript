import ast
import unittest

from onnxscript.main import get_ast
from onnxscript.analysis import do_liveness_analysis
from onnxscript.converter import Converter
from onnxscript.onnx_opset import opset15 as op

class AnalysisResultsVisitor(ast.NodeVisitor):
    '''
    Visitor used to return the results of liveness analysis as a flattened list
    in pre-order traversal.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.results = []

    def generic_visit(self, node):
        if hasattr(node, "live_in"):
            self.results.append(node.live_in)
        ast.NodeVisitor.generic_visit(self, node)

class TestAnalysis(unittest.TestCase):
    def analyze(self, fun):
        ast = get_ast(fun)
        do_liveness_analysis(ast, Converter())
        visitor = AnalysisResultsVisitor()
        visitor.visit(ast)
        return visitor.results

    def test_loop(self):
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
            # live = {x}
            return x
        results = self.analyze(loop_eg)
        # See annotations in example above for expected results at each point:
        self.assertEqual(results, [
            set([]),
            set(["sum"]),
            set(["x", "sum"]),
            set(["x", "sum", "i"]),
            set(["x", "sum"]),
            set(["x"])
            ])


if __name__ == '__main__':
    unittest.main(verbosity=2)