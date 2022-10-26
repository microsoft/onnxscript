import unittest

import numpy as np

from onnxscript import eager_mode_evaluator, graph, script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT


class EvaluatorTest(unittest.TestCase):
    def test_mixed_evaluator(self):
        @script()
        def seq_map(x: FLOAT["N"]):  # noqa: F821
            seq1 = op.SequenceConstruct(x, x + 1, x + 2)

            @graph()
            def square(y: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
                return op.Mul(y, y)

            seq2 = op.SequenceMap(seq1, body=square)
            return seq2

        x = np.array([0.0, 1.0], dtype=np.float32)
        output = seq_map[eager_mode_evaluator.ort_mixed_evaluator](x)
        expected = [t * t for t in [x, x + 1, x + 2]]
        np.testing.assert_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
