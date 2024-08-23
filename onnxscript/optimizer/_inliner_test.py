from __future__ import annotations
import unittest

from onnx import parser
from onnxscript import ir
from onnxscript.optimizer._inliner import inline

class InlinerTest(unittest.TestCase):
    def test_basic(self):
        model_proto = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """
        )
        model_ir = ir.serde.deserialize_model(model_proto)
        inline(model_ir)

        graph = model_ir.graph

        # function-call should be replaced by Add, followed by Mul
        self.assertEqual(len(graph), 2)
        self.assertEqual(graph.node(0).op_type, "Add")
        self.assertEqual(graph.node(1).op_type, "Mul")

if __name__ == "__main__":
    unittest.main()