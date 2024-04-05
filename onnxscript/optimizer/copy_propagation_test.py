import unittest

import onnx

from onnxscript import optimizer


class RemoveUnusedTest(unittest.TestCase):
    def test_simple_identity_removal(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                t = Identity(x)
                t2 = Identity(t)
                z = Identity(t2)
            }
        """
        )
        optimizer.do_copy_propagation(model)
        self.assertEqual(len(model.graph.node), 1)

    def test_subgraph_identity_removal(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, bool cond) => (float[N] z) {
                t = Identity(x)
                t2 = Identity(t)
                t3 = If (cond) <
                    then_branch = then_graph() => (t4) {
                        t5 = Identity(t2)
                        t4 = Identity(t5)
                    },
                    else_branch = else__graph() => (t6) {
                        t7 = Identity(t)
                        t6 = Identity(t7)
                    }
                >
                z = Identity(t3)
            }
        """
        )
        optimizer.do_copy_propagation(model)
        self.assertEqual(len(model.graph.node), 2)


if __name__ == "__main__":
    unittest.main()
