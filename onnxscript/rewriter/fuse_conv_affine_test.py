import unittest

import numpy as np

from onnxscript import ir
from onnxscript.rewriter import rewrite, testing
from onnxscript.rewriter.fuse_conv_affine import (
    affine_conv_fusion_rule,
    conv_affine_fusion_rule,
)


class FuseConvAffineTest(unittest.TestCase):
    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def test_conv_affine_fusion(self):
        tape = ir.tape.Tape()
        x = ir.Input(
            "x", shape=ir.Shape([1, 3, 32, 32]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        scale = tape.initializer(ir.tensor(np.array([2.0], dtype=np.float32), name="scale"))
        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        conv_out = tape.op("Conv", [x, w, b], attributes={"pads": [1, 1, 1, 1]})
        mul_out = tape.op("Mul", [conv_out, scale])
        z = tape.op(
            "Add",
            [mul_out, offset],
            output=ir.Input(
                "z",
                shape=ir.Shape([1, 3, 32, 32]),
                type=ir.TensorType(ir.DataType.FLOAT),
            ),
        )

        model = ir.Model(
            ir.Graph(
                inputs=[x],
                outputs=[z],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 17},
            ),
            ir_version=8,
        )
        rewritten_model = self.clone_model(model)
        rewritten_model = rewrite(
            rewritten_model,
            pattern_rewrite_rules=[conv_affine_fusion_rule],
        )
        # Check that Mul and Add are fused into Conv
        self.assertEqual(model.graph.num_nodes() - 2, rewritten_model.graph.num_nodes())

        # Check that the results are numerically equal
        rng = np.random.default_rng(42)
        inputs = [
            rng.random((1, 3, 32, 32), dtype=np.float32),
        ]
        testing.assert_numerically_equal(model, rewritten_model, inputs)

    def test_affine_conv_fusion_without_pad(self):
        tape = ir.tape.Tape()
        x = ir.Input(
            "x", shape=ir.Shape([1, 3, 32, 32]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        scale = tape.initializer(ir.tensor(np.array([2.0], dtype=np.float32), name="scale"))
        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        mul_out = tape.op("Mul", [x, scale])
        z = tape.op(
            "Add",
            [mul_out, offset],
            output=ir.Input(
                "z",
                shape=ir.Shape([1, 3, 32, 32]),
                type=ir.TensorType(ir.DataType.FLOAT),
            ),
        )
        conv_out = tape.op("Conv", [z, w, b], attributes={"pads": [0, 0, 0, 0]})

        model = ir.Model(
            ir.Graph(
                inputs=[x],
                outputs=[conv_out],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 17},
            ),
            ir_version=8,
        )
        model.display()
        rewritten_model = self.clone_model(model)
        rewritten_model = rewrite(
            rewritten_model,
            pattern_rewrite_rules=[affine_conv_fusion_rule],
        )
        # Check that Mul and Add are fused into Conv
        self.assertEqual(model.graph.num_nodes() - 2, rewritten_model.graph.num_nodes())

        # Check that the results are numerically equal
        rng = np.random.default_rng(42)
        inputs = [
            rng.random((1, 3, 32, 32), dtype=np.float32),
        ]
        testing.assert_numerically_equal(model, rewritten_model, inputs)


if __name__ == "__main__":
    unittest.main()
