# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import itertools
import unittest

import numpy as np
import onnx
import onnx.reference
import onnx_ir as ir

from onnxscript.rewriter import rewrite, testing
from onnxscript.rewriter.rules.common import (
    affine_conv_fusion_rule,
    conv_affine_fusion_rule,
)


class FuseConvAffineTest(unittest.TestCase):
    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def test_broadcast_scalar_shapes(self):
        for before_conv, scale_rank, offset_rank, dtype in itertools.product(
            (False, True), (*range(5), None), (*range(5), None), (np.float32, np.float64)
        ):
            with self.subTest(
                before_conv=before_conv,
                scale_rank=scale_rank,
                offset_rank=offset_rank,
                dtype=dtype,
            ):
                tape = ir.tape.Tape()
                x_array = np.arange(18, dtype=dtype).reshape(1, 2, 3, 3)
                w_array = np.arange(6, dtype=dtype).reshape(3, 2, 1, 1)
                b_array = np.arange(3, dtype=dtype)
                x = ir.val(
                    "x",
                    dtype=ir.DataType.from_numpy(x_array.dtype),
                    shape=ir.Shape(x_array.shape),
                )
                w = tape.initializer(ir.tensor(w_array, name="w"))
                b = tape.initializer(ir.tensor(b_array, name="b"))
                channel_shape = (1, 2 if before_conv else 3, 1, 1)
                scale_shape = (1,) * scale_rank if scale_rank is not None else channel_shape
                offset_shape = (1,) * offset_rank if offset_rank is not None else channel_shape
                scale = tape.initializer(
                    ir.tensor(np.full(scale_shape, 2, dtype=dtype), name="scale")
                )
                offset = tape.initializer(
                    ir.tensor(np.full(offset_shape, 3, dtype=dtype), name="offset")
                )
                if before_conv:
                    affine = tape.op("Add", [tape.op("Mul", [x, scale]), offset])
                    y = tape.op("Conv", [affine, w, b], attributes={"pads": [0] * 4})
                    rule = affine_conv_fusion_rule
                    expected = np.einsum("nchw,oc->nohw", x_array * 2 + 3, w_array[:, :, 0, 0])
                    expected += b_array.reshape(1, 3, 1, 1)
                else:
                    conv = tape.op("Conv", [x, w, b])
                    y = tape.op("Add", [tape.op("Mul", [conv, scale]), offset])
                    rule = conv_affine_fusion_rule
                    expected = np.einsum("nchw,oc->nohw", x_array, w_array[:, :, 0, 0])
                    expected = (expected + b_array.reshape(1, 3, 1, 1)) * 2 + 3
                y.name = "y"
                y.type = x.type
                y.shape = ir.Shape([1, 3, 3, 3])
                model = ir.Model(
                    ir.Graph(
                        name="broadcast_scalar",
                        inputs=[x],
                        outputs=[y],
                        nodes=tape.nodes,
                        initializers=tape.initializers,
                        opset_imports={"": 18},
                    ),
                    ir_version=8,
                )
                original = ir.to_proto(model)
                onnx.checker.check_model(original, full_check=True)
                rewritten = rewrite(original, pattern_rewrite_rules=[rule])
                if scale_rank is None or offset_rank is None:
                    # Per-channel operands remain outside these scalar-only rules.
                    self.assertEqual(rewritten, original)
                    continue
                self.assertEqual([node.op_type for node in rewritten.graph.node], ["Conv"])
                fused = ir.from_proto(rewritten).graph[0]
                self.assertEqual(fused.inputs[1].const_value.numpy().shape, w_array.shape)
                self.assertEqual(fused.inputs[2].const_value.numpy().shape, b_array.shape)
                self.assertEqual(fused.inputs[1].const_value.numpy().dtype, dtype)
                self.assertEqual(fused.inputs[2].const_value.numpy().dtype, dtype)
                onnx.checker.check_model(rewritten, full_check=True)
                for graph in (original, rewritten):
                    actual = onnx.reference.ReferenceEvaluator(graph).run(
                        None, {"x": x_array}
                    )[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_affine_rank_promotion_not_fused(self):
        for (before_conv, unknown_rank), promoting_operand in itertools.product(
            ((False, False), (True, False), (True, True)), ("scale", "offset")
        ):
            with self.subTest(
                before_conv=before_conv,
                unknown_rank=unknown_rank,
                promoting_operand=promoting_operand,
            ):
                tape = ir.tape.Tape()
                x_array = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
                if before_conv:
                    x_array = x_array[0]
                x = ir.val("x", dtype=ir.DataType.FLOAT, shape=ir.Shape(x_array.shape))
                w = tape.initializer(
                    ir.tensor(np.arange(6, dtype=np.float32).reshape(3, 2, 1, 1), name="w")
                )
                b = tape.initializer(ir.tensor(np.arange(3, dtype=np.float32), name="b"))
                promoted_shape = (1,) * (x_array.ndim + 1)
                scale = tape.initializer(
                    ir.tensor(
                        np.full(
                            promoted_shape if promoting_operand == "scale" else (),
                            2,
                            np.float32,
                        ),
                        name="scale",
                    )
                )
                offset = tape.initializer(
                    ir.tensor(
                        np.full(
                            promoted_shape if promoting_operand == "offset" else (),
                            3,
                            np.float32,
                        ),
                        name="offset",
                    )
                )
                if before_conv:
                    affine_input = tape.op("Identity", [x]) if unknown_rank else x
                    affine = tape.op("Add", [tape.op("Mul", [affine_input, scale]), offset])
                    y = tape.op("Conv", [affine, w, b], attributes={"pads": [0] * 4})
                    rule = affine_conv_fusion_rule
                    output_shape = [1, 3, 3, 3]
                else:
                    conv = tape.op("Conv", [x, w, b])
                    y = tape.op("Add", [tape.op("Mul", [conv, scale]), offset])
                    rule = conv_affine_fusion_rule
                    output_shape = [1, 1, 3, 3, 3]
                y.name = "y"
                y.type = x.type
                y.shape = ir.Shape(output_shape)
                model = ir.Model(
                    ir.Graph(
                        name="rank_promotion",
                        inputs=[x],
                        outputs=[y],
                        nodes=tape.nodes,
                        initializers=tape.initializers,
                        opset_imports={"": 18},
                    ),
                    ir_version=8,
                )
                original = ir.to_proto(model)
                onnx.checker.check_model(original, full_check=True)
                expected = onnx.reference.ReferenceEvaluator(original).run(
                    None, {"x": x_array}
                )[0]
                self.assertEqual(list(expected.shape), output_shape)
                rewritten = rewrite(original, pattern_rewrite_rules=[rule])
                self.assertEqual(rewritten, original)

    def test_conv_affine_fusion(self):
        tape = ir.tape.Tape()
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32]))
        w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        scale = tape.initializer(ir.tensor(np.array([2.0], dtype=np.float32), name="scale"))
        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        conv_out = tape.op("Conv", [x, w, b], attributes={"pads": [1, 1, 1, 1]})
        mul_out = tape.op("Mul", [conv_out, scale])
        z = tape.op(
            "Add",
            [mul_out, offset],
            output=ir.val(
                "z",
                dtype=ir.DataType.FLOAT,
                shape=ir.Shape([1, 3, 32, 32]),
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
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32]))
        w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        scale = tape.initializer(ir.tensor(np.array([2.0], dtype=np.float32), name="scale"))
        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        mul_out = tape.op("Mul", [x, scale])
        z = tape.op(
            "Add",
            [mul_out, offset],
            output=ir.val(
                "z",
                dtype=ir.DataType.FLOAT,
                shape=ir.Shape([1, 3, 32, 32]),
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
