# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa: F821

import unittest

import numpy as np
import onnx.parser
import onnx_ir as ir
import onnxruntime
from onnx_ir.passes.common import CheckerPass, ShapeInferencePass

import onnxscript.optimizer
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter import redundant_scatter_nd

shape_inference = ShapeInferencePass()
onnx_check = CheckerPass(True)


class RedundantScatterNdTest(unittest.TestCase):
    def test_redundant_scatter_nd_dynamic_indices(self):
        """Test redundant ScatterND with dynamically constructed indices."""

        @script()
        def model_script(
            data: FLOAT[8, "N", 16], updates: FLOAT[8, "N", 16]
        ) -> FLOAT[8, "N", 16]:
            # Construct update-indices spanning an entire axis:
            axis = op.Constant(value_int=1)
            shape = op.Shape(data, start=0)
            dim = op.Gather(shape, axis, axis=0)
            full_range = op.Range(0, dim, 1)
            full_range_2d = op.Unsqueeze(full_range, [-1])
            # The update is applied to the data transposed to bring the updated axis to the front:
            transposed_data = op.Transpose(data, perm=[1, 0, 2])
            transposed_updates = op.Transpose(updates, perm=[1, 0, 2])
            scattered = op.ScatterND(
                transposed_data, full_range_2d, transposed_updates, reduction="none"
            )
            # Transpose the result back to the original shape:
            output = op.Transpose(scattered, perm=[1, 0, 2])
            return output

        input_model_proto = model_script.to_model_proto()
        model = ir.serde.deserialize_model(input_model_proto)
        onnx_check(model)
        shape_inference(model)
        onnxscript.optimizer.fold_constants(model)
        count = redundant_scatter_nd.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        onnx_check(model)
        optimized_model_proto = ir.serde.serialize_model(model)
        # Test that both models are equivalent:
        inputs = {
            "data": np.random.rand(8, 4, 16).astype(np.float32),
            "updates": np.random.rand(8, 4, 16).astype(np.float32),
        }
        session = onnxruntime.InferenceSession(
            input_model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(None, inputs)
        optimized_session = onnxruntime.InferenceSession(
            optimized_model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_outputs = optimized_session.run(None, inputs)
        # Compare outputs
        for output, optimized_output in zip(outputs, optimized_outputs):
            np.testing.assert_allclose(output, optimized_output, rtol=1e-6, atol=1e-6)

    def test_redundant_scatter_nd_static_indices(self):
        """Test redundant ScatterND with static indices (moved from collapse_slices_test.py)."""
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[112, 16, 512] data, float[112, 16, 512] updates) => (float[112, 16, 512] output)
            {
                output = ScatterND (data, indices, updates)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        indices = np.arange(112).reshape(112, 1).astype(np.int64)
        model = ir.serde.deserialize_model(model_proto)
        # from numpy to ir.Tensor
        indices_ir_tensor = ir.Tensor(
            name="indices",
            value=indices,
        )
        # assign the tensor to a value
        indices_value = model.graph[0].inputs[1]
        indices_value.const_value = indices_ir_tensor
        model.graph.initializers["indices"] = indices_value
        original_model_proto = ir.serde.serialize_model(model)

        count = redundant_scatter_nd.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertIn("Identity", [node.op_type for node in model.graph])

        # Test numerical equivalence
        input_data = np.random.rand(112, 16, 512).astype(np.float32)
        inputs = {"data": input_data, "updates": input_data}

        # Run original model
        session = onnxruntime.InferenceSession(
            original_model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_outputs = session.run(None, inputs)

        # Run optimized model
        optimized_model_proto = ir.serde.serialize_model(model)
        optimized_session = onnxruntime.InferenceSession(
            optimized_model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_outputs = optimized_session.run(None, inputs)

        # Compare outputs
        for original_output, optimized_output in zip(original_outputs, optimized_outputs):
            np.testing.assert_allclose(original_output, optimized_output, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
