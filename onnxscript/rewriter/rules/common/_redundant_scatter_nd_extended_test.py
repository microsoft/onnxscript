# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for redundant ScatterND rules.

Adds coverage for: dynamic full-range scatter on axis=0 (positive),
partial scatter (negative for static), and shape mismatch (negative for static).
"""

import unittest

import numpy as np
import onnx.parser
import onnx_ir as ir
import onnxruntime
from onnx_ir.passes.common import CheckerPass, ShapeInferencePass

from onnxscript import FLOAT, optimizer, script
from onnxscript import opset18 as op
from onnxscript.rewriter.rules.common import _redundant_scatter_nd

N = "N"
shape_inference = ShapeInferencePass()
onnx_check = CheckerPass(True)


class RedundantScatterNdExtendedTest(unittest.TestCase):
    """Extended tests for redundant ScatterND rewrite rules."""

    # --- Positive: axis=0 dynamic ---

    def test_dynamic_indices_axis_0(self):
        """Full-range scatter on axis=0 → should be eliminated."""

        @script()
        def model_fn(data: FLOAT[N, 16], updates: FLOAT[N, 16]) -> FLOAT[N, 16]:
            axis = op.Constant(value_int=0)
            shape = op.Shape(data, start=0)
            dim = op.Gather(shape, axis, axis=0)
            full_range = op.Range(0, dim, 1)
            full_range_2d = op.Unsqueeze(full_range, [-1])
            scattered = op.ScatterND(data, full_range_2d, updates, reduction="none")
            return scattered

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        onnx_check(model)
        shape_inference(model)
        optimizer.fold_constants(model)
        count = _redundant_scatter_nd.rules.apply_to_model(model)
        self.assertEqual(count, 1)

        # Verify numerical equivalence
        inputs = {
            "data": np.random.rand(8, 16).astype(np.float32),
            "updates": np.random.rand(8, 16).astype(np.float32),
        }
        original_session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_outputs = original_session.run(None, inputs)
        optimized_proto = ir.serde.serialize_model(model)
        optimized_session = onnxruntime.InferenceSession(
            optimized_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_outputs = optimized_session.run(None, inputs)
        np.testing.assert_allclose(original_outputs[0], optimized_outputs[0])

    # --- Negative: partial indices (static) ---

    def test_static_partial_indices_no_fusion(self):
        """Indices that don't cover full dim → should NOT be eliminated."""
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[8, 16] data, float[4, 16] updates) => (float[8, 16] output)
            {
                output = ScatterND (data, indices, updates)
            }
            """
        )
        # Only scatter to first 4 rows of 8
        indices = np.arange(4).reshape(4, 1).astype(np.int64)
        model = ir.serde.deserialize_model(model_proto)
        indices_value = model.graph[0].inputs[1]
        indices_value.const_value = ir.Tensor(name="indices", value=indices)
        model.graph.initializers["indices"] = indices_value

        count = _redundant_scatter_nd.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("ScatterND", op_types)

    # --- Negative: shape mismatch (static) ---

    def test_static_shape_mismatch_no_fusion(self):
        """data.shape != updates.shape → should NOT be eliminated."""
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[8, 16] data, float[8, 32] updates) => (float[8, 16] output)
            {
                output = ScatterND (data, indices, updates)
            }
            """
        )
        indices = np.arange(8).reshape(8, 1).astype(np.int64)
        model = ir.serde.deserialize_model(model_proto)
        indices_value = model.graph[0].inputs[1]
        indices_value.const_value = ir.Tensor(name="indices", value=indices)
        model.graph.initializers["indices"] = indices_value

        count = _redundant_scatter_nd.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("ScatterND", op_types)


if __name__ == "__main__":
    unittest.main()
