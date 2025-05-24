# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import parameterized

import onnx
from onnxscript import script, FLOAT, INT64, opset18
from onnxscript import ir
from onnxscript.rewriter.ort_fusions import shape_optimization
import numpy as np

def _make_model(starts: list[int], ends: list[int]) -> onnx.ModelProto:
    @script()
    def model_script(
        x: FLOAT["N"],
        dim0: INT64[1],
        dim1: INT64[1],
        dim2: INT64[1],
        dim3: INT64[1],
    ) -> INT64["M"]:
        shape = opset18.Concat(dim0, dim1, dim2, dim3, axis=0)
        reshaped = opset18.Reshape(x, shape, allowzero=1)
        transposed = opset18.Transpose(reshaped, perm=[0, 2, 1, 3])
        final_shape = opset18.Shape(transposed)
        final_dim = opset18.Slice(final_shape, starts, ends)
        return opset18.Add(final_dim, final_dim)

    model_proto = model_script.to_model_proto()
    return model_proto

# Example input data
_model_inputs = {
    "x": np.zeros((24,), dtype=np.float32),
    "dim0": np.array([2], dtype=np.int64),
    "dim1": np.array([3], dtype=np.int64),
    "dim2": np.array([4], dtype=np.int64),
    "dim3": np.array([1], dtype=np.int64),
}

class ShapeOptimizationTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ([0], [1], "singleton"),
        ([1], [3], "two_elements"),
        ([1], [-1], "negative_index"),
        ([-2], [1000], "out_of_bounds"),
        ([-200], [-1], "negative_out_of_bounds"),
        ([2],[2], "empty_slice"),
    ])
    def test_shape_optimization(self, starts: list[int], ends: list[int], _name: str):
        model_proto = _make_model(starts, ends)
        model = ir.serde.deserialize_model(model_proto)

        count = shape_optimization.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        optimized_proto = ir.serde.serialize_model(model)

        import onnxruntime as ort
        sess = ort.InferenceSession(model_proto.SerializeToString(), providers=["CPUExecutionProvider"])
        outputs = sess.run(None, _model_inputs)
        sess = ort.InferenceSession(optimized_proto.SerializeToString(), providers=["CPUExecutionProvider"])
        optimized_outputs = sess.run(None, _model_inputs)
        for orig, opt in zip(outputs, optimized_outputs):
            np.testing.assert_array_equal(orig, opt)               

if __name__ == "__main__":
    unittest.main()