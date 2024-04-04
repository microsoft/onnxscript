import os
import tempfile
import unittest

import numpy as np
import onnx

from onnxscript import proto2python, script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT


class TestConverter(unittest.TestCase):
    def test_external_tensor(self):
        weight = np.random.rand(1024, 10).astype(np.float32)
        bias = np.random.rand(10).astype(np.float32)

        @script()
        def TestFun(X: FLOAT[1024]) -> FLOAT[1024]:
            return op.MatMul(X, weight) + bias

        model = TestFun.to_model_proto()

        with tempfile.TemporaryDirectory() as dir:
            # Convert model to use external-tensors and save
            modelfile = os.path.join(dir, "model.onnx")
            onnx.save_model(
                model,
                modelfile,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
                size_threshold=32,
                convert_attribute=True,
            )

            # Convert model to python:
            pymodel = proto2python(model)
            self.assertIn(
                "external_tensor('weight', 1, [1024, 10], 'weight', length=40960)", pymodel
            )
            self.assertIn("external_tensor('bias', 1, [10], 'bias', length=40)", pymodel)


if __name__ == "__main__":
    unittest.main()
