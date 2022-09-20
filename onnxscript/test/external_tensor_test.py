import os
import shutil
import unittest
import numpy as np
import onnx
from onnxscript import script, proto2python
from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset17 as op


class TempFolder:
    def __init__(self, folder_name=None) -> None:
        if folder_name is None:
            folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TempDir")
        if os.path.exists(folder_name):
            raise ValueError(folder_name + ": already exists!")
        os.makedirs(folder_name)
        self.folder_name = folder_name

    def __enter__(self):
        return self.folder_name

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.folder_name)


class TestConverter(unittest.TestCase):

    def test_external_tensor(self):
        weight = np.random.rand(1024, 10).astype(np.float32)
        bias = np.random.rand(10).astype(np.float32)

        @script()
        def TestFun(X: FLOAT[1024]) -> FLOAT[1024]:
            return op.MatMul(X, weight) + bias

        model = TestFun.to_model_proto()

        with TempFolder() as dir:
            # Convert model to use external-tensors and save
            modelfile = os.path.join(dir, "model.onnx")
            onnx.save_model(model, modelfile, save_as_external_data=True,
                            all_tensors_to_one_file=False, size_threshold=32,
                            convert_attribute=True)

            # Convert model to python:
            pymodel = proto2python(model, clean_code=False)
            self.assertIn("external_tensor('weight', 1, [1024, 10], 'weight', length=40960)",
                          pymodel)
            self.assertIn("external_tensor('bias', 1, [10], 'bias', length=40)", pymodel)


if __name__ == '__main__':
    unittest.main()
