# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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


    def test_onnx_type_to_onnxscript_repr_tensor_with_dim_params(self):
        from onnxscript.onnx_types import onnx_type_to_onnxscript_repr
        from onnx import TypeProto, TensorShapeProto
        onnx_type = TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        dim1 = onnx_type.tensor_type.shape.dim.add()
        dim1.dim_param = 'dim1'
        dim2 = onnx_type.tensor_type.shape.dim.add()
        dim2.dim_param = 'dim2'
        result = onnx_type_to_onnxscript_repr(onnx_type)
        self.assertEqual(result, "FLOAT['dim1','dim2']")


    def test_onnx_type_to_onnxscript_repr_tensor_unknown_rank(self):
        from onnxscript.onnx_types import onnx_type_to_onnxscript_repr
        from onnx import TypeProto
        onnx_type = TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        result = onnx_type_to_onnxscript_repr(onnx_type)
        self.assertEqual(result, "FLOAT[...]")


    def test_onnx_type_to_onnxscript_repr_not_implemented(self):
        from onnxscript.onnx_types import onnx_type_to_onnxscript_repr
        from onnx import TypeProto
        unsupported_type = TypeProto()
        with self.assertRaises(NotImplementedError):
            onnx_type_to_onnxscript_repr(unsupported_type)


    def test_class_getitem_shape_already_specified(self):
        from onnxscript.onnx_types import FLOAT
        with self.assertRaises(ValueError):
            FLOAT[None][None]


    def test_tensor_type_instantiation(self):
        with self.assertRaises(NotImplementedError):
            from onnxscript.onnx_types import TensorType
            TensorType()


    def test_check_dim_invalid_type(self):
        with self.assertRaises(TypeError):
            from onnxscript.onnx_types import _check_dim
            _check_dim(3.14)  # Invalid type, should raise TypeError


if __name__ == "__main__":
    unittest.main()
