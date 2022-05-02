# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import types
import numpy as np
import onnx
from onnx.helper import printable_graph
from onnx.onnx_cpp2py_export.checker import ValidationError
import onnxruntime
from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.values import OnnxFunction

TEST_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEST_OUTPUT_DIR = os.path.join(TEST_INPUT_DIR, "testoutputs")


class TestConverter(unittest.TestCase):
    def validate(self, script):
        if isinstance(script, types.ModuleType):
            fnlist = [f for f in script.__dict__.values() if isinstance(f, OnnxFunction)]
        elif isinstance(script, OnnxFunction):
            fnlist = [script]
        else:
            fnlist = script
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
        for f in fnlist:
            with self.subTest(f=f.name):
                model = f.to_function_proto()

    def validate_save(self, script, save_text=False, check_ort=False):
        if isinstance(script, types.ModuleType):
            fnlist = [f for f in script.__dict__.values() if isinstance(f, OnnxFunction)]
        elif isinstance(script, OnnxFunction):
            fnlist = [script]
        else:
            fnlist = script
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
        for f in fnlist:
            with self.subTest(f=f.name):
                model = f.to_model_proto()
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".txt"), 'w') as f:
                        f.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                if check_ort:
                    onnxruntime.InferenceSession(model.SerializeToString())
                model = onnx.shape_inference.infer_shapes(model)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".shape.txt"), 'w') as f:
                        f.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                try:
                    onnx.checker.check_model(model)
                except ValidationError as e:
                    onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".error.onnx"))
                    raise AssertionError(
                        "Verification of model failed.") from e
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))

    def test_error_undefined(self):
        with self.assertRaises(ValueError) as e:
            @script()
            def square(x):
                return op.Mul(undefined, x)
        self.assertIn("string:3", str(e.exception))

    def test_run_ort(self):
        @script()
        def square(x):
            return op.Mul(x, x)
        model = square.to_model_proto()
        sess = onnxruntime.InferenceSession(model.SerializeToString())
        x = np.array([5, 6], dtype=np.float32)
        got = sess.run(None, {'x': x})
        self.assertEqual((x * x).tolist(), got[0].tolist())

    def test_onnxfns1(self):
        from .models import onnxfns1
        self.validate(onnxfns1)

    def test_onnxfns1A(self):
        from .models import onnxfns1A
        self.validate(onnxfns1A)

    def test_models(self):
        from .models import onnxmodels
        self.validate_save(onnxmodels)

    def test_unary_op(self):
        from .models import m1
        self.validate_save(m1)

    def test_subfunction(self):
        from onnxscript.test.models import subfunction
        model = subfunction.MyElu.function_ir.to_model_proto(producer_name='p2o')
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

    def test_if_models(self):
        from .models import if_statement
        self.validate_save(if_statement)

    def test_docstring(self):
        @script()
        def sumprod(x: FLOAT['N'], N: INT64) -> (FLOAT['N'], FLOAT['N']):
            """
            Combines ReduceSum, ReduceProd.
            """
            sum = op.Identity(x)
            prod = op.Identity(x)
            for i in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod
        proto = sumprod.to_function_proto()
        self.assertEqual(proto.doc_string.strip(), "Combines ReduceSum, ReduceProd.")


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    TestConverter().test_unary_op()
    unittest.main()
