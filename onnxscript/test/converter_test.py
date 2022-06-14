# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
import os
import warnings
import types
import numpy as np
import onnx
from onnx.helper import printable_graph
from onnx.onnx_cpp2py_export.checker import ValidationError
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidGraph, InvalidArgument
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
                f.to_function_proto()

    def validate_save(self, script, save_text=False, check_ort=False, shape_inference=True):
        if isinstance(script, types.ModuleType):
            fnlist = [f for f in script.__dict__.values() if isinstance(f, OnnxFunction)]
        elif isinstance(script, OnnxFunction):
            fnlist = [script]
        else:
            fnlist = script
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
        fcts = {}
        for f in fnlist:
            with self.subTest(f=f.name):
                model = f.to_model_proto(io_types=FLOAT)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".txt"), 'w') as fi:
                        fi.write(printable_graph(model.graph))
                        for fct in model.functions:
                            fi.write("\n-------------------------\n")
                            fi.write(printable_graph(fct))
                if check_ort:
                    try:
                        onnxruntime.InferenceSession(model.SerializeToString())
                    except (Fail, InvalidGraph, InvalidArgument) as e:
                        raise AssertionError(
                            f"onnxruntime cannot load function "
                            f"{f.name}\n{str(model)}") from e
                if shape_inference:
                    model = onnx.shape_inference.infer_shapes(model)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".shape.txt"), 'w') as fi:
                        fi.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                try:
                    onnx.checker.check_model(model)
                except ValidationError as e:
                    if "Field 'shape' of 'type' is required but missing" in str(e):
                        # input or output shapes are missing because the function
                        # was defined with FLOAT[...].
                        warnings.warn(str(e))
                    else:
                        onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".error.onnx"))
                        raise AssertionError(
                            "Verification of model failed.") from e
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))
                fcts[f.name] = model
        return fcts

    def test_error_undefined(self):
        with self.assertRaises(ValueError) as e:
            @script()
            def square(x):
                return op.Mul(undefined, x)  # noqa: F821
        self.assertIn("square:3", str(e.exception))

    def test_run_ort(self):
        @script()
        def square(x):
            return op.Mul(x, x)
        with self.assertRaises(TypeError) as cm:
            # checking that the function raises an exception when types are not defined.
            square.to_model_proto()
        self.assertIn('square:2', str(cm.exception))
        self.assertIn("Variable 'x' is missing", str(cm.exception))
        model = square.to_model_proto(io_types=FLOAT)
        sess = onnxruntime.InferenceSession(model.SerializeToString())
        x = np.array([5, 6], dtype=np.float32)
        got = sess.run(None, {'x': x})
        self.assertEqual((x * x).tolist(), got[0].tolist())

    def test_onnxfns1(self):
        from onnxscript.test.models import onnxfns1
        self.validate(onnxfns1)

    def test_onnxfns1A(self):
        from onnxscript.test.models import onnxfns1A
        self.validate(onnxfns1A)

    def test_unary_op(self):
        from onnxscript.test.models import m1
        self.validate_save(m1)

    def test_subfunction_check_model(self):
        from onnxscript.test.models import subfunction
        model = subfunction.MyElu.function_ir.to_model_proto(producer_name='p2o')
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

    def test_subfunction(self):
        from onnxscript.test.models import subfunction
        self.validate_save(subfunction, check_ort=True)

    def test_if_models(self):
        from onnxscript.test.models import if_statement
        self.validate_save(if_statement)

    def test_docstring(self):
        @script()
        def sumprod(x: FLOAT['N'], N: INT64) -> (FLOAT['N'], FLOAT['N']):   # noqa: F821
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

    def test_signal(self):
        from onnxscript.test.models import signal_dft
        # shape_inference crashes on stft.
        self.validate_save(signal_dft, shape_inference=False)

    def test_none_as_input(self):
        '''
        Test that use of None as an actual parameter is accepted.
        '''
        @script()
        def clipmax(x: FLOAT, max: FLOAT):  # noqa: F821
            return op.Clip(x, None, max)
        self.validate_save(clipmax)

    def test_type_double(self):
        from onnxscript.test.models import type_double
        fcts = self.validate_save(type_double, check_ort=False)
        f = fcts['double_abs']
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 11)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        f = fcts['double_cast']
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 7)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        f = fcts['double_abs_subgraph']
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 11)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        g = f.graph.node[3].attribute[0].g
        self.assertEqual(g.output[0].type.tensor_type.elem_type, 11)
        g = f.graph.node[3].attribute[1].g
        self.assertEqual(g.output[0].type.tensor_type.elem_type, 11)
        self.validate_save(type_double, check_ort=True)


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # TestConverter().test_none_as_input()
    unittest.main()
