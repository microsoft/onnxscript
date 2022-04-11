# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import textwrap
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnx.helper import printable_graph
from onnx.onnx_cpp2py_export.checker import ValidationError
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, InvalidGraph
from onnxscript.converter import Converter
from onnxscript.values import Opset

TEST_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEST_OUTPUT_DIR = os.path.join(TEST_INPUT_DIR, "testoutputs")


class TestConverter(unittest.TestCase):
    def _convert(self, script):
        converter = Converter()
        return converter.convert(script)

    def _convert_and_save(self, script, save_text=False, check_ort=False,
                          tests=None, decimal=5):
        converter = Converter()
        fnlist = converter.convert(script)
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
        for f in fnlist:
            with self.subTest(f=f.name):
                model = f.to_model_proto(producer_name='p2o')
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".txt"), 'w') as f:
                        f.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                if check_ort:
                    try:
                        onnxruntime.InferenceSession(model.SerializeToString())
                    except (Fail, InvalidArgument, InvalidGraph) as e:
                        onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".error.ort.onnx"))
                        with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".error.ort.txt"), 'w') as f:
                            f.write(str(model))
                        raise AssertionError("onnxruntime failed.") from e                        
                model = onnx.shape_inference.infer_shapes(model)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f.name + ".shape.txt"), 'w') as f:
                        f.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))
                try:
                    onnx.checker.check_model(model)
                except ValidationError as e:
                    onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".error.onnx"))
                    raise AssertionError("Verification of model failed.") from e
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))

                # checking inputs and expected outputs with onnxruntime
                if tests is not None and f.name in tests:
                    test = tests[f.name]
                    try:
                        sess = onnxruntime.InferenceSession(model.SerializeToString())
                    except Fail as e:
                        onnx.save(model, os.path.join(
                            TEST_OUTPUT_DIR, f.name + ".error.ort.onnx"))
                        raise AssertionError("Loading model failed.") from e
                    got = sess.run(None, test['inputs'])
                    self.assertEqual(len(test['expected']), len(got))
                    for e, r in zip(test['expected'], got):
                        assert_almost_equal(e, r, decimal=decimal)

    def test_source_input(self):
        script = textwrap.dedent("""
            def square(x):
                return oxs.Mul(x, x)
            """)
        res = self._convert(script)
        self.assertEqual(len(res), 1)

    def test_source_input_error_undefined(self):
        script = textwrap.dedent("""
            def square(x):
                return oxs.Mul(undefined, x)
            """)
        with self.assertRaises(ValueError) as e:
            self._convert(script)
        self.assertIn("string:3", str(e.exception))

    def test_source_input_ort(self):
        script = textwrap.dedent("""
            def square(x):
                return oxs.Mul(x, x)
            """)
        res = self._convert(script)
        self.assertEqual(len(res), 1)
        proto = res[0].to_graph_proto()
        model = onnx.helper.make_model(
            proto, producer_name='p2o',
            opset_imports=[onnx.helper.make_opsetid("", 15)])
        sess = onnxruntime.InferenceSession(model.SerializeToString())
        x = np.array([5, 6], dtype=np.float32)
        got = sess.run(None, {'x': x})
        self.assertEqual((x * x).tolist(), got[0].tolist())

    def test_msdomain(self):
        # Temporary patch to use com.microsoft domain
        script = textwrap.dedent("""
            def foo(x):
                return msdomain.bar(x, x)
            """)
        self._convert(script)

    def test_onnxfns1(self):
        self._convert(os.path.join(TEST_INPUT_DIR, "onnxfns1.py"))

    def test_onnxfns1A(self):
        self._convert(os.path.join(TEST_INPUT_DIR, "onnxfns1A.py"))

    def test_onnxfft(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "onnxfft.py"),
                               save_text=False, check_ort=True)

    def test_models(self):
        self._convert_and_save(os.path.join(TEST_INPUT_DIR, "onnxmodels.py"))

    def test_subfunction(self):
        from .models import subfunction
        model = subfunction.MyElu.function_ir.to_model_proto(producer_name='p2o')
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

    def test_if_models(self):
        self._convert_and_save(os.path.join(TEST_INPUT_DIR, "if_statement.py"))

    def test_nested_if_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "if_nested.py"))

    def test_loop_models(self):
        self._convert_and_save(os.path.join(TEST_INPUT_DIR, "loop.py"))

    def test_docstring(self):
        res = self._convert(os.path.join(TEST_INPUT_DIR, "docstring.py"))
        self.assertEqual(len(res), 1)
        proto = res[0].to_function_proto(Opset('custom_domain', 1))
        self.assertEqual(proto.doc_string, "\n    Combines ReduceSum, ReduceProd.\n    ")

    def test_dummy_tensor(self):
        self._convert_and_save(
            os.path.join(CURRENT_DIR, "dummy_tensor.py"),
            tests={'dummy_tensor': dict(
                inputs={'N': np.array([3], dtype=np.int64)},
                expected=[np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4]], dtype=np.float32)])})


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    TestConverter().test_onnxfft()
    unittest.main()
