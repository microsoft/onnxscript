# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import textwrap
import numpy as np
import onnx
from onnx.onnx_cpp2py_export.checker import ValidationError
import onnxruntime
from onnxscript.converter import Converter

CURRENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class TestConverter(unittest.TestCase):
    def _convert(self, script):
        converter = Converter()
        return converter.convert(script)

    def _convert_and_save(self, script):
        converter = Converter()
        fnlist = converter.convert(script)
        TEST_OUTPUT_DIR = os.path.join(CURRENT_DIR, "testoutputs")
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
        for f in fnlist:
            model = f.to_model_proto(producer_name='p2o')
            model = onnx.shape_inference.infer_shapes(model)
            try:
                onnx.checker.check_model(model)
            except ValidationError as e:
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".error.onnx"))
                raise AssertionError(
                    "Verification of model failed.") from e
            onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))

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

    def test_onnxfns(self):
        self._convert(os.path.join(CURRENT_DIR, "onnxfns.py"))

    def test_onnxfft(self):
        self._convert(os.path.join(CURRENT_DIR, "onnxfft.py"))

    def test_onnxfns_with_cast(self):
        self._convert(os.path.join(CURRENT_DIR, "onnxfns_with_cast.py"))

    def test_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "onnxmodels.py"))

    def test_subfunction(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "subfunction.py"))

    def test_if_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "if.py"))

    def test_nested_if_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "if_nested.py"))

    def test_loop_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "loop.py"))

    def test_docstring(self):
        res = self._convert(os.path.join(CURRENT_DIR, "docstring.py"))
        self.assertEqual(len(res), 1)
        proto = res[0].to_function_proto()
        self.assertEqual(proto.doc_string, "\n    Combines ReduceSum, ReduceProd.\n    ")


if __name__ == '__main__':
    unittest.main()
