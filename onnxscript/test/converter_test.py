# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import onnx
from onnxscript.converter import Converter

CURRENT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "models")


class TestConverter(unittest.TestCase):
    def _convert(self, script):
        converter = Converter()
        converter.convert(script)

    def _convert_and_save(self, script):
        converter = Converter()
        fnlist = converter.convert(script)
        TEST_OUTPUT_DIR = os.path.join(CURRENT_DIR, "testoutputs")
        if (not os.path.exists(TEST_OUTPUT_DIR)):
            os.makedirs(TEST_OUTPUT_DIR)
        for f in fnlist:
            graph = f.to_graph_proto()
            model = onnx.helper.make_model(graph, producer_name='p2o', opset_imports=[
                                           onnx.helper.make_opsetid("", 15)])
            model = onnx.shape_inference.infer_shapes(model)
            onnx.checker.check_model(model)
            onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f.name + ".onnx"))

    def test_source_input(self):
        script = """
def square(x):
    return onnx.Mul(x, x)
"""
        self._convert(script)

    def test_msdomain(self):
        # Temporary patch to use com.microsoft domain
        script = """
def foo(x):
    return msdomain.bar(x, x)
"""
        self._convert(script)

    def test_onnxfns(self):
        self._convert(os.path.join(CURRENT_DIR, "onnxfns.py"))

    def test_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "onnxmodels.py"))

    def test_if_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "if.py"))

    def test_loop_models(self):
        self._convert_and_save(os.path.join(CURRENT_DIR, "loop.py"))


if __name__ == '__main__':
    unittest.main()
