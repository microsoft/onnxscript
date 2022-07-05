# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import unittest
import numpy
from numpy import array, float32, int64, int8, int32, uint8
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_graph,
    make_tensor_value_info, make_opsetid, make_tensor,
    __file__ as onnx_file)
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail, NotImplemented, InvalidArgument, RuntimeException)
from onnxscript.backend.onnx_backend import (
    enumerate_onnx_tests, assert_almost_equal_string)


class TestOnnxBackEnd(unittest.TestCase):

    def test_onnx_backend_test(self):
        name = 'test_abs'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te)
        self.assertEqual(len(code), 1)

    @staticmethod
    def load_fct(obj, runtime='python'):
        return InferenceSession(obj.SerializeToString())

    @staticmethod
    def run_fct(obj, *inputs):
        names = [i.name for i in obj.get_inputs()]
        if len(names) < len(inputs):
            raise AssertionError(
                "Got %d inputs but expecting %d." % (
                    len(inputs), len(names)))
        feeds = {names[i]: inputs[i] for i in range(len(inputs))}
        got = obj.run(None, feeds)
        return got

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for te in enumerate_onnx_tests('node', lambda folder: folder == 'test_abs'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_tests_run(self):

        with self.assertRaises(FileNotFoundError):
            list(enumerate_onnx_tests('NNN'))
        missed = []
        load_failed = []
        exec_failed = []
        mismatch = []
        success = 0
        for te in enumerate_onnx_tests('node'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            try:
                te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            except NotImplementedError as e:
                missed.append((te, e))
                continue
            except (IndexError, RuntimeError, TypeError, ValueError,
                    AttributeError, Fail, NotImplemented, InvalidArgument) as e:
                load_failed.append((te, e))
                continue
            except RuntimeException as e:
                exec_failed.append((te, e))
                continue
            except AssertionError as e:
                mismatch.append((te, e))
                continue
            success += 1

        if __name__ == '__main__':
            path = os.path.dirname(onnx_file)
            print(success, len(missed), len(load_failed), len(exec_failed), len(mismatch))
            for t in load_failed:
                print("loading failed",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))
            for t in exec_failed:
                print("execution failed",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))
            for t in mismatch:
                print("mismatch",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))
            for t in missed:
                print("missed",
                      str(t[0]).replace('\\\\', '\\').replace(
                          path, 'onnx').replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
