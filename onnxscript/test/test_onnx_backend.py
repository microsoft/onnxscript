# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import unittest
import importlib
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_tensor, __file__ as onnx_file)
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail, NotImplemented, InvalidArgument, RuntimeException)
from onnxscript.backend.onnx_export import export2python
from onnxscript.backend.onnx_backend import enumerate_onnx_tests
from onnxscript.values import Opset, OnnxFunction
from onnxscript.onnx_types import ParametricTensor
from onnxscript import script, onnx_opset


def print_code(code, begin=1):
    """
    Returns the code with line number.
    """
    rows = code.split("\n")
    return "\n".join("%03d %s" % (i + begin, s)
                     for i, s in enumerate(rows))


class TestOnnxBackEnd(unittest.TestCase):

    folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                          'onnx_backend_test_code')

    def test_onnx_backend_test(self):
        name = 'test_abs'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te)
        self.assertEqual(len(code), 1)

    @staticmethod
    def load_fct(obj):
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

    def verify(self, name, content, more_context=None):
        if not os.path.exists(TestOnnxBackEnd.folder):
            os.mkdir(TestOnnxBackEnd.folder)
            init = os.path.join(TestOnnxBackEnd.folder, '__init__.py')
            with open(init, "w"):
                pass
        filename = os.path.join(TestOnnxBackEnd.folder, name + ".py")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        import_name = "onnxscript.test.%s.%s" % (TestOnnxBackEnd.folder, name)
        try:
            mod = importlib.import_module(import_name)
        except ImportError as e:
            raise ImportError(
                "Unable to import %r (file: %r)." % (import_name, filename)) from e
        fcts = {k: v for k, v in mod.__dict__.items() if isinstance(v, OnnxFunction)}
        return fcts

    def test_enumerate_onnx_tests_run(self):
        with self.assertRaises(FileNotFoundError):
            list(enumerate_onnx_tests('NNN'))
        missed = []
        load_failed = []
        exec_failed = []
        mismatch = []
        success = 0
        for te in enumerate_onnx_tests('node'):
            with self.subTest(name=te.name):
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
                code = export2python(te.onnx_model, function_name="bck_" + te.name)
                self.assertIn("@script()", code)
                self.assertIn("def bck_%s(" % te.name, code)
                fcts = self.verify(te.name, code)
                main = fcts["bck_" + te.name]
                self.assertFalse(main is None)
                proto = main.to_model_proto()

                # check converted onnx
                load_fct = lambda obj: InferenceSession(proto.SerializeToString())
                te.run(load_fct, TestOnnxBackEnd.run_fct)

                # check eager mode
                def exec_main(f, *inputs):
                    assert id(f) == id(main)
                    output = f(*inputs)
                    if isinstance(output, tuple):
                        return list(output)
                    return [output]

                te.run(lambda obj: main, exec_main)

        if __name__ == '__main__':
            path = os.path.dirname(onnx_file)
            failed = [len(missed), len(load_failed), len(exec_failed), len(mismatch)]
            print(success, failed)
            print("coverage %f%" % (success / sum(failed)))
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
