# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import unittest
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
from onnxscript.values import Opset
from onnxscript import script


def print_code(code, begin=1):
    """
    Returns the code with line number.
    """
    rows = code.split("\n")
    return "\n".join("%03d %s" % (i + begin, s)
                     for i, s in enumerate(rows))


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

    def verify(self, content, more_context=None):
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'Opset': Opset, 'script': script,
               'TensorProto': TensorProto,
               'numpy': numpy,
               'make_tensor': make_tensor}
        if more_context is not None:
            loc.update(more_context)
            glo.update(more_context)
        out, err = StringIO(), StringIO()

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

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
                code = export2python(te.onnx_model)
                glo, loc = self.verify(code)
                main = loc['main']
                self.assertFalse(main is None)
                # print(dir(main))

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
    # TestOnnxBackEnd().test_enumerate_onnx_tests_run()
    unittest.main()
