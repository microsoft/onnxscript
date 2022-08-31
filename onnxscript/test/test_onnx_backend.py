# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import unittest
import importlib
from onnx.helper import __file__ as onnx_file
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail, NotImplemented, InvalidArgument, RuntimeException)
from onnxscript.backend.onnx_export import export2python
from onnxscript.backend.onnx_backend import enumerate_onnx_tests
from onnxscript.values import OnnxFunction
from onnxscript.eager_mode_evaluator import EagerModeError


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

        import_name = "onnxscript.test.%s.%s" % (
            os.path.split(TestOnnxBackEnd.folder)[-1], name)
        try:
            mod = importlib.import_module(import_name)
        except (SyntaxError, ImportError) as e:
            raise AssertionError(
                "Unable to import %r (file: %r)\n----\n%s" % (
                    import_name, filename, content)) from e
        fcts = {k: v for k, v in mod.__dict__.items() if isinstance(v, OnnxFunction)}
        return fcts

    def common_test_enumerate_onnx_tests_run(self, valid, verbose=0):
        with self.assertRaises(FileNotFoundError):
            list(enumerate_onnx_tests('NNN'))
        missed = []
        load_failed = []
        exec_failed = []
        mismatch = []
        success = 0
        for te in enumerate_onnx_tests('node'):
            if "_scan_" in te.name or "test_scan" in te.name:
                # Operator Scan is not supported by onnx-script.
                continue
            if valid is not None and not valid(te.name):
                continue
            if verbose:
                print("TEST:", te.name)
            with self.subTest(name=te.name):
                if verbose > 1:
                    print("  check onnxruntime")
                    if verbose > 4:
                        print(te.onnx_model)
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
                if verbose > 1:
                    print("  convert into python")
                code = export2python(te.onnx_model, function_name="bck_" + te.name)
                self.assertIn("@script()", code)
                self.assertIn("def bck_%s(" % te.name, code)
                if verbose > 1:
                    print("  check syntax, compilation")
                    if verbose > 2:
                        print(code)
                if te.name == "test_resize_downsample_scales_cubic":
                    self.assertIn("Resize(X, None, scales,", code)
                if "test_loop" in te.name or te.name in {
                        'test_range_float_type_positive_delta_expanded',
                        'test_range_int32_type_negative_delta_expanded'}:
                    # TODO: change change when the converter supports
                    # support something like 'while i < n and cond:'
                    continue
                fcts = self.verify(te.name, code)
                main = fcts["bck_" + te.name]
                self.assertFalse(main is None)
                proto = main.to_model_proto()
                # opset may be different when an binary operator is used.
                if te.onnx_model.ir_version != proto.ir_version:
                    if (not te.name.startswith("test_add") and
                            not te.name.startswith("test_and") and
                            not te.name.startswith("test_div") and
                            not te.name.startswith("test_equal") and
                            not te.name.startswith("test_greater") and
                            not te.name.startswith("test_less") and
                            not te.name.startswith("test_matmul") and
                            not te.name.startswith("test_mod") and
                            not te.name.startswith("test_mul") and
                            not te.name.startswith("test_not") and
                            not te.name.startswith("test_or") and
                            not te.name.startswith("test_pow") and
                            not te.name.startswith("test_sub") and
                            (te.onnx_model.ir_version, proto.ir_version) not in {
                                (3, 4), (5, 6)}):
                        # unexpected behaviour for old opsets
                        raise AssertionError(
                            "Incompatible ir_version %d != %d\n%s\n-----\n%s" % (
                                te.onnx_model.ir_version, proto.ir_version,
                                te.onnx_model, proto))

                # check converted onnx
                def load_fct(obj):
                    if verbose > 2:
                        print("    load ONNX")
                    try:
                        sess = InferenceSession(proto.SerializeToString())
                    except Exception as e:
                        raise AssertionError(
                            "Unable to load onnx for test %r.\n%s\n-----\n%s" % (
                                te.name, str(proto), str(te.onnx_model))) from e
                    if verbose > 2:
                        print("    done.")
                    return sess

                def run_fct(obj, *inputs):
                    if verbose > 2:
                        print("    run ONNX")
                        for i, inp in enumerate(inputs):
                            if inp is None:
                                print("    input %d: None" % i)
                            else:
                                print("    input %d: dtype=%r shape=%r %r" % (
                                    i, inp.dtype, inp.shape, inp.ravel().tolist()))
                    try:
                        res = TestOnnxBackEnd.run_fct(obj, *inputs)
                    except Exception as e:
                        raise AssertionError(
                            "Unable to run test %r after conversion.\n%s" % (
                                te.name, str(proto))) from e
                    if verbose > 2:
                        print("    done.")
                    return res

                if verbose > 1:
                    print("  check ModelProto")
                te.run(load_fct, run_fct)

                # check eager mode
                if not te.name.startswith("test_split") and te.name not in {
                        'test_lstm_defaults',
                        'test_lstm_with_initial_bias',
                        'test_lstm_with_peepholes'}:
                    # split has an undefined number of outputs.
                    # current implementation of eager mode is not aware of them.
                    # same goes for lstm
                    if verbose > 1:
                        print("  check eager")

                    def exec_main(f, *inputs):
                        assert id(f) == id(main)
                        output = f(*inputs)
                        if isinstance(output, tuple):
                            return list(output)
                        return [output]
                    try:
                        te.run(lambda obj: main, exec_main)
                    except EagerModeError as e:
                        # Does not work.
                        if verbose > 0:
                            print("ERROR: ", e)
                        continue
                if verbose > 1:
                    print("  end example.")

        if __name__ == '__main__':
            path = os.path.dirname(onnx_file)
            failed = [len(missed), len(load_failed), len(exec_failed), len(mismatch)]
            print(success, failed)
            print("coverage ratio %f" % (success / (success + sum(failed))))
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

    def test_enumerate_onnx_tests_run(self):
        self.common_test_enumerate_onnx_tests_run(None)

    def test_enumerate_onnx_tests_run_one_case(self):
        self.common_test_enumerate_onnx_tests_run(
            lambda name: "test_layer_normalization_3d_axis2_epsilon_expanded" in name,
            verbose=4 if __name__ == "__main__" else 0)


if __name__ == "__main__":
    # TestOnnxBackEnd().test_enumerate_onnx_tests_run_one_case()
    unittest.main(verbosity=2)
