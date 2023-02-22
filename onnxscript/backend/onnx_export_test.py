# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# pylint: disable=too-many-boolean-expressions

import importlib
import os
import pathlib
import unittest

import onnxruntime as ort
import parameterized
from onnx.helper import __file__ as onnx_file
from onnxruntime.capi import onnxruntime_pybind11_state

import onnxscript
from onnxscript import evaluator
from onnxscript.backend import onnx_backend, onnx_export
from onnxscript.tests.models import type_double


def load_function(obj):
    return ort.InferenceSession(obj.SerializeToString())


def run_function(obj, *inputs):
    names = [i.name for i in obj.get_inputs()]
    if len(names) < len(inputs):
        raise AssertionError(f"Got {len(inputs)} inputs but expecting {len(names)}.")
    feeds = {names[i]: inputs[i] for i in range(len(inputs))}
    got = obj.run(None, feeds)
    return got


def extract_functions(name: str, content: str, test_folder: pathlib.Path):
    """Write the content into a file and import all OnnxFunctions from it."""
    if not test_folder.exists():
        test_folder.mkdir()
        init = test_folder / "__init__.py"
        init.touch()
    file = test_folder / f"{name}.py"
    file.write_text(content, encoding="utf-8")

    import_name = f"onnxscript.tests.{test_folder.parts[-1]}.{name}"
    try:
        mod = importlib.import_module(import_name)
    except (SyntaxError, ImportError) as e:
        raise AssertionError(
            f"Unable to import {import_name!r} (file: {file!r})\n----\n{content}"
        ) from e
    functions = {
        k: v for k, v in mod.__dict__.items() if isinstance(v, onnxscript.OnnxFunction)
    }
    return functions


def exec_main(f, *inputs):
    output = f(*inputs)
    if isinstance(output, tuple):
        return list(output)
    return [output]


class TestOnnxBackEnd(unittest.TestCase):

    test_folder = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "onnx_backend_test_code"
    )

    def test_export2python(self):
        proto = type_double.double_abs_subgraph.to_model_proto()
        code = onnx_export.export2python(proto, rename=True, use_operators=True)
        self.assertIn("v4 = v2 > v1", code)

    @parameterized.parameterized.expand(
        [
            (backend_test.name, backend_test)
            for backend_test in onnx_backend.enumerate_onnx_tests("node")
        ]
    )
    def test_enumerate_onnx_tests_run(
        self, _: str, backend_test: onnx_backend.OnnxBackendTest
    ):
        if "_scan_" in backend_test.name or "test_scan" in backend_test.name:
            self.skipTest("Operator Scan is not supported by onnx-script")

        self.assertIn(backend_test.name, repr(backend_test))
        self.assertGreater(len(backend_test), 0)
        try:
            backend_test.run(load_function, run_function)
        except NotImplementedError as e:
            self.skipTest(f"Not implemented {e}")

        code = onnx_export.export2python(
            backend_test.onnx_model, function_name=f"bck_{backend_test.name}"
        )
        self.assertIn("@script()", code)
        self.assertIn(f"def bck_{backend_test.name}(", code)

        if backend_test.name == "test_resize_downsample_scales_cubic":
            self.assertIn("Resize(X, None, scales,", code)
        if "test_loop" in backend_test.name or backend_test.name in {
            "test_range_float_type_positive_delta_expanded",
            "test_range_int32_type_negative_delta_expanded",
        }:
            # TODO: change change when the converter supports
            # support something like 'while i < n and cond:'
            return
        functions = extract_functions(backend_test.name, code, pathlib.Path(self.test_folder))
        main_function = functions[f"bck_{backend_test.name}"]
        self.assertIsNotNone(main_function)
        proto = main_function.to_model_proto()

        # Opset may be different when an binary operator is used.
        if backend_test.onnx_model.ir_version != proto.ir_version:
            if (
                not backend_test.name.startswith("test_add")
                and not backend_test.name.startswith("test_and")
                and not backend_test.name.startswith("test_div")
                and not backend_test.name.startswith("test_equal")
                and not backend_test.name.startswith("test_greater")
                and not backend_test.name.startswith("test_less")
                and not backend_test.name.startswith("test_matmul")
                and not backend_test.name.startswith("test_mod")
                and not backend_test.name.startswith("test_mul")
                and not backend_test.name.startswith("test_not")
                and not backend_test.name.startswith("test_or")
                and not backend_test.name.startswith("test_pow")
                and not backend_test.name.startswith("test_sub")
                and (backend_test.onnx_model.ir_version, proto.ir_version)
                not in {(3, 4), (5, 6)}
            ):
                # Unexpected behavior for old opsets
                raise AssertionError(
                    f"Incompatible ir_version {(backend_test.onnx_model.ir_version)} !="
                    f" {(proto.ir_version)}\n"
                    f"{backend_test.onnx_model}\n"
                    f"-----\n"
                    f"{proto}"
                )

        # Check converted onnx
        def _load_function(_):
            try:
                return ort.InferenceSession(proto.SerializeToString())
            except Exception as e:
                raise AssertionError(
                    f"Unable to load onnx for test {backend_test.name!r}.\n"
                    f"{onnxscript.proto2text(proto)}\n"
                    f"-----\n"
                    f"{backend_test.onnx_model}"
                ) from e

        def _run_function(obj, *inputs):
            print("    run ONNX")
            for i, inp in enumerate(inputs):
                if inp is None:
                    print(f"    input {i}: None")
                else:
                    print(
                        f"    input {i}: "
                        f"dtype={inp.dtype!r} shape={inp.shape!r}"
                        f"{inp.ravel().tolist()!r}"
                    )
            try:
                return run_function(obj, *inputs)
            except Exception as e:
                raise AssertionError(
                    f"Unable to run test {backend_test.name!r} after conversion.\n"
                    f"{onnxscript.proto2text(proto)}"
                ) from e

        backend_test.run(_load_function, _run_function)

        # Check eager mode
        if backend_test.name.startswith("test_split") or backend_test.name in {
            "test_lstm_defaults",
            "test_lstm_with_initial_bias",
            "test_lstm_with_peepholes",
        }:
            # split has an undefined number of outputs.
            # current implementation of eager mode is not aware of them.
            # same goes for lstm
            self.skipTest(backend_test.name)

        backend_test.run(lambda _: main_function, exec_main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
