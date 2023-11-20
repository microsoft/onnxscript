# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import importlib
import pathlib
import re
import unittest
from typing import Pattern

import onnxruntime as ort
import parameterized
from onnxruntime.capi import onnxruntime_pybind11_state

import onnxscript
import onnxscript.testing
import onnxscript.values
from onnxscript.backend import onnx_backend, onnx_export
from onnxscript.tests.models import type_double


@dataclasses.dataclass
class SkipInfo:
    pattern: Pattern
    reason: str
    condition: bool


def skip(pattern: str | Pattern, reason: str, *, condition: bool = True):
    """Create a SkipInfo object.

    Args:
        pattern: A string or a regular expression to match the ONNX backend test name.
        reason: The reason why the test is skipped.
        condition: If False, the test is not skipped.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    return SkipInfo(pattern, reason, condition)


SKIP_TESTS = (
    skip(
        r"^test_ai_onnx_ml_array_feature_extractor",
        "ImportError: cannot import name 'opset' from 'onnxscript.onnx_opset'",
    ),
    skip(
        r"^test_ai_onnx_ml_binarizer",
        "ImportError: cannot import name 'opset' from 'onnxscript.onnx_opset'",
    ),
    skip(r"^test_center_crop_pad_crop_negative_axes_hwc", "fixme: ORT segfaults"),
    skip(r"_scan_", "Operator Scan is not supported by onnxscript"),
    skip(r"^test_scan", "Operator Scan is not supported by onnxscript"),
    skip(
        r"^test_split",
        "split has an undefined number of outputs. Current implementation of eager mode is not aware of them",
    ),
    skip(
        r"^test_lstm_defaults",
        "LSTM has an undefined number of outputs. Current implementation of eager mode is not aware of them",
    ),
    skip(
        r"^test_lstm_with_initial_bias",
        "LSTM has an undefined number of outputs. Current implementation of eager mode is not aware of them",
    ),
    skip(
        r"^test_lstm_with_peepholes",
        "LSTM has an undefined number of outputs. Current implementation of eager mode is not aware of them",
    ),
    skip(
        r"^test_optional_get_element_tensor",
        "ONNX backend test produces an invalid graph: https://github.com/onnx/onnx/issues/5067",
    ),
    skip(
        r"test_loop",
        "Change when the converter supports support something like 'while i < n and cond:'",
    ),
    skip(
        r"^test_range_float_type_positive_delta_expanded",
        "Change when the converter supports support something like 'while i < n and cond:'",
    ),
    skip(
        r"^test_range_int32_type_negative_delta_expanded",
        "Change when the converter supports support something like 'while i < n and cond:'",
    ),
)


def load_function(obj):
    return ort.InferenceSession(obj.SerializeToString(), providers=("CPUExecutionProvider",))


def run_function(obj, *inputs):
    names = [i.name for i in obj.get_inputs()]
    if len(names) < len(inputs):
        raise AssertionError(f"Got {len(inputs)} inputs but expecting {len(names)}.")
    feeds = {names[i]: inputs[i] for i in range(len(inputs))}
    got = obj.run(None, feeds)
    return got


def extract_functions(name: str, content: str, test_folder: pathlib.Path):
    if not test_folder.exists():
        test_folder.mkdir(exist_ok=True, parents=True)
        init = test_folder / "__init__.py"
        init.touch(exist_ok=True)
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
    root_folder = pathlib.Path(__file__).parent.parent
    test_folder = root_folder / "tests" / "onnx_backend_test_code"
    temp_folder = root_folder / "tests" / "export"

    def _proto_to_os_and_back(self, proto: onnxscript.FunctionProto, **export_options):
        """Convert a proto to onnxscript code and convert it back to a proto."""
        code = onnx_export.export2python(proto, **export_options)
        map = extract_functions(proto.name, code, TestOnnxBackEnd.temp_folder)
        return map[proto.name]

    def _round_trip_check(self, script_function, **export_options):
        proto = script_function.to_function_proto()
        code = onnx_export.export2python(proto, **export_options)
        map = extract_functions(proto.name, code, TestOnnxBackEnd.temp_folder)
        result_proto = map[proto.name]
        onnxscript.testing.assert_isomorphic(proto, result_proto)

    def test_attr_ref(self):
        """Test functions using attribute-parameters."""
        op = onnxscript.opset17

        @onnxscript.script()
        def fun_with_attr_param(X, dtype: int):
            return op.Cast(X, to=dtype)

        self._round_trip_check(fun_with_attr_param)

    def test_double_attr_val_promotion(self):
        op = onnxscript.opset17

        @onnxscript.script()
        def fun_with_double_attr_promotion(X, dtype: int):
            Y = op.Add(X, dtype)
            Z = op.Add(Y, dtype)
            return Z

        self._round_trip_check(fun_with_double_attr_promotion)

    def test_qualified_domain(self):
        """Test use of qualified domain name."""
        op = onnxscript.opset17
        custom_opset = onnxscript.values.Opset("my.domain.com", 1)

        @onnxscript.script(custom_opset)
        def twice(X):
            return op.Add(X, X)

        self._round_trip_check(twice)

    def test_export2python(self):
        proto = type_double.double_abs_subgraph.to_model_proto()
        code = onnx_export.export2python(proto, rename=True, use_operators=True)
        self.assertIn("v4 = v2 > v1", code)

    @parameterized.parameterized.expand(  # type: ignore[misc]
        [
            (backend_test.name, backend_test)
            for backend_test in onnx_backend.enumerate_onnx_tests("node")
        ]
    )
    def test_export2python_produces_correct_onnx_script_model(
        self, _: str, backend_test: onnx_backend.OnnxBackendTest
    ):
        for skip_info in SKIP_TESTS:
            if skip_info.pattern.match(backend_test.name) and skip_info.condition:
                self.skipTest(skip_info.reason)

        self.assertIn(backend_test.name, repr(backend_test))
        self.assertGreater(len(backend_test), 0)
        try:
            backend_test.run(load_function, run_function)
        except NotImplementedError as e:
            self.skipTest(f"Not implemented {e}")
        except (
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
            AttributeError,
            onnxruntime_pybind11_state.Fail,  # pylint: disable=c-extension-no-member
            onnxruntime_pybind11_state.NotImplemented,  # pylint: disable=c-extension-no-member
            onnxruntime_pybind11_state.InvalidArgument,  # pylint: disable=c-extension-no-member
        ) as e:
            self.skipTest(f"Unable to load the model: {e}")
        except (
            onnxruntime_pybind11_state.RuntimeException  # pylint: disable=c-extension-no-member
        ) as e:
            self.skipTest(f"Unable to run the model: {e}")
        except AssertionError as e:
            self.skipTest(f"ORT result mismatches with the expected: {e}")

        code = onnx_export.export2python(
            backend_test.onnx_model, function_name=f"bck_{backend_test.name}"
        )
        self.assertIn("@script()", code)
        self.assertIn(f"def bck_{backend_test.name}(", code)

        if backend_test.name == "test_resize_downsample_scales_cubic":
            self.assertIn("Resize(X, None, scales,", code)

        functions = extract_functions(backend_test.name, code, self.test_folder)
        main_function = functions[f"bck_{backend_test.name}"]
        self.assertIsNotNone(main_function)
        proto = main_function.to_model_proto()

        # Opset may be different when an binary operator is used.
        if backend_test.onnx_model.ir_version != proto.ir_version:
            if (
                not backend_test.name.startswith(  # pylint: disable=too-many-boolean-expressions
                    "test_add"
                )
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

        try:
            session = ort.InferenceSession(
                proto.SerializeToString(), providers=("CPUExecutionProvider",)
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to load onnx for test {backend_test.name!r}.\n"
                f"{onnxscript.proto2text(proto)}\n"
                f"-----\n"
                f"{backend_test.onnx_model}"
            ) from e

        # Check converted onnx
        def _load_function(_):
            return session

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
        backend_test.run(lambda _: main_function, exec_main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
