# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
import inspect
import os
import pathlib
import sys
import textwrap
import types
import unittest
import warnings

import numpy as np
import onnx
import onnxruntime
from numpy.testing import assert_almost_equal
from onnx import TensorProto
from onnx.helper import make_tensor, printable_graph
from onnx.onnx_cpp2py_export import checker
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
)
from packaging.version import Version

from onnxscript import OnnxFunction, converter, graph, script, tensor
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tests.common import onnx_script_test_case, testutils

TEST_INPUT_DIR = pathlib.Path(__file__).parent / "tests" / "models"
TEST_OUTPUT_DIR = TEST_INPUT_DIR / "testoutputs"


class TestConverter(testutils.TestBase):
    def validate(self, script):
        if isinstance(script, types.ModuleType):
            fnlist = [f for f in script.__dict__.values() if isinstance(f, OnnxFunction)]
        elif isinstance(script, OnnxFunction):
            fnlist = [script]
        else:
            fnlist = script
        if not TEST_INPUT_DIR.exists():
            TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for f in fnlist:
            with self.subTest(f=f.name):
                f.to_function_proto()

    def validate_save(
        self,
        script,
        save_text=False,
        check_ort=False,
        shape_inference=True,
        skip_check_ort=None,
    ):
        if isinstance(script, types.ModuleType):
            fnlist = [f for f in script.__dict__.values() if isinstance(f, OnnxFunction)]
        elif isinstance(script, OnnxFunction):
            fnlist = [script]
        else:
            fnlist = script
        if not TEST_OUTPUT_DIR.exists():
            TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fcts = {}
        for f in fnlist:
            with self.subTest(f=f.name):
                model = f.to_model_proto(io_types=FLOAT)
                if save_text:
                    with (TEST_OUTPUT_DIR / f"{f.name}.txt").open("w", encoding="utf-8") as fi:
                        fi.write(printable_graph(model.graph))
                        for fct in model.functions:
                            fi.write("\n-------------------------\n")
                            fi.write(printable_graph(fct))
                if check_ort and (skip_check_ort is None or f.name not in skip_check_ort):
                    try:
                        onnxruntime.InferenceSession(model.SerializeToString())
                    except (Fail, InvalidGraph, InvalidArgument) as e:
                        raise AssertionError(
                            f"onnxruntime cannot load function " f"{f.name}\n--\n{model}"
                        ) from e
                if shape_inference:
                    model = onnx.shape_inference.infer_shapes(model)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f"{f.name}.shape.txt"), "w") as fi:
                        fi.write(printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(printable_graph(fct))
                try:
                    onnx.checker.check_model(model)
                except checker.ValidationError as e:
                    if "Field 'shape' of 'type' is required but missing" in str(
                        e
                    ) or "Field 'shape' of type is required but missing" in str(e):
                        # input or output shapes are missing because the function
                        # was defined with FLOAT[...].
                        warnings.warn(str(e))
                    else:
                        onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f"{f.name}.error.onnx"))
                        raise AssertionError("Verification of model failed.") from e
                onnx.save(model, os.path.join(TEST_OUTPUT_DIR, f"{f.name}.onnx"))
                fcts[f.name] = model
        return fcts

    def validate_expansion(self, script):
        functions = self.validate_save(script, check_ort=True)
        for name in functions:
            if not name.endswith("_expanded"):
                f = functions[name]
                name_expanded = f"{name}_expanded"
                if name_expanded in functions:
                    with self.subTest("Expansion test", function=name):
                        f_expanded = functions[name_expanded]
                        self.assertSame(f, f_expanded)

    def validate_run(self, script_tests):
        for key, val in script_tests.__dict__.items():
            if isinstance(val, onnx_script_test_case.FunctionTestParams):
                with self.subTest(name=key):
                    self.check_run(val.function, val.input, val.output[0])

    def test_eager_op(self):
        from onnxscript.tests.models import eager_op

        test_functions = self.validate_save(eager_op, check_ort=True)

        x = np.array([0, 5, -2], dtype=np.float32)

        onx = test_functions["eager_op"]
        self.assertIn('name: "fmod"', str(onx))
        session = onnxruntime.InferenceSession(onx.SerializeToString())
        y = session.run(None, {"X": x})[0]
        self.assertEqual(y.tolist(), [0.0, 0.5, -0.5])
        # numpy fmod and operator % disagree on this example
        res = eager_op.eager_op(x)
        self.assertEqual(res.tolist(), [0.0, 0.5, -0.5])

        onx = test_functions["eager_abs"]
        session = onnxruntime.InferenceSession(onx.SerializeToString())
        y = session.run(None, {"X": x})[0]
        self.assertEqual(y.tolist(), [1, 6, 3])
        res = eager_op.eager_abs(x)
        self.assertEqual(res.tolist(), [1, 6, 3])

    def test_error_undefined(self):
        with self.assertRaises(ValueError) as e:

            @script()
            def square(x):
                return op.Mul(undefined, x)  # noqa: F821

        self.assertIn("square:3", str(e.exception))

    def test_model_generation(self):
        @script()
        def cast_add(x, y):
            return op.Mul(x, op.CastLike(y, x))

        # Converting "cast_add" to a ModelProto will generate an incomplete ModelProto,
        # with input-types undefined (since the script has no type-annotation).
        model = cast_add.to_model_proto()
        x_value_info = model.graph.input[0]
        self.assertFalse(x_value_info.HasField("type"))

        # Specify input-types in the call to to_model_proto to generate complete ModelProto.
        model = cast_add.to_model_proto(io_types=FLOAT["N"])
        x_value_info = model.graph.input[0]
        y_value_info = model.graph.input[1]
        output_value_info = model.graph.output[0]
        self.assertEqual(x_value_info.type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(y_value_info.type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(output_value_info.type.tensor_type.elem_type, TensorProto.FLOAT)

        # Or, use input_types and output_types, as below, for the more general case.
        model = cast_add.to_model_proto(
            input_types=[FLOAT["N"], INT64["N"]], output_types=[FLOAT["N"]]
        )
        x_value_info = model.graph.input[0]
        y_value_info = model.graph.input[1]
        output_value_info = model.graph.output[0]
        self.assertEqual(x_value_info.type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(y_value_info.type.tensor_type.elem_type, TensorProto.INT64)
        self.assertEqual(output_value_info.type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_onnxfns1(self):
        from onnxscript.tests.models import onnxfns1

        self.validate(onnxfns1)

    def test_onnxfns1A(self):
        from onnxscript.tests.models import onnxfns1A

        self.validate(onnxfns1A)

    def test_ort_custom_ops(self):
        from onnxscript.tests.functions import ort_custom_ops

        self.validate(ort_custom_ops)

    def test_unary_op(self):
        from onnxscript.tests.models import m1

        self.validate_save(m1)

    def test_subfunction_check_model(self):
        from onnxscript.tests.models import subfunction

        model = subfunction.MyElu.function_ir.to_model_proto(producer_name="p2o")
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

    @unittest.skipIf(
        Version(onnxruntime.__version__) < Version("1.12"),
        reason="onnxruntime does not support that scenario.",
    )
    def test_subfunction(self):
        from onnxscript.tests.models import subfunction

        self.validate_save(subfunction, check_ort=True)

    def test_if_models(self):
        from onnxscript.tests.models import if_statement

        self.validate_save(if_statement)

    def test_docstring(self):
        @script()
        def sumprod(x: FLOAT["N"], N: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
            """Combines ReduceSum, ReduceProd."""
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod

        proto = sumprod.to_function_proto()
        self.assertEqual(proto.doc_string.strip(), "Combines ReduceSum, ReduceProd.")

    def test_signal(self):
        from onnxscript.tests.models import signal_dft

        # shape_inference crashes on stft.
        self.validate_save(signal_dft, shape_inference=False)

    def test_multi(self):
        from onnxscript.tests.models import multi

        self.validate_save(multi, shape_inference=False)

    def test_dropout(self):
        from onnxscript.tests.models import dropout

        self.validate_save(dropout, shape_inference=False)

    def test_attrref(self):
        from onnxscript.tests.models import attrref

        self.validate_save(attrref, shape_inference=False)

    def test_renaming(self):
        from onnxscript.tests.models import renaming

        self.validate_save(renaming, shape_inference=False)

    @unittest.skipIf(True, reason="TypeError: val must be numeric not <class 'NoneType'>")
    def test_opt_output(self):
        from onnxscript.tests.models import opt_output

        self.validate_save(opt_output, shape_inference=False)

    def test_opt_input(self):
        from onnxscript.tests.models import opt_input

        self.validate_save(opt_input, shape_inference=False)

    @unittest.skipIf(
        True, reason="ValueError: A function with attributes " "cannot be exported as a model."
    )
    def test_onnxfns2(self):
        from onnxscript.tests.models import onnxfns2

        self.validate_save(onnxfns2, shape_inference=False)

    def test_none_as_input(self):
        """Test that use of None as an actual parameter is accepted."""

        @script()
        def clipmax(x: FLOAT, max: FLOAT):  # noqa: F821
            return op.Clip(x, None, max)

        self.validate_save(clipmax)

    def test_type_double(self):
        from onnxscript.tests.models import type_double

        fcts = self.validate_save(type_double, check_ort=False)
        f = fcts["double_abs"]
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 11)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        f = fcts["double_cast"]
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 7)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        f = fcts["double_abs_subgraph"]
        self.assertEqual(f.graph.input[0].type.tensor_type.elem_type, 11)
        self.assertEqual(f.graph.output[0].type.tensor_type.elem_type, 11)
        g = f.graph.node[3].attribute[0].g
        self.assertEqual(g.output[0].type.tensor_type.elem_type, 11)
        g = f.graph.node[3].attribute[1].g
        self.assertEqual(g.output[0].type.tensor_type.elem_type, 11)
        self.validate_save(type_double, check_ort=True)

    def test_cast_like(self):
        from onnxscript.tests.models import cast_like

        self.validate_expansion(cast_like)

    def test_identity(self):
        from onnxscript.tests.models import identity

        self.validate_expansion(identity)

    def test_opset_import(self):
        from onnxscript.tests.models import different_opset

        fcts = self.validate_save(different_opset, shape_inference=False)
        s16 = str(fcts["shape_A"])
        s14 = str(fcts["shape_B"])
        sdef = str(fcts["inc_any"])
        self.assertIn("version: 16", s16)
        self.assertNotIn("version: 14", s16)
        self.assertIn("version: 14", s14)
        self.assertNotIn("version: 16", s14)
        self.assertIn("version: 16", sdef)
        self.assertNotIn("version: 14", sdef)
        self.assertNotIn("version: 15", sdef)

    def test_sequences(self):
        from onnxscript.tests.models import sequences

        test_functions = self.validate_save(sequences, check_ort=True)

        f = test_functions["make_sequence_tensor"]

        A = np.array([[0, 1, 2]], dtype=np.float32)
        eager_mode = sequences.make_sequence_tensor(A)
        self.assertEqual(eager_mode.shape, (5, 3))
        self.assertEqual(eager_mode.dtype, np.float32)

        session = onnxruntime.InferenceSession(f.SerializeToString())
        result = session.run(None, {"A": A})[0]
        assert_almost_equal(eager_mode, result)

        f = test_functions["make_sequence_tensor_accumulated"]

        A = np.array([[0, 1, 2]], dtype=np.float32)
        eager_mode = sequences.make_sequence_tensor_accumulated(A)
        self.assertEqual(eager_mode.shape, (5, 3))
        self.assertEqual(eager_mode.dtype, np.float32)

        session = onnxruntime.InferenceSession(f.SerializeToString())
        result = session.run(None, {"A": A})[0]
        assert_almost_equal(eager_mode, result)

    def test_loops_break(self):
        from onnxscript.tests.models import loops_break

        test_functions = self.validate_save(loops_break, check_ort=True)
        self.assertIn("loop1", test_functions)
        for name in ("loop1", "loop_range_cond"):
            with self.subTest(fct=name):
                f = test_functions[name]
                self.assertIn('op_type: "Loop"', str(f))
        onx = test_functions["loop_range_cond"]
        session = onnxruntime.InferenceSession(onx.SerializeToString())
        x = np.array([0, 1, 2], dtype=np.float32)
        y = session.run(None, {"A": x})[0]
        self.assertEqual(loops_break.loop_range_cond(x).tolist(), [0.0, 46.0, 92.0])
        self.assertEqual(y.tolist(), [0.0, 46.0, 92.0])
        x = np.array([0, 1, -2], dtype=np.float32)
        y = session.run(None, {"A": x})[0]
        self.assertEqual(loops_break.loop_range_cond(x).tolist(), [0, 11, -22])
        self.assertEqual(y.tolist(), [0, 11, -22])

    def test_loops_while(self):
        from onnxscript.tests.models import loops_while

        test_functions = self.validate_save(loops_while, check_ort=True)
        self.assertIn("loop1", test_functions)
        for name in ("loop1", "loop_range_cond_only"):
            with self.subTest(fct=name):
                f = test_functions[name]
                self.assertIn('op_type: "Loop"', str(f))
        onx = test_functions["loop_range_cond_only"]
        session = onnxruntime.InferenceSession(onx.SerializeToString())
        x = np.array([0, 1, -2], dtype=np.float32)
        y = session.run(None, {"A": x})[0]
        self.assertEqual(y.tolist(), [0, 10, -20])
        res = loops_while.loop_range_cond_only(x)
        self.assertEqual(res.tolist(), [0, 10, -20])

    @unittest.skipIf(
        sys.version_info[:2] < (3, 8), reason="Notation [...] not supported in python 3.7."
    )
    def test_getitem(self):
        from onnxscript.tests.models import getitem

        if sys.version_info[:2] >= (3, 8):
            skip_check_ort = None
        else:
            # negative indices are not supported in python 3.7
            # one constant is evaluated as float
            skip_check_ort = ["getitem_i_slice_neg", "getitem_i_slice_step"]
        test_functions = self.validate_save(
            getitem, check_ort=True, skip_check_ort=skip_check_ort
        )

        # eager mode is disabled because A[np.array([0]): np.array([1])] is not a valid
        # expression.
        A = np.array([0, 1, 2])
        i = np.array([0])
        try:
            A[i : i + 1]
            eager = True
        except Exception:
            # TypeError: only integer scalar arrays can be converted to a scalar index
            eager = False

        def check_function(x, name, expected, eager=True):
            if skip_check_ort is not None and name in skip_check_ort:
                return
            with self.subTest(name=name):
                onx = test_functions[name]
                session = onnxruntime.InferenceSession(onx.SerializeToString())
                try:
                    y = session.run(None, {"A": x})[0]
                except Exception as e:
                    raise AssertionError(
                        f"Unable to run ONNX for function {name!r} " f"due to {e!r}\n{onx}."
                    ) from e
                self.assertEqual(y.tolist(), expected)
                f = getattr(getitem, name)
                if eager:
                    self.assertEqual(f(x).tolist(), expected)

        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float32)

        check_function(x, "getitem_i", [0.0, 1.0, 2.0])
        check_function(x, "getitem_i_last", [9.0, 10.0, 11.0])
        check_function(x, "getitem_i_expr", [1.0, 2.0, 3.0])
        check_function(x, "getitem_i_slice", [[3.0, 4.0, 5.0]])
        check_function(x, "getitem_i_slice_left", [[3, 4, 5], [6, 7, 8], [9, 10, 11]])
        check_function(x, "getitem_i_slice_right", [[0, 1, 2], [3, 4, 5]])
        check_function(x, "getitem_i_slice_neg", [[3, 4, 5], [6, 7, 8]])
        check_function(x, "getitem_i_slice_step", [[6.0, 7.0, 8.0], [3.0, 4.0, 5.0]])
        # TODO: force eager to True when the following issue is resolved.
        check_function(x, "getitem_i_var", [[3.0, 4.0, 5.0]], eager=eager)
        check_function(x, "getitem_i_tuple", [[0], [3]])
        check_function(x, "getitem_i_mixed_tuple", [0, 3])
        check_function(x, "getitem_column", [1.0, 4.0, 7.0, 10.0])
        check_function(x, "getitem_index_int0_1", [3, 4, 5], eager=eager)
        check_function(x, "getitem_index_int0", [0, 1, 2], eager=eager)
        check_function(x, "getitem_rev", x[:0:-1].tolist())
        check_function(x, "getitem_rev0", x[0, :0:-1].tolist())

    @unittest.skipIf(
        sys.version_info[:2] < (3, 9), reason="Notation [...] not supported in python 3.8."
    )
    def test_getitem39(self):
        from onnxscript.tests.models import getitem39

        test_functions = self.validate_save(getitem39, check_ort=True)

        # eager mode is disabled because A[np.array([0]): np.array([1])] is not a valid
        # expression.
        A = np.array([0, 1, 2])
        i = np.array([0])
        try:
            A[i : i + 1]
            eager = True
        except Exception:
            # TypeError: only integer scalar arrays can be converted to a scalar index
            eager = False

        def check_function(x, name, expected, eager=True):
            with self.subTest(name=name):
                onx = test_functions[name]
                session = onnxruntime.InferenceSession(onx.SerializeToString())
                try:
                    y = session.run(None, {"A": x})[0]
                except Exception as e:
                    raise AssertionError(
                        f"Unable to run ONNX for function {name!r} " f"due to {e!r}\n{onx}."
                    ) from e
                self.assertEqual(y.tolist(), expected)
                f = getattr(getitem39, name)
                if eager:
                    self.assertEqual(f(x).tolist(), expected)

        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float32)

        check_function(x, "getitem_index_int", [2.0], eager=eager)
        check_function(x, "getitem_index_int2", [2.0], eager=eager)

    def check_failure(self, f, msg):
        source = textwrap.dedent(inspect.getsource(f))
        global_names = globals().copy()
        top_level_ast = ast.parse(source)
        f_ast = top_level_ast.body[0]
        cvt = converter.Converter(
            opset=op, global_names=global_names, source=source, default_opset=op
        )
        try:
            cvt.top_level_stmt(f_ast)
        except converter.TranslationError as e:
            if msg not in str(e):
                raise AssertionError(f"Unable to find {msg!r} in {e!r} in\n{source}") from e
            return
        raise AssertionError("No raised exception.")

    @unittest.skipIf(
        sys.version_info[:2] < (3, 8), reason="Notation [...] not supported in python 3.7."
    )
    def test_getitem_failure(self):
        def f1(A: FLOAT[...]) -> FLOAT[...]:
            zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
            index = zero, zero + 1
            r = A[index]
            return r

        ast_name = "_ast" if sys.version_info[:2] < (3, 9) else "ast"
        self.check_failure(f1, f"Left term must be a tuple not <class '{ast_name}.Name'>")

        def f2(A: FLOAT[...]) -> FLOAT[...]:
            return A[::-1]

        ast_name = "_ast" if sys.version_info[:2] < (3, 9) else "ast"
        self.check_failure(f2, "`?::-1` cannot be expressed with ONNX")

    def check_run(self, onnxfn, inputs, expected_output):
        # Test by converting to model and running with ORT
        model = onnxfn.to_model_proto()
        session = onnxruntime.InferenceSession(model.SerializeToString())
        input_names = [x.name for x in model.graph.input]
        input_dict = {x: value for (x, value) in zip(input_names, inputs)}
        output = session.run(None, input_dict)[0]
        np.testing.assert_equal(output, expected_output)

        # Test running model in eager mode
        output = onnxfn(*inputs)
        if isinstance(output, tensor.Tensor):
            # unwrap Tensor wrapper
            output = output.value
        np.testing.assert_equal(output, expected_output)

    def test_graph_attr_scan(self):
        from onnxscript.tests.models.graph_attr import cumulative_sum

        inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int64)]
        expected_output = np.array([1, 3, 6, 10, 15], dtype=np.int64)
        self.check_run(cumulative_sum, inputs, expected_output)

    def test_graph_attr_loop(self):
        from onnxscript.tests.models.graph_attr import sum_to

        inputs = [np.array(6, dtype=np.int64)]
        expected_output = np.array([0, 1, 3, 6, 10, 15], dtype=np.int64)
        self.check_run(sum_to, inputs, expected_output)

    def test_graph_attr_loop_error(self):
        from onnxscript.tests.models.graph_attr import sum_to_error

        input = np.array(6, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "@graph"):
            sum_to_error(input)

    def test_loop_outer_scope(self):
        from onnxscript.tests.models.graph_attr import loop_add

        input_x = np.array([1, 2, 3], dtype=np.int64)
        input_m = np.array(3, dtype=np.int64)
        inputs = [input_x, input_m]
        expected_output = np.array([4, 8, 12], dtype=np.int64)
        self.check_run(loop_add, inputs, expected_output)

    def test_outer_scope_redefinition(self):
        """Test that outer scope variables used in a function are not redefined."""
        with self.assertRaisesRegex(Exception, "Outer scope variable"):

            @script()
            def redefine(X):
                Temp = op.Neg(X)

                @graph()
                def inner():
                    return op.Add(X, Temp)

                Temp = op.Abs(X)
                return op.DummyOp(body=inner)

    def test_attr(self):
        from onnxscript.tests.functions import attr_test

        self.validate_run(attr_test)

    def test_renaming_parameter(self):
        @script()
        def model_script(x: FLOAT[100]) -> FLOAT[100]:
            x = op.Add(x, x)
            return x

        proto = model_script.to_model_proto()
        onnx.shape_inference.infer_shapes(proto)

    def test_cast_like_attr(self):
        @script(default_opset=op)
        def inc_alpha(A: FLOAT[...], alpha: int) -> FLOAT[...]:
            return A + alpha

        @script()
        def inc_alpha_expanded(A: FLOAT[...], alpha: int) -> FLOAT[...]:
            return A + op.CastLike(alpha, A)

        self.assertSame(inc_alpha, inc_alpha_expanded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
