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
import typing
import unittest
import warnings

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
)

import onnxscript
import onnxscript.testing
from onnxscript import BOOL, FLOAT, INT64, converter, graph, script, tensor
from onnxscript.onnx_opset import opset11 as op11
from onnxscript.onnx_opset import opset15 as op
from onnxscript.tests.common import onnx_script_test_case, testutils

TEST_INPUT_DIR = pathlib.Path(__file__).parent / "tests" / "models"
TEST_OUTPUT_DIR = TEST_INPUT_DIR / "testoutputs"


def create_cpu_inference_session(model_bytes: bytes) -> ort.InferenceSession:
    return ort.InferenceSession(model_bytes, providers=("CPUExecutionProvider",))


class TestConverter(testutils.TestBase):
    def validate(self, script):
        if isinstance(script, types.ModuleType):
            fnlist = [
                f for f in script.__dict__.values() if isinstance(f, onnxscript.OnnxFunction)
            ]
        elif isinstance(script, onnxscript.OnnxFunction):
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
            fnlist = [
                f for f in script.__dict__.values() if isinstance(f, onnxscript.OnnxFunction)
            ]
        elif isinstance(script, onnxscript.OnnxFunction):
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
                        fi.write(onnx.helper.printable_graph(model.graph))
                        for fct in model.functions:
                            fi.write("\n-------------------------\n")
                            fi.write(onnx.helper.printable_graph(fct))
                if check_ort and (skip_check_ort is None or f.name not in skip_check_ort):
                    try:
                        create_cpu_inference_session(model.SerializeToString())
                    except (Fail, InvalidGraph, InvalidArgument) as e:
                        raise AssertionError(
                            f"onnxruntime cannot load function {f.name}\n--\n{model}"
                        ) from e
                if shape_inference:
                    model = onnx.shape_inference.infer_shapes(model)
                if save_text:
                    with open(os.path.join(TEST_OUTPUT_DIR, f"{f.name}.shape.txt"), "w") as fi:
                        fi.write(onnx.helper.printable_graph(model.graph))
                        for fct in model.functions:
                            f.write("\n-------------------------\n")
                            f.write(onnx.helper.printable_graph(fct))
                try:
                    onnx.checker.check_model(model)
                except onnx.checker.ValidationError as e:
                    if "Field 'shape' of 'type' is required but missing" in str(
                        e
                    ) or "Field 'shape' of type is required but missing" in str(e):
                        # input or output shapes are missing because the function
                        # was defined with FLOAT[...].
                        warnings.warn(str(e), stacklevel=1)
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
                        onnxscript.testing.assert_isomorphic(f, f_expanded)

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
        session = create_cpu_inference_session(onx.SerializeToString())
        y = session.run(None, {"X": x})[0]
        self.assertEqual(y.tolist(), [0.0, 0.5, -0.5])
        # numpy fmod and operator % disagree on this example
        res = eager_op.eager_op(x)
        self.assertEqual(res.tolist(), [0.0, 0.5, -0.5])

        onx = test_functions["eager_abs"]
        session = create_cpu_inference_session(onx.SerializeToString())
        y = session.run(None, {"X": x})[0]
        self.assertEqual(y.tolist(), [1, 6, 3])
        res = eager_op.eager_abs(x)
        self.assertEqual(res.tolist(), [1, 6, 3])

    def test_error_undefined(self):
        with self.assertRaises(ValueError) as e:

            @script()
            def square(x):
                return op.Mul(undefined, x)  # noqa: F821

        self.assertIn("Unbound name: undefined", str(e.exception))

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
        self.assertEqual(x_value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(y_value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(output_value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

        # Or, use input_types and output_types, as below, for the more general case.
        model = cast_add.to_model_proto(
            input_types=[FLOAT["N"], INT64["N"]], output_types=[FLOAT["N"]]
        )
        x_value_info = model.graph.input[0]
        y_value_info = model.graph.input[1]
        output_value_info = model.graph.output[0]
        self.assertEqual(x_value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(y_value_info.type.tensor_type.elem_type, onnx.TensorProto.INT64)
        self.assertEqual(output_value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

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

    @unittest.skip(reason="TypeError: val must be numeric not <class 'NoneType'>")
    def test_opt_output(self):
        from onnxscript.tests.models import opt_output

        self.validate_save(opt_output, shape_inference=False)

    def test_opt_input(self):
        from onnxscript.tests.models import opt_input

        self.validate_save(opt_input, shape_inference=False)

    @unittest.skip("A function with attributes cannot be exported as a model.")
    def test_onnxfns2(self):
        from onnxscript.tests.models import onnxfns2

        self.validate_save(onnxfns2, shape_inference=False)

    def test_none_as_input(self):
        """Test that use of None as an actual parameter is accepted."""

        @script()
        def clipmax(x: FLOAT, max: FLOAT):
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

        session = create_cpu_inference_session(f.SerializeToString())
        result = session.run(None, {"A": A})[0]
        np.testing.assert_almost_equal(eager_mode, result)

        f = test_functions["make_sequence_tensor_accumulated"]

        A = np.array([[0, 1, 2]], dtype=np.float32)
        eager_mode = sequences.make_sequence_tensor_accumulated(A)
        self.assertEqual(eager_mode.shape, (5, 3))
        self.assertEqual(eager_mode.dtype, np.float32)

        session = create_cpu_inference_session(f.SerializeToString())
        result = session.run(None, {"A": A})[0]
        np.testing.assert_almost_equal(eager_mode, result)

    def test_loops_break(self):
        from onnxscript.tests.models import loops_break

        test_functions = self.validate_save(loops_break, check_ort=True)
        self.assertIn("loop1", test_functions)
        for name in ("loop1", "loop_range_cond"):
            with self.subTest(fct=name):
                f = test_functions[name]
                self.assertIn('op_type: "Loop"', str(f))
        onx = test_functions["loop_range_cond"]
        session = create_cpu_inference_session(onx.SerializeToString())
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
        session = create_cpu_inference_session(onx.SerializeToString())
        x = np.array([0, 1, -2], dtype=np.float32)
        y = session.run(None, {"A": x})[0]
        self.assertEqual(y.tolist(), [0, 10, -20])
        res = loops_while.loop_range_cond_only(x)
        self.assertEqual(res.tolist(), [0, 10, -20])

    def test_getitem(self):
        from onnxscript.tests.models import getitem

        self.validate_save(getitem, check_ort=True, skip_check_ort=None)
        self.validate_run(getitem)

    def check_failure(self, f, msg):
        source = textwrap.dedent(inspect.getsource(f))
        global_names = globals().copy()
        top_level_ast = ast.parse(source)
        f_ast = top_level_ast.body[0]
        cvt = converter.Converter(
            opset=op, global_names=global_names, source=source, default_opset=op
        )
        try:
            cvt.translate_function_def(f_ast)
        except converter.TranslationError as e:
            if msg not in str(e):
                raise AssertionError(f"Unable to find {msg!r} in {e!r} in\n{source}") from e
            return
        raise AssertionError("No raised exception.")

    def test_getitem_failure(self):
        def f1(A: FLOAT[...]) -> FLOAT[...]:
            zero = op.Constant(
                value=onnx.helper.make_tensor("zero", onnx.TensorProto.INT64, [1], [0])
            )
            index = zero, zero + 1
            r = A[index]
            return r

        ast_name = "_ast" if sys.version_info[:2] < (3, 9) else "ast"
        self.check_failure(f1, f"Left term must be a tuple not '<class '{ast_name}.Name'>'")

    def check_run(self, onnxfn, inputs, expected_output):
        # Test by converting to model and running with ORT
        model = onnxfn.to_model_proto()
        session = create_cpu_inference_session(model.SerializeToString())
        input_names = [x.name for x in model.graph.input]
        input_dict = dict(zip(input_names, inputs))
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

        onnxscript.testing.assert_isomorphic_function(inc_alpha, inc_alpha_expanded)

    def test_none_attribute(self):
        """Test converter handles a None value specified as an attribute value.
        Use Squeeze from opset 11 as an example, since it has an attribute named "axes"
        that is optional.
        """

        @script()
        def explicit_none(X):
            return op11.Squeeze(X, axes=None)

        @script()
        def implicit_none(X):
            return op11.Squeeze(X)

        onnxscript.testing.assert_isomorphic_function(explicit_none, implicit_none)

    def test_input_and_attr_classification(self):
        """Test that inputs and attributes are classified correctly, when positional and keyword arguments are used."""

        @script()
        def positional(X, shape):
            return op.Expand(X, shape)

        @script()
        def keyword(X, shape):
            return op.Expand(shape=shape, input=X)

        onnxscript.testing.assert_isomorphic_function(positional, keyword)

    def test_none_as_input_for_op_with_no_schema(self):
        """Test conversion of None as an input value in a call to an op with no known schema."""

        @script()
        def none_as_input(X):
            return op.UnknownOp(X, None, X)

        # None should be translated into an empty string in NodeProto's input list
        node = none_as_input.to_function_proto().node[0]
        self.assertEqual(node.input[1], "")

    def test_unique_names_in_subscript_expr(self):
        @script()
        def nested_index_expr(X):
            return op.Add(op.Shape(X)[-1], 1)

        nodes = nested_index_expr.to_function_proto().node
        assigned_names = [n.output[0] for n in nodes]
        self.assertEqual(len(assigned_names), len(set(assigned_names)))

    def test_no_duplicate_output_name(self):
        """Test that the converter does not generate duplicate output names."""

        @script()
        def duplicate_output(X):
            Y = op.Neg(X)
            return Y, Y

        # The converter should generate distinct names for the two outputs
        outputs = duplicate_output.to_function_proto().output
        self.assertNotEqual(outputs[0], outputs[1])

    def test_bool_attr_promotion(self):
        @script()
        def if_then_else(flag: bool, Y, Z):
            return op.Where(flag, Y, Z)

        @script()
        def if_then_else_expanded(flag: bool, Y, Z):
            tmp1 = op.Constant(value_int=flag)
            tmp2 = op.Cast(tmp1, to=BOOL.dtype)
            return op.Where(tmp2, Y, Z)

        onnxscript.testing.assert_isomorphic(if_then_else, if_then_else_expanded)

    def test_bool_list_attr_promotion(self):
        @script()
        def if_then_else(flag: typing.List[bool], Y, Z):
            return op.Where(flag, Y, Z)

        @script()
        def if_then_else_expanded(flag: typing.List[bool], Y, Z):
            tmp1 = op.Constant(value_ints=flag)
            tmp2 = op.Cast(tmp1, to=9)
            return op.Where(tmp2, Y, Z)

        onnxscript.testing.assert_isomorphic(if_then_else, if_then_else_expanded)

    def test_empty_ints_attribute(self):
        @script()
        def empty_ints():
            return op.Constant(value_ints=[])

        expected = np.array([], dtype=np.int64)
        self.check_run(empty_ints, [], expected)

    def test_empty_floats_attribute(self):
        @script()
        def empty_floats():
            return op.Constant(value_floats=[])

        expected = np.array([], dtype=np.float32)
        self.check_run(empty_floats, [], expected)

    def test_int_as_tensor_attribute(self):
        @script()
        def int_as_tensor():
            return op.Constant(value=17)

        expected = np.array(17, dtype=np.int64)
        self.check_run(int_as_tensor, [], expected)

    def test_int_list_as_tensor_attribute(self):
        @script()
        def int_list_as_tensor():
            return op.Constant(value=[13, 17])

        expected = np.array([13, 17], dtype=np.int64).reshape((2,))
        self.check_run(int_list_as_tensor, [], expected)

    def test_float_as_tensor_attribute(self):
        @script()
        def float_as_tensor():
            return op.Constant(value=17.0)

        expected = np.array([17], dtype=np.float32).reshape(())
        self.check_run(float_as_tensor, [], expected)

    def test_float_list_as_tensor_attribute(self):
        @script()
        def float_list_as_tensor():
            return op.Constant(value=[13.0, 17.0])

        expected = np.array([13, 17], dtype=np.float32).reshape((2,))
        self.check_run(float_list_as_tensor, [], expected)

    def test_loop_inside_if(self):
        @script(default_opset=op)
        def sum(n: INT64) -> INT64:
            sum = op.Constant(value=0)
            if n > 0:
                for i in range(n):
                    sum = sum + i
            return sum

        self.check_run(sum, [np.array(5, dtype=np.int64)], np.array(10, dtype=np.int64))
        self.check_run(sum, [np.array(-5, dtype=np.int64)], np.array(0, dtype=np.int64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
