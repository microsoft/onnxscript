# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx
import onnx.reference
import onnxruntime as ort
import parameterized

import onnxscript.evaluator
import onnxscript.tensor
from onnxscript import BOOL, DOUBLE, FLOAT, INT64, graph, script
from onnxscript import opset17 as op


@parameterized.parameterized_class(
    (
        "name",
        "evaluator",
    ),
    [
        (
            "reference_runtime",
            onnxscript.evaluator.OnnxReferenceRuntimeEvaluator(),
        ),
        (
            "onnxruntime",
            onnxscript.evaluator.ORTEvaluator(),
        ),
    ],
)
class EagerModeTest(unittest.TestCase):
    evaluator: onnxscript.evaluator.Evaluator

    def setUp(self):
        self.default_evaluator = onnxscript.evaluator.default()
        onnxscript.evaluator.set_default(self.evaluator)

    def tearDown(self):
        onnxscript.evaluator.set_default(self.default_evaluator)

    def assert_compiled_eager(
        self,
        function,
        inputs,
        expected,
        *,
        output_type=np.ndarray,
        model_kwargs=None,
        kwargs=None,
        ort_model=False,
    ):
        """Compare independent model execution and eager execution, including representation."""
        model = function.to_model_proto(ir_version=9, **(model_kwargs or {}))
        onnx.checker.check_model(model)
        self.assertEqual(len(model.graph.input), len(inputs) + len(kwargs or {}))
        feeds = {}
        for value_info, value in zip(model.graph.input, inputs):
            feeds[value_info.name] = (
                value.value
                if isinstance(value, onnxscript.tensor.Tensor)
                else np.asarray(value, dtype=np.int64 if type(value) is int else None)
            )
        for name, value in (kwargs or {}).items():
            feeds[name] = (
                value.value
                if isinstance(value, onnxscript.tensor.Tensor)
                else np.asarray(value)
            )
        if ort_model or isinstance(self.evaluator, onnxscript.evaluator.ORTEvaluator):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
        else:
            session = onnx.reference.ReferenceEvaluator(model)
        compiled = session.run(None, feeds)
        eager = function(*inputs, **(kwargs or {}))
        self.assertIsInstance(compiled, list)
        if isinstance(expected, tuple):
            self.assertIsInstance(eager, tuple)
            eager_outputs = eager
            expected_outputs = expected
        else:
            self.assertIsInstance(eager, output_type)
            eager_outputs = (eager,)
            expected_outputs = (expected,)
        self.assertEqual(len(compiled), len(expected_outputs))
        self.assertEqual(len(eager_outputs), len(expected_outputs))
        for model_output, eager_output, expected_output in zip(
            compiled, eager_outputs, expected_outputs
        ):
            self.assertIsInstance(model_output, np.ndarray)
            self.assertIsInstance(eager_output, output_type)
            if isinstance(eager_output, onnxscript.tensor.Tensor):
                eager_output = eager_output.value
            for output in (model_output, eager_output):
                self.assertEqual(output.dtype, expected_output.dtype)
                self.assertEqual(output.shape, expected_output.shape)
                np.testing.assert_array_equal(output, expected_output)
            np.testing.assert_array_equal(model_output, eager_output)

    def test_optional_operator_inputs(self):
        @script()
        def optional_clip(x: FLOAT["N"]):  # noqa: F821
            unchanged = op.Clip(x)
            explicit_none = op.Clip(x, None, None)
            upper = op.Clip(x, None, 1.0)
            lower = op.Clip(input=x, min=-1.0)
            return unchanged, explicit_none, upper, lower

        data = np.array([-2, 0, 2], dtype=np.float32)
        self.assert_compiled_eager(
            optional_clip,
            [data],
            (data, data, np.minimum(data, 1), np.maximum(data, -1)),
            model_kwargs={"output_types": [FLOAT["N"]] * 4},
        )
        clips = [
            node for node in optional_clip.to_function_proto().node if node.op_type == "Clip"
        ]
        self.assertEqual(list(clips[0].input), ["x"])
        self.assertEqual(list(clips[1].input), ["x", "", ""])
        self.assertEqual(clips[2].input[1], "")

    def test_function_default_and_keyword_arguments(self):
        @script()
        def defaults(x: FLOAT["N"], y: FLOAT["N"]):  # noqa: F821
            default_alpha = add_with_alpha(x, y)
            explicit_alpha = add_with_alpha(alpha=2.0, other=y, this=x)
            return default_alpha, explicit_alpha

        x = np.array([1, 2], dtype=np.float32)
        y = np.array([3, 4], dtype=np.float32)
        self.assert_compiled_eager(
            defaults,
            [x],
            (x + y, x + 2 * y),
            kwargs={"y": y},
            model_kwargs={"output_types": [FLOAT["N"]] * 2},
            # ReferenceEvaluator does not resolve omitted local-function attribute defaults.
            ort_model=True,
        )

    def test_scalar_literal_promotion(self):
        @script(default_opset=op)
        def promote(x):
            return 2 * x + 1.0

        for dtype, onnx_type in (
            (np.float32, FLOAT),
            (np.float64, DOUBLE),
            (np.int64, INT64),
        ):
            with self.subTest(dtype=dtype):
                data = np.array([-2, 0, 3], dtype=dtype)
                self.assert_compiled_eager(
                    promote,
                    [data],
                    np.array([-3, 1, 7], dtype=dtype),
                    model_kwargs={"io_types": onnx_type["N"]},
                )

    def test_scalar_input_wrapping(self):
        @script()
        def identity(x):
            return op.Identity(x)

        for value, onnx_type in ((3, INT64), (2.5, DOUBLE), (True, BOOL)):
            with self.subTest(value=value):
                self.assert_compiled_eager(
                    identity,
                    [value],
                    np.array(
                        value,
                        dtype={INT64: np.int64, DOUBLE: np.float64, BOOL: np.bool_}[onnx_type],
                    ),
                    output_type=onnxscript.tensor.Tensor,
                    model_kwargs={"io_types": onnx_type},
                )

    def test_array_tensor_output_wrapping(self):
        @script(default_opset=op)
        def pair(x, y):
            return x + y, x - y

        for data in (np.array(2, dtype=np.float32), np.array([1, 2], dtype=np.float32)):
            for input_kinds in ("arrays", "tensors", "mixed"):
                with self.subTest(shape=data.shape, input_kinds=input_kinds):
                    wrapped = onnxscript.tensor.Tensor(data)
                    inputs = {
                        "arrays": [data, data],
                        "tensors": [wrapped, wrapped],
                        "mixed": [wrapped],
                    }[input_kinds]
                    self.assert_compiled_eager(
                        pair,
                        inputs,
                        (data + data, data - data),
                        kwargs={"y": data} if input_kinds == "mixed" else None,
                        output_type=(
                            onnxscript.tensor.Tensor
                            if input_kinds == "tensors"
                            else np.ndarray
                        ),
                        model_kwargs={"io_types": FLOAT[data.shape]},
                    )

    def test_supported_control_flow(self):
        @script(default_opset=op)
        def conditional(x: FLOAT["N"], condition: BOOL) -> FLOAT["N"]:  # noqa: F821
            if condition:
                result = x + 1
            else:
                result = x - 1
            return result

        @script(default_opset=op)
        def counted(x: FLOAT["N"], count: INT64) -> FLOAT["N"]:  # noqa: F821
            result = x
            for i in range(count):
                result = result + op.CastLike(i, x)
            return result

        @script(default_opset=op)
        def conditional_loop(x: FLOAT["N"], count: INT64) -> FLOAT["N"]:  # noqa: F821
            result = x
            i = op.Constant(value_int=0)
            condition = i < count
            while condition:
                result = result + 1
                i = i + 1
                condition = i < count
            return result

        data = np.array([-2, 3], dtype=np.float32)
        for condition in (False, True):
            with self.subTest(condition=condition):
                self.assert_compiled_eager(
                    conditional,
                    [data, np.array(condition)],
                    data + (1 if condition else -1),
                )
        for count in (0, 3):
            with self.subTest(count=count):
                self.assert_compiled_eager(
                    counted,
                    [data, np.array(count, dtype=np.int64)],
                    data + sum(range(count)),
                    # Reference Loop treats an omitted initial condition as false.
                    # Still compare both eager evaluators against the independent ORT model.
                    ort_model=True,
                )
                self.assert_compiled_eager(
                    conditional_loop,
                    [data, np.array(count, dtype=np.int64)],
                    data + count,
                )

    def test_comparison_and_slices(self):
        @script(default_opset=op)
        def compare(x: FLOAT["N"]) -> BOOL["N"]:  # noqa: F821
            return (x > 0) & (x < 3)

        @script(default_opset=op)
        def slices(x: FLOAT["N"], start: INT64, stop: INT64, step: INT64):  # noqa: F821
            forward = x[1:]
            reverse = x[::-1]
            dynamic = x[start:stop:step]
            return forward, reverse, dynamic

        data = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assert_compiled_eager(compare, [data], (data > 0) & (data < 3))
        for start, stop, step in ((0, 5, 2), (4, 0, -2)):
            with self.subTest(step=step):
                self.assert_compiled_eager(
                    slices,
                    [data, *[np.array(v, dtype=np.int64) for v in (start, stop, step)]],
                    (data[1:], data[::-1], data[start:stop:step]),
                    model_kwargs={"output_types": [FLOAT["M"]] * 3},
                )

    def test_nested_graph_with_capture(self):
        @script()
        def scan(x: FLOAT["N"], bias: FLOAT) -> FLOAT["N"]:  # noqa: F821
            @graph()
            def body(state: FLOAT, element: FLOAT) -> tuple[FLOAT, FLOAT]:
                updated = op.Add(state, element)
                output = op.Add(updated, bias)
                return updated, output

            zero = op.Constant(value_float=0.0)
            _, result = op.Scan(zero, x, body=body, num_scan_inputs=1)
            return result

        data = np.array([1, 2, 3], dtype=np.float32)
        self.assert_compiled_eager(
            scan,
            [data, np.array(0.5, dtype=np.float32)],
            np.cumsum(data) + np.float32(0.5),
        )

    def test_sequence_input(self):
        @script()
        def Concat(seq):
            return op.ConcatFromSequence(seq, axis=0)

        np_array = np.array([1, 2, 3], dtype=np.float32)
        output1 = Concat([np_array, np_array])
        self.assertIsInstance(output1, np.ndarray)

        os_tensor = onnxscript.tensor.Tensor(np_array)
        output2 = Concat([os_tensor, os_tensor])
        self.assertIsInstance(output2, onnxscript.tensor.Tensor)


@script()
def add_with_alpha(this, other, alpha: float = 1.0):
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)
    return op.Add(this, other)


@parameterized.parameterized_class(
    (
        "name",
        "evaluator",
    ),
    [
        (
            "reference_runtime",
            onnxscript.evaluator.OnnxReferenceRuntimeEvaluator(),
        ),
        (
            "onnxruntime",
            onnxscript.evaluator.ORTEvaluator(),
        ),
    ],
)
class TestEagerModeArguments(unittest.TestCase):
    evaluator: onnxscript.evaluator.Evaluator

    def setUp(self):
        self.default_evaluator = onnxscript.evaluator.default()
        onnxscript.evaluator.set_default(self.evaluator)

    def tearDown(self):
        onnxscript.evaluator.set_default(self.default_evaluator)

    def test_op_some_input_by_kwargs(self):
        self.assertEqual(op.Add(1, B=2), 3)

    def test_op_all_input_by_kwargs(self):
        self.assertEqual(op.Add(A=1, B=2), 3)

    def test_op_attribute_by_positional_args(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        axes = np.array([0], dtype=np.int64)
        self.assertEqual(op.ReduceSum(data, axes, keepdims=True), 21)

    def test_op_input_and_attribute_by_kwargs_out_of_order(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        axes = np.array([0], dtype=np.int64)
        self.assertEqual(op.ReduceSum(keepdims=True, axes=axes, data=data), 21)

    def test_function_some_input_by_kwargs(self):
        self.assertEqual(add_with_alpha(1.0, other=2.0), 3.0)

    def test_function_all_input_by_kwargs(self):
        self.assertEqual(add_with_alpha(this=1.0, other=2.0), 3.0)

    def test_function_attribute_by_positional_args(self):
        self.assertEqual(add_with_alpha(1.0, 2.0, 3.0), 7.0)

    def test_function_input_and_attribute_by_kwargs_out_of_order(self):
        self.assertEqual(add_with_alpha(alpha=3.0, other=2.0, this=1.0), 7.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
