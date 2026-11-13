# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import asyncio
import concurrent.futures
import contextvars
import functools
import threading
import unittest

import numpy as np

from onnxscript import evaluator as public_evaluator
from onnxscript import graph, script, tensor
from onnxscript._internal import evaluator
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT


@script()
def _identity(x):
    return op.Identity(x)


class _SelectionEvaluator:
    def eval_op(self, op, args, kwargs):
        return self

    def eval_function(self, function, args, kwargs):
        return self


class EvaluatorContextTest(unittest.TestCase):
    def assert_selected(self, selected):
        self.assertIs(public_evaluator.default(), selected)
        self.assertIs(op.Identity(0), selected)
        self.assertIs(_identity(0), selected)

    def test_nested_overrides_restore_after_exception(self):
        original = public_evaluator.default()
        outer, inner = _SelectionEvaluator(), _SelectionEvaluator()

        def raise_with_inner_override():
            with public_evaluator.default_as(inner):
                self.assert_selected(inner)
                raise RuntimeError("test exception")

        with public_evaluator.default_as(outer):
            self.assert_selected(outer)
            with self.assertRaisesRegex(RuntimeError, "test exception"):
                raise_with_inner_override()
            self.assert_selected(outer)
        self.assertIs(public_evaluator.default(), original)

    def test_global_default_changes_do_not_replace_local_override(self):
        original = public_evaluator.default()
        global_choice, local_choice = _SelectionEvaluator(), _SelectionEvaluator()
        try:
            with public_evaluator.default_as(local_choice):
                public_evaluator.set_default(global_choice)
                self.assert_selected(local_choice)
                contextvars.Context().run(self.assert_selected, global_choice)
            self.assert_selected(global_choice)
        finally:
            public_evaluator.set_default(original)

    def test_copied_context_inherits_but_does_not_share_changes(self):
        original = public_evaluator.default()
        inherited, replacement = _SelectionEvaluator(), _SelectionEvaluator()
        with public_evaluator.default_as(inherited):
            copied = contextvars.copy_context()
        self.assertIs(public_evaluator.default(), original)

        def run_in_copy():
            self.assert_selected(inherited)
            with public_evaluator.default_as(replacement):
                self.assert_selected(replacement)
            self.assert_selected(inherited)

        copied.run(run_in_copy)
        self.assertIs(public_evaluator.default(), original)

    def test_async_tasks_isolate_overlapping_overrides(self):
        original = public_evaluator.default()
        first, second = _SelectionEvaluator(), _SelectionEvaluator()

        async def run():
            first_entered, second_entered, first_exited = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )

            async def first_task():
                try:
                    with public_evaluator.default_as(first):
                        first_entered.set()
                        await second_entered.wait()
                        self.assert_selected(first)
                finally:
                    first_exited.set()

            async def second_task():
                await first_entered.wait()
                with public_evaluator.default_as(second):
                    second_entered.set()
                    await first_exited.wait()
                    self.assert_selected(second)

            await asyncio.wait_for(asyncio.gather(first_task(), second_task()), timeout=10)

        asyncio.run(run())
        self.assertIs(public_evaluator.default(), original)

    def test_async_task_inherits_context_at_creation(self):
        inherited, later = _SelectionEvaluator(), _SelectionEvaluator()

        async def run():
            proceed = asyncio.Event()

            async def child():
                await proceed.wait()
                self.assert_selected(inherited)
                with public_evaluator.default_as(later):
                    self.assert_selected(later)
                self.assert_selected(inherited)

            with public_evaluator.default_as(inherited):
                task = asyncio.create_task(child())
            with public_evaluator.default_as(later):
                proceed.set()
                await asyncio.wait_for(task, timeout=10)
                self.assert_selected(later)

        original = public_evaluator.default()
        asyncio.run(run())
        self.assertIs(public_evaluator.default(), original)

    def test_threads_isolate_overlapping_overrides(self):
        original = public_evaluator.default()
        first, second = _SelectionEvaluator(), _SelectionEvaluator()
        first_entered, second_entered, first_exited = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def first_thread():
            try:
                self.assertIs(public_evaluator.default(), original)
                with public_evaluator.default_as(first):
                    first_entered.set()
                    self.assertTrue(second_entered.wait(timeout=10))
                    self.assert_selected(first)
            finally:
                first_exited.set()

        def second_thread():
            self.assertTrue(first_entered.wait(timeout=10))
            with public_evaluator.default_as(second):
                second_entered.set()
                self.assertTrue(first_exited.wait(timeout=10))
                self.assert_selected(second)

        # Explicit empty contexts make the test independent of thread inheritance flags.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(contextvars.Context().run, worker)
                for worker in (first_thread, second_thread)
            ]
            for future in futures:
                future.result(timeout=30)
        self.assertIs(public_evaluator.default(), original)

    def test_thread_with_copied_context_inherits_override(self):
        inherited = _SelectionEvaluator()
        with public_evaluator.default_as(inherited):
            copied = contextvars.copy_context()
        original = public_evaluator.default()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(copied.run, self.assert_selected, inherited).result(timeout=10)
        self.assertIs(public_evaluator.default(), original)


@script()
def _graph_inner(x):
    @graph()
    def body(y):
        return op.Add(y, x)

    seq = op.SequenceConstruct(x, x)
    result = op.SequenceMap(seq, body=body)
    return op.SequenceAt(result, 0)


@script()
def _graph_outer(x):
    z = _graph_inner(x)

    @graph()
    def body(y):
        return op.Add(y, z)

    seq = op.SequenceConstruct(x, x)
    return op.SequenceMap(seq, body=body)


class _WrappedEvaluator(evaluator.ORTEvaluator):
    def eval_function(self, function, args, kwargs):
        def invoke():
            return super(_WrappedEvaluator, self).eval_function(function, args, kwargs)

        return invoke()


class _CustomEvaluator:
    """A protocol implementation that invokes Python without BaseEvaluator."""

    def eval_function(self, function, args, kwargs):
        def invoke():
            return function.function(*args, **kwargs)

        return invoke()

    def eval_op(self, op, args, kwargs):
        return evaluator.ort_evaluator.eval_op(op, args, kwargs)


class GraphContextTest(unittest.TestCase):
    def test_wrappers_nested_calls_and_closures(self):
        @functools.wraps(_graph_outer)
        def wrapped(x):
            return _graph_outer(x)

        for selected in (
            evaluator.ort_evaluator,
            evaluator.ort_mixed_evaluator,
            _WrappedEvaluator(),
            _CustomEvaluator(),
        ):
            with self.subTest(evaluator=type(selected).__name__):
                with public_evaluator.default_as(selected):
                    for x in (
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([3.0, 4.0], dtype=np.float32),
                    ):
                        output = wrapped(tensor.Tensor(x))
                        self.assertEqual(len(output), 2)
                        for item in output:
                            np.testing.assert_equal(item.value, 3 * x)
                self.assertIsNone(evaluator._current_function.get())

    def test_reference_evaluator_nested_calls_without_captures(self):
        @script()
        def inner(x):
            @graph()
            def body(y):
                return op.Mul(y, y)

            seq = op.SequenceConstruct(x)
            result = op.SequenceMap(seq, body=body)
            return op.ConcatFromSequence(result, axis=0)

        @script()
        def outer(x):
            z = inner(x)

            @graph()
            def body(y):
                return op.Add(y, y)

            seq = op.SequenceConstruct(z)
            return op.SequenceMap(seq, body=body)

        with public_evaluator.default_as(evaluator.OnnxReferenceRuntimeEvaluator()):
            x = np.array([1.0, 2.0], dtype=np.float32)
            np.testing.assert_equal(outer(x), [2 * x * x])
        self.assertIsNone(evaluator._current_function.get())

    def test_direct_base_evaluator_call_establishes_context(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        output = evaluator.ort_evaluator.eval_function(_graph_outer, [x], {})
        np.testing.assert_equal(output, [3 * x, 3 * x])
        self.assertIsNone(evaluator._current_function.get())

    def test_function_context_restored_after_exception(self):
        class FailingEvaluator(_CustomEvaluator):
            def eval_op(self, op, args, kwargs):
                raise RuntimeError("test exception")

        with evaluator._function_context(_identity):
            with public_evaluator.default_as(FailingEvaluator()), self.assertRaisesRegex(
                RuntimeError, "test exception"
            ):
                _graph_outer(np.array([1.0], dtype=np.float32))
            self.assertIs(evaluator._current_function.get(), _identity)
        self.assertIsNone(evaluator._current_function.get())

    def test_graph_requires_active_script_function(self):
        with self.assertRaisesRegex(RuntimeError, "active ONNXScript function"):
            _graph_inner.function(np.array([1.0], dtype=np.float32))

    def test_graph_rejects_indirect_nesting(self):
        with evaluator._function_context(_graph_outer), self.assertRaisesRegex(
            RuntimeError, "defined directly"
        ):
            _graph_inner.function(np.array([1.0], dtype=np.float32))

    def test_graph_nested_inside_graph_body_is_not_supported_in_eager_mode(self):
        @script()
        def nested(x):
            @graph()
            def body(y):
                @graph()
                def inner(z):
                    return op.Identity(z)

                seq = op.SequenceConstruct(y, y)
                return op.SequenceMap(seq, body=inner)

            seq = op.SequenceConstruct(x, x)
            return op.SequenceMap(seq, body=body)

        with public_evaluator.default_as(
            evaluator.ort_mixed_evaluator
        ), self.assertRaisesRegex(
            RuntimeError, "graphs nested inside graph bodies are not supported"
        ):
            nested(np.array([1.0], dtype=np.float32))
        self.assertIsNone(evaluator._current_function.get())


class EvaluatorTest(unittest.TestCase):
    def test_evaluator(self):
        @script()
        def seq_map(x: FLOAT["N"]):  # noqa: F821
            seq1 = op.SequenceConstruct(x, x + 1, x + 2)

            @graph()
            def square(y: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
                return op.Mul(y, y)

            seq2 = op.SequenceMap(seq1, body=square)
            return seq2

        x = np.array([0.0, 1.0], dtype=np.float32)
        expected = [t * t for t in (x, x + 1, x + 2)]

        # Test using (current) default evaluator
        output = seq_map(x)
        np.testing.assert_equal(output, expected)

        # Test using ort-mixed-evaluator
        with evaluator.default_as(evaluator.ort_mixed_evaluator):
            output = seq_map(x)
        np.testing.assert_equal(output, expected)

        # Test using ort-evaluator
        with evaluator.default_as(evaluator.ort_evaluator):
            output = seq_map(x)
        np.testing.assert_equal(output, expected)


class ORTEvaluatorTest(unittest.TestCase):
    def test_it_ignores_unknown_function_kwargs_when_option_set_to_true(self):
        @script()
        def test_function(x, y: float = 1.0):
            return op.Add(x, y)

        x = np.array(0.0, dtype=np.float32)
        expected = np.array(1.0, dtype=np.float32)
        with evaluator.default_as(evaluator.ORTEvaluator(ignore_unknown_function_kwargs=True)):
            output = test_function(x, unknown=42)  # pylint: disable=unexpected-keyword-arg

        np.testing.assert_equal(output, expected)

    def test_it_raise_on_unknown_function_kwargs_by_default(self):
        @script()
        def test_function(x, y: float = 1.0):
            return op.Add(x, y)

        x = np.array(0.0, dtype=np.float32)
        with evaluator.default_as(evaluator.ORTEvaluator()):  # noqa: SIM117
            with self.assertRaises(TypeError):
                _ = test_function(x, unknown=42)  # pylint: disable=unexpected-keyword-arg


if __name__ == "__main__":
    unittest.main()
