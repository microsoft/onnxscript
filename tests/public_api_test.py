# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Smoke tests for the supported core facades, not compiler implementation details."""

import inspect
import os
import tempfile
import unittest

import onnx

import onnxscript
from onnxscript import FLOAT, evaluator, export_onnx_lib, graph, opset18, script, values
from onnxscript.values import OnnxFunction, Op, Opset, TracedOnnxFunction


class PublicCoreApiTest(unittest.TestCase):
    def test_public_value_imports(self):
        self.assertIs(onnxscript.OnnxFunction, OnnxFunction)
        self.assertIs(onnxscript.TracedOnnxFunction, TracedOnnxFunction)
        self.assertIsInstance(opset18, Opset)
        self.assertIsInstance(Op(opset18, "Identity"), Op)
        for name in ("OnnxFunction", "TracedOnnxFunction", "Op", "Opset"):
            self.assertIn(name, values.__all__)

    def test_public_entry_point_signatures(self):
        inspect.signature(script).bind()
        inspect.signature(script).bind(
            opset=Opset("example", 1), default_opset=opset18, input_types=[FLOAT["N"]]
        )
        inspect.signature(graph).bind()
        inspect.signature(export_onnx_lib).bind([], "library.onnx")
        inspect.signature(OnnxFunction.to_model_proto).bind(
            object(), input_types=[FLOAT["N"]], output_types=[FLOAT["N"]]
        )
        inspect.signature(OnnxFunction.to_function_proto).bind(object())
        inspect.signature(OnnxFunction.graph).bind(object())

    def test_script_preserves_public_callable_signature(self):
        def function(x: FLOAT["N"], alpha: float = 1.0) -> FLOAT["N"]:  # noqa: F821
            return opset18.LeakyRelu(x, alpha=alpha)

        compiled = script()(function)
        self.assertIsInstance(compiled, OnnxFunction)
        self.assertEqual(inspect.signature(compiled), inspect.signature(function))
        inspect.signature(compiled).bind(object(), alpha=0.5)
        onnx.checker.check_model(compiled.to_model_proto())
        self.assertIsInstance(compiled.to_function_proto(), onnx.FunctionProto)

    def test_evaluator_facade(self):
        for name in (
            "Evaluator",
            "ORTEvaluator",
            "ORTMixedEvaluator",
            "OnnxReferenceRuntimeEvaluator",
            "default",
            "set_default",
            "default_as",
        ):
            self.assertIn(name, evaluator.__all__)
            self.assertTrue(callable(getattr(evaluator, name)))
        previous = evaluator.default()
        reference = evaluator.OnnxReferenceRuntimeEvaluator()
        inspect.signature(evaluator.default).bind()
        inspect.signature(evaluator.set_default).bind(reference)
        inspect.signature(evaluator.default_as).bind(reference)
        with evaluator.default_as(reference):
            self.assertIs(evaluator.default(), reference)
        self.assertIs(evaluator.default(), previous)

    def test_legacy_library_export_remains_available(self):
        @script()
        def identity(x: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            return opset18.Identity(x)

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "library.onnx")
            export_onnx_lib([identity], filename)
            container = onnx.load(filename)
        self.assertEqual(len(container.graph.node), 0)
        self.assertEqual(len(container.graph.input), 0)
        self.assertEqual(len(container.graph.output), 0)
        self.assertEqual(len(container.functions), 1)
        exported = container.functions[0]
        expected = identity.to_function_proto()
        self.assertEqual(exported.name, expected.name)
        self.assertEqual(exported.domain, expected.domain)
        self.assertEqual(exported.input, expected.input)
        self.assertEqual(exported.output, expected.output)
        self.assertEqual(exported.node, expected.node)


if __name__ == "__main__":
    unittest.main()
