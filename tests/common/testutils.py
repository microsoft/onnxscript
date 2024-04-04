# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import pathlib
import unittest

import numpy as np
import onnx
import onnxruntime

from onnxscript import optimizer
from onnxscript._legacy_ir import visitor
from onnxscript.rewriter import onnxruntime as ort_rewriter
from onnxscript.utils import evaluation_utils


class TestBase(unittest.TestCase):
    """The base class for testing ONNX Script functions for internal use."""

    def validate(self, fn):
        """Validate script function translation."""
        return fn.to_function_proto()


def skip_if_no_cuda(reason: str):
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not onnxruntime.get_device() == "GPU":
                raise unittest.SkipTest(f"GPU is not available. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


class OpTypeAnalysisVisitor(visitor.ProtoVisitorCore):
    def __init__(self):
        super().__init__()
        self.op_types = set()

    def visit_model(self, model: onnx.ModelProto):
        self.op_types = set()
        super().visit_model(model)

    def process_node(self, node: onnx.NodeProto):
        self.op_types.add((node.domain, node.op_type, getattr(node, "overload", "")))
        return super().process_node(node)


def test_onnxruntime_rewrite(
    model_basename: str,
    model_count: int,
    expected_optypes: set[tuple[str, str, str]],
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    unittest_root_dir = dir_path.parent.parent / "testdata" / "unittest_models"
    for model_index in range(model_count):
        model_name = f"{model_basename}_{model_index}"
        model_dir = unittest_root_dir / f"{model_name}"
        model_path = model_dir / f"{model_name}.onnx"
        model = onnx.load(model_path)

        # TODO: Parity issue with randomly generated data. Need investigation.
        # inputs = generate_random_input(model)
        inputs, expected_outputs = evaluation_utils.load_test_data(
            model_dir, [i.name for i in model.graph.input]
        )

        optimized = optimizer.optimize(
            model,
            onnx_shape_inference=False,
            num_iterations=2,
        )
        rewritten = ort_rewriter.rewrite(optimized)
        # NOTE: uncomment this to save the optimized model.
        # onnx.save(rewritten, model_dir / f"{model_name}_opt.onnx")

        # Check expected operator is found.
        optype_analysis = OpTypeAnalysisVisitor()
        optype_analysis.visit_model(rewritten)
        for domain, op_type, overload in expected_optypes:
            if (domain, op_type, overload) not in optype_analysis.op_types:
                raise AssertionError(
                    f"Expected op type {domain}:{op_type}:{overload} not found in rewritten model."
                )

        # Run baseline model
        providers = ["CUDAExecutionProvider"]

        # Run optimized model
        optimized_session = onnxruntime.InferenceSession(
            rewritten.SerializeToString(), providers=providers
        )
        optimized_outputs = optimized_session.run(None, inputs)

        for i, (baseline_output, optimized_output) in enumerate(
            zip(expected_outputs, optimized_outputs)
        ):
            try:
                np.testing.assert_equal(baseline_output.shape, optimized_output.shape)
                np.testing.assert_allclose(
                    baseline_output, optimized_output, rtol=rtol, atol=atol
                )
            except AssertionError as e:
                print(
                    f"Failed for model {model_name} and output {i} with rtol={rtol} and atol={atol}\n{e}"
                )
                raise
