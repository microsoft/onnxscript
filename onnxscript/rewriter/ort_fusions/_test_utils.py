# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnx
import onnxruntime
import packaging.version

import onnxscript.ir as ir
import onnxscript.ir._io as io


def _save(model, modelpath):
    if isinstance(model, onnx.ModelProto):
        onnx.save(model, modelpath)
    else:
        assert isinstance(model, ir.Model)
        io.save(model, modelpath)


ORT_VERSION = packaging.version.Version(onnxruntime.__version__)


def ort_run(model_name: str, model, inputs):
    providers = ["CPUExecutionProvider"]
    model_proto = ir.serde.serialize_model(model)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        model_proto.SerializeToString(), options, providers=providers
    )
    return session.run(None, inputs)


def assert_allclose(outputs, expected_outputs, rtol=1e-3, atol=1e-3):
    for i, (baseline_output, optimized_output) in enumerate(zip(expected_outputs, outputs)):
        try:
            np.testing.assert_equal(baseline_output.shape, optimized_output.shape)
            np.testing.assert_allclose(baseline_output, optimized_output, rtol=rtol, atol=atol)
        except AssertionError as e:
            diff_mask = ~np.isclose(baseline_output, optimized_output, rtol=rtol, atol=atol)
            diff = np.where(diff_mask, "X", " ")
            print(diff)
            print(f"Failed for output {i} with rtol={rtol} and atol={atol}\n{e}")
            raise
