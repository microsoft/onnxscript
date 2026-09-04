# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx_ir as ir
import onnxruntime
import packaging.version

ORT_VERSION = packaging.version.Version(onnxruntime.__version__)


def ort_run(model_name: str, model, inputs):
    providers = ["CPUExecutionProvider"]
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Save the model to a temporary file and load it from disk. This uses external data
    # to store tensors so that large models are not limited by the 2GB protobuf
    # serialization limit, which can otherwise raise
    # ``google.protobuf.message.EncodeError: Failed to serialize proto``.
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, f"{model_name}.onnx")
        ir.save(model, model_path, external_data=f"{model_name}.onnx.data")
        session = onnxruntime.InferenceSession(model_path, options, providers=providers)
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
