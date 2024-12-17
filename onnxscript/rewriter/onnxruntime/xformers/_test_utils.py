# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx
import onnxruntime

import onnxscript.ir as ir
import onnxscript.ir._io as io


def _save(model, modelpath):
    if isinstance(model, onnx.ModelProto):
        onnx.save(model, modelpath)
    else:
        assert isinstance(model, ir.Model)
        io.save(model, modelpath)


def ort_run(model_name: str, model, inputs):
    providers = ["CPUExecutionProvider"]
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, f"{model_name}.onnx")
        io.save(model, model_path)
        # Run model
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        ort_outputs = session.run(None, inputs)
    return ort_outputs


def assert_allclose(outputs, expected_outputs, rtol=1e-2, atol=1e-2):
    for i, (baseline_output, optimized_output) in enumerate(zip(expected_outputs, outputs)):
        try:
            np.testing.assert_equal(baseline_output.shape, optimized_output.shape)
            np.testing.assert_allclose(baseline_output, optimized_output, rtol=rtol, atol=atol)
        except AssertionError as e:
            print(f"Failed for output {i} with rtol={rtol} and atol={atol}\n{e}")
            raise
