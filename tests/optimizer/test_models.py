# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import onnx
import onnx.inliner
import onnxruntime
import parameterized

from onnxscript import optimizer
from onnxscript.rewriter import onnxruntime as ort_rewriter
from onnxscript.utils import evaluation_utils

_SKIP_TABLE = {}

model_folder_path = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "testdata" / "e2e_models"
)

# List all entries in the directory and filter for directories
model_names = [entry.name for entry in model_folder_path.iterdir() if entry.is_dir()]


class ModelTest(unittest.TestCase):
    @parameterized.parameterized.expand(model_names)
    def test_model_runs_and_matches_accuracy_after_optimization(self, model_name):
        test_id = model_name  # This can be expanded in the future with more parameters, e.g. optimization options
        if (skip_reason := _SKIP_TABLE.get(test_id)) is not None:
            self.skipTest(skip_reason)

        model_dir = pathlib.Path(model_folder_path) / model_name / "dynamo"
        model_path = model_dir / f"{model_name}_dynamo.onnx"
        if not model_path.exists():
            self.skipTest(f"Model {model_name!r} does not exist")
        model = onnx.load(model_path)
        model = optimizer.optimize(model)

        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            optimized_model_path = tmp_folder / f"{model_name}_opt.onnx"
            onnx.save(
                model,
                optimized_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
            )

            session = onnxruntime.InferenceSession(
                optimized_model_path, providers=("CPUExecutionProvider",)
            )

            inputs, expected_outputs = evaluation_utils.load_test_data(
                model_dir, [i.name for i in model.graph.input]
            )

            input_names = [i.name for i in session.get_inputs()]
            assert set(input_names) == set(inputs.keys())

            outputs = session.run(None, inputs)
            # Free the session so the model file is no longer used
            del session

            for output, expected_output in zip(outputs, expected_outputs):
                np.testing.assert_allclose(output, expected_output, rtol=1e-3, atol=1e-3)

    def test_optimizer_after_inlining(self):
        model_dir = pathlib.Path(model_folder_path) / ".." / "dort_models"
        filename = model_dir / "llama_forward.onnx"

        onnx_model = onnx.load(filename)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        # first time
        onnx_model = optimizer.optimize(onnx_model)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_model = ort_rewriter.rewrite(onnx_model)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        # inline
        onnx_model = onnx.inliner.inline_local_functions(onnx_model)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        # second time
        onnx_model = optimizer.optimize(onnx_model)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_model = ort_rewriter.rewrite(onnx_model)
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
