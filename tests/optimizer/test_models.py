from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime
import parameterized

from onnxscript import optimizer
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

        model_dir = f"{model_folder_path}/{model_name}/dynamo"
        model = onnx.load(f"{model_dir}/{model_name}_dynamo.onnx")
        model = optimizer.optimize(
            model,
            onnx_shape_inference=False,
        )

        with tempfile.TemporaryDirectory() as tmp_folder:
            optimized_model_path = f"{tmp_folder}/{model_name}_opt.onnx"
            onnx.save(
                model,
                optimized_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
            )

            sess = onnxruntime.InferenceSession(
                optimized_model_path,
                providers=["CPUExecutionProvider"],
            )

            inputs, expected_outputs = evaluation_utils.load_test_data(
                model_dir, [i.name for i in model.graph.input]
            )

            input_names = [i.name for i in sess.get_inputs()]
            assert set(input_names) == set(inputs.keys())

            outputs = sess.run(None, inputs)

            for output, expected_output in zip(outputs, expected_outputs):
                np.testing.assert_allclose(
                    output, expected_output, rtol=1e-3, atol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
