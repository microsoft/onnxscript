# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import pathlib
import unittest

from onnxscript import ir, version_converter

model_folder_path = pathlib.Path(__file__).resolve().parent.parent.parent / "testdata"


class ModelTest(unittest.TestCase):
    def test_model_runs_and_matches_accuracy_after_conversion_fallback_true(self):
        model_path = model_folder_path / "e2e_models/torchscript_model/torchscript_model.onnx"
        model = ir.load(model_path)

        # Down convert the model with the onnx version converter
        version_converter.convert_version(model, target_version=16, fallback=True)
        self.assertEqual(model.opset_imports[""], 16)


if __name__ == "__main__":
    unittest.main()
