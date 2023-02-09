# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import numpy as np

from onnxscript.tests.common import onnx_script_test_case
from onnxscript.tests.models import if_statement


class TestOnnxIf(onnx_script_test_case.OnnxScriptTestCase):
    def test_if(self):
        n = 8
        np.random.seed(0)
        a = np.random.rand(n).astype("float32").T
        b = np.random.rand(n).astype("float32").T

        # FIXME(liqunfu): expected are from ort evaluation.
        # needs numpy oxs to provide expected instead.
        expected = np.array(
            [
                0.5488135,
                0.71518934,
                0.60276335,
                0.5448832,
                0.4236548,
                0.6458941,
                0.4375872,
                0.891773,
            ],
            dtype=np.float32,
        )

        cases = [
            onnx_script_test_case.FunctionTestParams(if_statement.maxsum, [a, b], [expected])
        ]
        for case in cases:
            # FAIL : Node () Op (local_function) [TypeInferenceError]
            # GraphProto attribute inferencing is not enabled
            # in this InferenceContextImpl instance.
            # self.run_converter_test(case)
            self.run_eager_test(case)


if __name__ == "__main__":
    unittest.main()
