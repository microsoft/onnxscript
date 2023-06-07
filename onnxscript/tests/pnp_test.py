import unittest
from copy import deepcopy

import numpy as np

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tests.common import onnx_script_test_case, testutils

from onnxscript.tests.models.pnp import roi_indices_3d, aggrregate_predictor_output, sliding_window_inference, predict_mock, predict_mock_2

class PnpOpTest(onnx_script_test_case.OnnxScriptTestCase):
    def test_roi_indices_3d(delf):
        roi_D, roi_H, roi_W = 2, 2, 2
        start = np.array([0, 1, 1], dtype=np.int64)
        stop = start + np.array([roi_D, roi_H, roi_W], dtype=np.int64)
        step = np.ones((3,), dtype=np.int64)
        indices, roi_shape = roi_indices_3d(start, stop, step)
        print(indices)


    def test_aggrregate_pred(self):
        D, H, W = 2, 3, 4
        roi_D, roi_H, roi_W = 2, 2, 2
        start = np.array([0, 1, 1], dtype=np.int64)
        stop = start + np.array([roi_D, roi_H, roi_W], dtype=np.int64)
        pred = np.ones((roi_D, roi_H, roi_W), dtype=np.float32)
        importance_map = np.ones((roi_D, roi_H, roi_W), dtype=np.float32)
        aggrregated_pred = np.zeros((D, H, W), dtype=np.float32)
        aggrregated_count = np.zeros((D, H, W), dtype=np.int64)

        aggrregated_pred_expected = deepcopy(aggrregated_pred)
        aggrregated_count_expected = deepcopy(aggrregated_count)
        for i in range(start[0], stop[0]):
            for j in range(start[1], stop[1]):
                for k in range(start[2], stop[2]):
                    aggrregated_pred_expected[i, j, k] += pred[i - start[0], j - start[1], k - start[2]]
                    aggrregated_count_expected[i, j, k] += 1
        

        case = onnx_script_test_case.FunctionTestParams(
            aggrregate_predictor_output,
            [pred, start, stop, importance_map, aggrregated_pred, aggrregated_count],
            [aggrregated_pred_expected, aggrregated_count_expected]
            )
        self.run_eager_test(case)
        self.run_converter_test(case)

    def test_sliding_window_inference(self):
        N, C, D, H, W = 1, 1, 2, 4, 6
        roi_D, roi_H, roi_W = 2, 2, 2
        input = np.ones((N, C, D, H, W), dtype=np.float32)
        roi_size = np.array([roi_D, roi_H, roi_W], dtype=np.int64)
        output = predict_mock_2(input)

        case = onnx_script_test_case.FunctionTestParams(
            sliding_window_inference,
            [input, roi_size],
            [output],
            )
        self.run_eager_test(case)
        # converter test failed but it is fune with local ort build. It may be that ort 1.14.1 has an issue but not the current 1.15.0
        self.run_converter_test(case)


if __name__ == "__main__":
    unittest.main()
