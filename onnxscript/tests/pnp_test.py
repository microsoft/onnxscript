import unittest
from copy import deepcopy

import numpy as np

import onnx
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tests.common import onnx_script_test_case, testutils

from onnxscript.tests.models.pnp import (
    roi_indices_3d,
    aggregate_predictor_output,
    sliding_window_inference,
    predict_mock,
    predict_mock_2,
    Opset18Ext,
    get_scan_interval_script,
)

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
            aggregate_predictor_output,
            [pred, start, stop, importance_map, aggrregated_pred, aggrregated_count],
            [aggrregated_pred_expected, aggrregated_count_expected]
            )
        self.run_eager_test(case)
        self.run_converter_test(case)

    def test_get_scan_interval_script(self):
        D, H, W = 100, 111, 127
        roi_D, roi_H, roi_W = 64, 64, 32
        image_size = np.array((D, H, W), dtype=np.int64)
        roi_size = np.array([roi_D, roi_H, roi_W], dtype=np.int64)
        overlap = (0.25, ) * 3

        scan_interval = [int((1 - overlap[i]) * roi_size[i]) if image_size[i] > roi_size[i] else image_size[i] for i in range(3)]
        save_model = True
        if save_model:
            model = get_scan_interval_script.function_ir.to_model_proto(producer_name="monai")
            onnx.save(model, "C:/temp/test_get_scan_interval.onnx")
        case = onnx_script_test_case.FunctionTestParams(
            get_scan_interval_script,
            [image_size, roi_size],
            [scan_interval],
            )
        # eager test failed with data Not equal. this is expected as the output is not run with predictor.
        self.run_eager_test(case)
        # converter test extect to fail with No Op registered for OpaqueOp with domain_version of 18
        self.run_converter_test(case)

    def test_sliding_window_inference(self):
        N, C, D, H, W = 1, 1, 100, 111, 127
        roi_D, roi_H, roi_W = 64, 64, 32
        input = np.ones((N, C, D, H, W), dtype=np.float32)
        roi_size = np.array([roi_D, roi_H, roi_W], dtype=np.int64)

        #output = predict_mock_2(input)
        seg_C = 2
        output_expected = np.zeros((N, seg_C, D, H, W), dtype=np.float32)
        outout_count = np.zeros((N, 1, D, H, W), dtype=np.int64)
        op = Opset18Ext()
        for d in range(0, D, roi_D):
            if d + roi_D > D:
                d = D - roi_D
            for h in range(0, H, roi_H):
                if h + roi_H > H:
                    h = H - roi_H
                for w in range(0, W, roi_W):
                    if w + roi_W > W:
                        w = W - roi_W
                    input_patch = input[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W]
                    output_expected[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W] += op.OpaqueOp(input_patch, model_path="C:/Temp/sliding_window_predictor_sw_batch_size_is_1.onnx")
                    outout_count[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W] += 1

        output_expected /= outout_count

        save_model = False
        if save_model:
            model = sliding_window_inference.function_ir.to_model_proto(producer_name="monai")
            onnx.save(model, "C:/temp/test_sliding_window_inference.onnx")
        case = onnx_script_test_case.FunctionTestParams(
            sliding_window_inference,
            [input, roi_size],
            [output_expected],
            )
        # eager test failed with data Not equal. this is expected as the output is not run with predictor.
        self.run_eager_test(case)
        # converter test extect to fail with No Op registered for OpaqueOp with domain_version of 18
        self.run_converter_test(case)


if __name__ == "__main__":
    unittest.main()
