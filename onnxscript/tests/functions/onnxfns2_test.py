import unittest

from onnxscript.tests.common import onnx_script_test_case
from onnxscript.tests.models import onnxfns2


class TestOnnxFns(onnx_script_test_case.OnnxScriptTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rtol = 1e-05

    def test_onnxfns_reduce_sum_square(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceSumSquare,
            keepdims=default_keepdims,
            skip_test_names=[
                "test_reduce_sum_square_empty_set",
            ],
        )

    def test_onnxfns_reduce_l1(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL1,
            keepdims=default_keepdims,
            skip_test_names=[
                "test_reduce_l1_empty_set",
            ],
        )

    def test_onnxfns_reduce_l2(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL2,
            keepdims=default_keepdims,
            skip_test_names=[
                "test_reduce_l2_empty_set",
            ],
        )

    def test_onnxfns_reduce_log_sum(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSum,
            keepdims=default_keepdims,
            skip_test_names=[
                "test_reduce_log_sum_empty_set",
            ],
        )

    def test_onnxfns_reduce_log_sum_exp(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSumExp,
            keepdims=default_keepdims,
            skip_test_names=[
                "test_reduce_log_sum_exp_empty_set",
            ],
        )

    def test_onnxfns_hardmax(self):
        default_axis = -1

        self.run_onnx_test(onnxfns2.Hardmax, axis=default_axis, skip_test_names=[])

    # converter generated model fails ort shape inferencing.
    # it does not yet support workflow op in a function node.
    # def test_onnxfns_depth_to_space(self):
    #     default_mode = 'DCR'

    #     self.run_onnx_test(
    #         onnxfns2.DepthToSpace,
    #         mode=default_mode,
    #         skip_test_names=[])

    def test_onnxfns_space_to_depth(self):
        self.run_onnx_test(onnxfns2.SpaceToDepth, skip_test_names=[], skip_eager_test=True)


if __name__ == "__main__":
    unittest.main()
