import unittest

import onnxruntime
import pytest

from onnxscript._internal import version_utils
from onnxscript.tests.common import onnx_script_test_case
from onnxscript.tests.models import onnxfns2


@pytest.mark.xfail(
    version_utils.onnxruntime_older_than("1.15") and not version_utils.onnx_older_than("1.14"),
    reason="ORT <=1.14 does not support IR version 9 produced by ONNX 1.14",
)
class TestOnnxFns(onnx_script_test_case.OnnxScriptTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rtol = 1e-05

    @unittest.skipIf(
        version_utils.onnxruntime_older_than("1.12"),
        reason="onnxruntime does not support that scenario.",
    )
    def test_onnxfns_reduce_sum_square(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceSumSquare,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                "test_reduce_sum_square_default_axes_keepdims_example",
                "test_reduce_sum_square_default_axes_keepdims_random",
            ],
        )

    def test_onnxfns_reduce_l1(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL1,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                "test_reduce_l1_default_axes_keepdims_example",
                "test_reduce_l1_default_axes_keepdims_random",
            ],
        )

    def test_onnxfns_reduce_l2(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL2,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                "test_reduce_l2_default_axes_keepdims_example",
                "test_reduce_l2_default_axes_keepdims_random",
            ],
        )

    @unittest.skipIf(
        version_utils.onnxruntime_older_than("1.12"),
        reason="onnxruntime does not support that scenario.",
    )
    def test_onnxfns_reduce_log_sum(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSum,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=["test_reduce_log_sum_default"],
        )

    @unittest.skipIf(
        version_utils.onnxruntime_older_than("1.12"),
        reason="onnxruntime does not support that scenario.",
    )
    def test_onnxfns_reduce_log_sum_exp(self):
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSumExp,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                "test_reduce_log_sum_exp_default_axes_keepdims_example",
                "test_reduce_log_sum_exp_default_axes_keepdims_random",
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

    @unittest.skipIf(
        onnxruntime.__version__[:4] == "1.14",
        reason="onnxruntime 1.14 Segfaults.",
    )
    def test_onnxfns_space_to_depth(self):
        self.run_onnx_test(onnxfns2.SpaceToDepth, skip_test_names=[], skip_eager_test=True)


if __name__ == "__main__":
    unittest.main()
