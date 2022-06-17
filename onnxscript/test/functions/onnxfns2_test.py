import unittest
from packaging.version import Version
import onnxruntime
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.test.models import onnxfns2


class TestOnnxFns(OnnxScriptTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestOnnxFns, cls).setUpClass()
        cls.rtol = 1e-05

    @unittest.skipIf(Version(onnxruntime.__version__) < Version('1.12'),
                     reason="onnxruntime does not support that scenario.")
    def test_onnxfns_reduce_sum_square(self):
        default_axes = None
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceSumSquare,
            axes=default_axes,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                'test_reduce_sum_square_default_axes_keepdims_example',
                'test_reduce_sum_square_default_axes_keepdims_random'])

    @unittest.skipIf(Version(onnxruntime.__version__) < Version('1.12'),
                     reason="onnxruntime does not support that scenario.")
    def test_onnxfns_reduce_l1(self):
        default_axes = None
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL1,
            axes=default_axes,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                'test_reduce_l1_default_axes_keepdims_example',
                'test_reduce_l1_default_axes_keepdims_random'])

    @unittest.skipIf(Version(onnxruntime.__version__) < Version('1.12'),
                     reason="onnxruntime does not support that scenario.")
    def test_onnxfns_reduce_l2(self):
        default_axes = None
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceL2,
            axes=default_axes,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                'test_reduce_l2_default_axes_keepdims_example',
                'test_reduce_l2_default_axes_keepdims_random'])

    @unittest.skipIf(Version(onnxruntime.__version__) < Version('1.12'),
                     reason="onnxruntime does not support that scenario.")
    def test_onnxfns_reduce_log_sum(self):
        default_axes = None
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSum,
            axes=default_axes,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                'test_reduce_log_sum_default'])

    def test_onnxfns_reduce_log_sum_exp(self):
        default_axes = None
        default_keepdims = 1

        self.run_onnx_test(
            onnxfns2.ReduceLogSumExp,
            axes=default_axes,
            keepdims=default_keepdims,
            # default attributes are not supported yet.
            skip_test_names=[
                'test_reduce_log_sum_exp_default_axes_keepdims_example',
                'test_reduce_log_sum_exp_default_axes_keepdims_random'])

    @unittest.skipIf(Version(onnxruntime.__version__) < Version('1.12'),
                     reason="onnxruntime does not support that scenario.")
    def test_onnxfns_hardmax(self):
        default_axis = -1

        self.run_onnx_test(
            onnxfns2.Hardmax,
            axis=default_axis,
            skip_test_names=[])

    # converter generated model fails ort shape inferencing.
    # it does not yet support workflow op in a function node.
    # def test_onnxfns_depth_to_space(self):
    #     default_mode = 'DCR'

    #     self.run_onnx_test(
    #         onnxfns2.DepthToSpace,
    #         mode=default_mode,
    #         skip_test_names=[])

    def test_onnxfns_space_to_depth(self):

        self.run_onnx_test(
            onnxfns2.SpaceToDepth,
            skip_test_names=[],
            skip_eager_test=True)


if __name__ == '__main__':
    unittest.main()
