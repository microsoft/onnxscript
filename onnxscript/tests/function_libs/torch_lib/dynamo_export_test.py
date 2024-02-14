import copy
import inspect
import itertools
import sys
import unittest

import torch
from torch.onnx import ExportOptions
from torch.onnx import _OrtBackend as OrtBackend
from torch.onnx import _OrtBackendOptions as OrtBackendOptions


class FuncModule(torch.nn.Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.ppp = torch.nn.Parameter(torch.Tensor([1]))
        self.params = torch.nn.ParameterList(list(params))

    def forward(self, *args):
        f_args = list(itertools.chain(args, self.params))
        f_args[0] = f_args[0] + self.ppp
        res = self.f(*f_args)
        return res


class FuncModuleModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.mod = f
        self.ppp = torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, *args):
        x = args[0] + self.ppp
        res = self.mod(x, *args[1:])
        return res


def make_aot_ort(dynamic: bool = False):

    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
            )
        )
    )
    return ort_backend, ort_backend


class TestOperatorsOnnxrt(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def assertONNX(
        self,
        f,
        args,
        onnx_export: str,
        params=None,
        fullgraph: bool = True,
        atol=1e-6,
        rtol=1e-6,
        opset_version=None,
        test_backward=True,
        #
        input_names=None,
        dynamic_axes=None,
        keep_initializers_as_inputs=None,
    ):
        if sys.platform == "win32":
            raise unittest.SkipTest("Windows not supported yet.")
        assert isinstance(onnx_export, str), f"Export onnx is wrong for f={f}"
        assert opset_version is None, f"opset={opset_version}, only default opset is supported"
        if isinstance(args, torch.Tensor):
            args = [args]
        if params is None:
            params = ()
        if isinstance(f, torch.nn.Module):
            model = FuncModuleModule(f)
        else:
            model = FuncModule(f, params)
        model.eval()

        if test_backward:
            # forward/backward
            local_aot_ort, _ = make_aot_ort(dynamic=False)

            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=local_aot_ort,
                dynamic=False,
                fullgraph=fullgraph,
            )

            baseline_result = model(*args)
            result = compiled_model(*args)

            if isinstance(baseline_result, tuple):
                baseline_result = baseline_result[0]
                result = result[0]
            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol, equal_nan=True
                )

                baseline_result.sum().backward()
                result.sum().backward()

                l1 = list(model.parameters())
                l2 = list(compiled_model.parameters())
                self.assertEqual(len(l1), len(l2))
                assert len(l1) > 0, "No gradient to test"
                n_gradient = 0
                for baseline_param, param in zip(l1, l2):
                    n_gradient += 1
                    torch.testing.assert_close(
                        baseline_param.grad,
                        param.grad,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=True,
                    )
                assert n_gradient > 0, "No gradient was checked"
            else:
                raise AssertionError(f"Unexpected type {type(baseline_result)}.")
        else:
            # forward only
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend="onnxrt",
                dynamic=False,
                fullgraph=fullgraph,
            )

            baseline_result = model(*args)
            result = compiled_model(*args)

            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol, equal_nan=True
                )

    def test_add(self):
        x = torch.zeros((10, 3), requires_grad=True, dtype=torch.float32)
        self.assertONNX(lambda x: x + x, x, onnx_export=inspect.currentframe().f_code.co_name)

    def test_index_put(self):
        x = torch.zeros((10, 3), requires_grad=True, dtype=torch.float32)
        indices = torch.arange(8, dtype=torch.int64).reshape((-1, 4))
        values = torch.arange(24, dtype=torch.float32).reshape((-1, 4, 3))

        # redondant test to make sure this expression is valid for torch
        assert x.index_put((indices,), values) is not None

        self.assertONNX(
            lambda x, indices, values: x.index_put((indices,), values),
            (x, indices, values),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_index_put_accumulate(self):
        x = torch.zeros((10, 3), requires_grad=True, dtype=torch.float32)
        indices = torch.arange(8, dtype=torch.int64).reshape((-1, 4))
        values = torch.arange(24, dtype=torch.float32).reshape((-1, 4, 3))

        # redondant test to make sure this expression is valid for torch
        assert x.index_put((indices,), values) is not None

        self.assertONNX(
            lambda x, indices, values: x.index_put((indices,), values, accumulate=True),
            (x, indices, values),
            onnx_export=inspect.currentframe().f_code.co_name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
