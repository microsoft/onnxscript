import copy
import inspect
import itertools
import sys
import unittest
import warnings
from typing import Optional

import onnxruntime
import torch
import torch.nn as nn
import torch.onnx
from onnx.inliner import inline_local_functions


def make_aot_ort(
    dynamic: bool = False,
    ort_optimization_level: Optional[str] = None,
) -> tuple:
    code = inspect.getsource(torch.onnx._internal.onnxruntime)
    if "optimizer.optimize" not in code:
        raise unittest.SkipTest(
            f"torch=={torch.__version__!r} is not recent enough, "
            f"file {torch.onnx._internal.onnxruntime.__file__!r} "
            f"does not optimize the exported model."
        )

    ort_session_options = onnxruntime.SessionOptions()
    if ort_optimization_level is not None:
        assert hasattr(onnxruntime.GraphOptimizationLevel, ort_optimization_level), (
            f"Unexpected value {ort_optimization_level!r} for GraphOptimizationLevel, "
            f"expecting one of the values in {dir(onnxruntime.GraphOptimizationLevel)}"
        )
        ort_session_options.graph_optimization_level = getattr(
            onnxruntime.GraphOptimizationLevel, ort_optimization_level
        )

    export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic)

    def inline_function(*args, **kwargs):
        first_model_proto = args[0]

        next_model = inline_local_functions(first_model_proto)

        del first_model_proto.graph.node[:]
        del first_model_proto.functions[:]
        del first_model_proto.graph.initializer[:]
        del first_model_proto.opset_import[:]
        first_model_proto.graph.node.extend(next_model.graph.node)
        first_model_proto.functions.extend(next_model.functions)
        first_model_proto.graph.initializer.extend(next_model.graph.initializer)
        first_model_proto.opset_import.extend(next_model.opset_import)

        return first_model_proto

    options = torch.onnx._OrtBackendOptions(
        export_options=export_options,
        ort_session_options=ort_session_options,
        pre_ort_model_transforms=[inline_function],
    )

    return torch.onnx._OrtBackend(options=options)


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


class TestSimpleDort(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()  # pylint: disable=protected-access
        # hides the warnings
        # torch.onnx.dynamo_export only implements opset version 18 for now.
        # torch.library.impl_abstract was renamed to torch.library.register_fake.
        warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))

    def assertONNX(
        self,
        f,
        args,
        onnx_export: str,
        params=None,
        fullgraph: bool = True,
        atol=1e-6,
        rtol=1e-6,
        dynamic_axes=None,
    ):
        if sys.platform == "win32":
            raise unittest.SkipTest("Windows not supported yet.")
        assert isinstance(onnx_export, str), f"Export onnx is wrong for f={f}"
        if isinstance(args, torch.Tensor):
            args = [args]
        if params is None:
            params = ()
        if isinstance(f, torch.nn.Module):
            model = FuncModuleModule(f)
        else:
            model = FuncModule(f, params)
        model.eval()

        # forward/backward
        local_aot_ort = make_aot_ort(dynamic=dynamic_axes is not None)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=dynamic_axes is not None,
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
            raise TypeError(f"Unexpected type {type(baseline_result)}.")

    def test_acos(self):
        # This test is just to make sure it is working.
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.acos(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_batchnorm(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_batchnorm_onnx_irv4(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(2), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_batchnorm_1d(self):
        x = torch.ones(2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm1d(2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
