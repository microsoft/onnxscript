# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO(pytorch/pytorch#129279): Migrate these tests to the PyTorch repo

import unittest

import torch
from torch.onnx._internal.exporter import _testing


class TorchLibe2eTest(unittest.TestCase):
    def test_investigate_one_particular_model(self):
        """This test can be used to investigate a particular issue."""
        red, include, stype = "amin", False, "int32"
        dtype = getattr(torch, stype)

        class Model(torch.nn.Module):
            def __init__(self, include, red):
                super().__init__()
                self.include = include
                self.red = red

            def forward(self, x, indices, updates):
                x = x.clone()
                return x.scatter_reduce(
                    0, indices, updates, self.red, include_self=self.include
                )

        model = Model(include, red)
        xs = (
            torch.tensor([[-2, 0, 2], [2, -2, 0]], dtype=dtype),
            torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
            torch.tensor([[-1, -1, -1], [-1, -1, -1]], dtype=dtype),
        )
        onnx_program = torch.onnx.export(model, xs, dynamo=True)
        _testing.assert_onnx_program(onnx_program)

    def test_pow_tensor_scalar_int_float(self):
        class PowModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x**0.5

        onnx_program = torch.onnx.export(
            PowModel(), (torch.tensor(2),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_pow_tensor_scalar_int_int(self):
        class PowModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x**2

        onnx_program = torch.onnx.export(
            PowModel(), (torch.tensor(2),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_pow_tensor_scalar_float16_int(self):
        class PowModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x**2

        onnx_program = torch.onnx.export(
            PowModel(), (torch.tensor(0.5, dtype=torch.float16),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_pow_tensor_scalar_float16_float(self):
        class PowModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x**0.5

        onnx_program = torch.onnx.export(
            PowModel(), (torch.tensor(0.5, dtype=torch.float16),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_repeat_interleave_integer_1(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.repeat_interleave(x, 3, dim=1)

        onnx_program = torch.onnx.export(
            Model(), (torch.randn(2, 3),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_repeat_interleave_integer_2(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.repeat_interleave(x, 3, dim=1)

        onnx_program = torch.onnx.export(
            Model(), (torch.randn(2, 3, 4),), dynamo=True, optimize=False
        )
        _testing.assert_onnx_program(onnx_program)

    def test_repeat_interleave_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind, dim=0)

        onnx_program = torch.onnx.export(
            Model(),
            (
                torch.arange(6, dtype=torch.float32).reshape((2, 3)),
                torch.tensor([1, 2], dtype=torch.int64),
            ),
            dynamo=True,
            optimize=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_repeat_interleave_tensor_none(self):
        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind)

        inputs = (
            torch.arange(4, dtype=torch.float32).reshape((2, 2)),
            torch.tensor([1, 2, 3, 2], dtype=torch.int64),
        )
        onnx_program = torch.onnx.export(
            Model(),
            inputs,
            dynamo=True,
            optimize=False,
        )
        onnx_program = torch.onnx.export(
            Model(),
            inputs,
            input_names=["x", "ind"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_repeat_interleave_symbolic_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.repeat_interleave(x, y.shape[1], dim=1) * torch.repeat_interleave(
                    y, x.shape[1], dim=1
                )

        inputs = (
            torch.arange(4, dtype=torch.float32).reshape((2, 2)),
            torch.arange(6, dtype=torch.float32).reshape((2, 3)),
        )
        onnx_program = torch.onnx.export(
            Model(),
            inputs,
            input_names=["x", "y"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_sdpa_with_bool_attn_mask(self):
        class ScaledDotProductAttention(torch.nn.Module):
            def forward(self, query, key, value, attn_mask):
                return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    query, key, value, attn_mask=attn_mask
                )

        model = ScaledDotProductAttention()
        attn_mask = torch.ones(2, 4, 8, 8).bool()  # boolean mask for attention
        attn_mask[0, 0, 0, :] = False  # masking an entire row (padding token)
        query = key = value = torch.randn(2, 4, 8, 16)

        onnx_program = torch.onnx.export(
            model,
            (query, key, value, attn_mask),
            input_names=["query", "key", "value", "attn_mask"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_dynamic_paddings(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                height = x.size(2)  # height is SymInt
                x = torch.nn.functional.pad(x, (0, 0, 0, height), mode="replicate")
                return x

        onnx_program = torch.onnx.export(
            Model(),
            (torch.rand(1, 1, 1, 1),),
            dynamo=True,
            dynamic_shapes=({2: torch.export.Dim("H")},),
        )
        _testing.assert_onnx_program(onnx_program)

    def test_enable_gqa_in_attention(self):
        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    q,
                    k,
                    v,
                    enable_gqa=True,
                )

        model = Model()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 2, 8, 16)
        value = torch.randn(2, 2, 8, 16)

        onnx_program = torch.onnx.export(
            model,
            (
                query,
                key,
                value,
            ),
            input_names=["query", "key", "value"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_bitwise_and_scalar(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x & 3

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor([1, 2, 3, 4, 5]),),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_avg_pool(self):
        class Model(torch.nn.Module):
            def forward(self, x2d, x3d, x4d, x5d):
                return (
                    torch.nn.functional.avg_pool1d(x2d, 2),
                    torch.nn.functional.avg_pool1d(x3d, 2),
                    torch.nn.functional.avg_pool2d(x3d, 2),
                    torch.nn.functional.avg_pool2d(x4d, 2),
                    torch.nn.functional.avg_pool3d(x4d, 2),
                    torch.nn.functional.avg_pool3d(x5d, 2),
                )

        x2d = torch.randn(10, 10)
        x3d = torch.randn(10, 10, 10)
        x4d = torch.randn(10, 10, 10, 10)
        x5d = torch.randn(10, 10, 10, 10, 10)
        onnx_program = torch.onnx.export(
            Model(),
            (x2d, x3d, x4d, x5d),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    unittest.main()
