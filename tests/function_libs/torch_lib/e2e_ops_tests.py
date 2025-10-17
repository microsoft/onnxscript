# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO(pytorch/pytorch#129279): Migrate these tests to the PyTorch repo

import unittest

import numpy as np
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

    def test_index_put_dynamic(self):
        for dimension in [3, 4, 2]:
            with self.subTest(dimension=dimension):

                class Model(torch.nn.Module):
                    def __init__(self, dimension):
                        super().__init__()
                        self.params = torch.zeros(
                            (4, 5)
                            if dimension == 2
                            else ((2, 4, 5) if dimension == 3 else (1, 1, 4, 5))
                        )
                        self.dimension = dimension

                    def forward(self, update, index1, index2):
                        copy = self.params.clone()
                        if self.dimension == 2:
                            copy[index1, index2] = update
                        elif self.dimension == 3:
                            copy[:, index1, index2] = update
                        else:
                            copy[:, :, index1, index2] = update
                        return copy

                update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
                index1 = torch.tensor([1, 2], dtype=torch.int64)
                index2 = torch.tensor([3, 4], dtype=torch.int64)
                feeds = dict(zip(["update", "index1", "index2"], (update, index1, index2)))
                onnx_program = torch.onnx.export(
                    Model(dimension),
                    tuple(feeds.values()),
                    input_names=["update", "index1", "index2"],
                    output_names=["output"],
                    opset_version=18,
                    dynamo=True,
                    dynamic_shapes={
                        "update": {0: "dn"},
                        "index1": {0: "dn"},
                        "index2": {0: "dn"},
                    },
                )
                _testing.assert_onnx_program(onnx_program)

    def test_index_put_scatter_nd(self):
        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                x = x.clone()
                return torch.ops.aten.index_put(x, [None, index, None], update)

        shape = (2, 3, 2)
        N = int(np.prod(shape))
        x = torch.arange(N, dtype=torch.float32).reshape(shape)
        update = (torch.arange(N, dtype=torch.float32).reshape(shape) + 1) * 100
        index = ((torch.arange(shape[-2])).to(torch.int64) + 1) % shape[-2]

        feeds = dict(zip(["x", "index", "update"], (x, index, update)))
        onnx_program = torch.onnx.export(
            Model(),
            tuple(feeds.values()),
            input_names=["x", "index", "update"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            dynamic_shapes=({0: "a", 1: "b", 2: "c"}, {0: "d"}, {0: "e", 1: "f", 2: "g"}),
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


if __name__ == "__main__":
    unittest.main()
