# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

# TODO(pytorch/pytorch#129279): Migrate these tests to the PyTorch repo
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

    def test_dft_axis_promoted_from_attribute_to_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._fft_r2c(x, [0], normalization=1, onesided=True)  # pylint: disable=protected-access

        onnx_program = torch.onnx.export(
            Model(),
            (torch.randn(2, 3),),
            opset_version=20,
            dynamic_shapes=({0: "dim_x"},),
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_avg_pool(self):
        class Model(torch.nn.Module):
            def forward(self, x2d, x3d, x4d, x5d):
                return (
                    torch.nn.functional.avg_pool1d(x2d, 2),  # pylint: disable=not-callable
                    torch.nn.functional.avg_pool1d(x3d, 2),  # pylint: disable=not-callable
                    torch.nn.functional.avg_pool2d(x3d, 2),  # pylint: disable=not-callable
                    torch.nn.functional.avg_pool2d(x4d, 2),  # pylint: disable=not-callable
                    torch.nn.functional.avg_pool3d(x4d, 2),  # pylint: disable=not-callable
                    torch.nn.functional.avg_pool3d(x5d, 2),  # pylint: disable=not-callable
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

    def test_concat_with_empty_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, torch.tensor([]), x], dim=0)

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor([1, 2]),),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_concat_with_empty_tensor_single_element(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, torch.tensor([])], dim=1)

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor([[1, 2]]),),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_lstm_unidirectional(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=10, hidden_size=20, num_layers=1, batch_first=True
                )

            def forward(self, x):
                return self.lstm(x)

        model = LSTMModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_lstm_bidirectional(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=10,
                    hidden_size=20,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )

            def forward(self, x):
                return self.lstm(x)

        model = LSTMModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_lstm_multilayer(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=10, hidden_size=20, num_layers=3, batch_first=True
                )

            def forward(self, x):
                return self.lstm(x)

        model = LSTMModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_gru_unidirectional(self):
        class GRUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(
                    input_size=10, hidden_size=20, num_layers=1, batch_first=True
                )

            def forward(self, x):
                return self.gru(x)

        model = GRUModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_gru_bidirectional(self):
        class GRUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(
                    input_size=10,
                    hidden_size=20,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )

            def forward(self, x):
                return self.gru(x)

        model = GRUModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_gru_multilayer(self):
        class GRUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(
                    input_size=10, hidden_size=20, num_layers=3, batch_first=True
                )

            def forward(self, x):
                return self.gru(x)

        model = GRUModel()
        x = torch.randn(5, 3, 10)  # (batch, seq, input_size)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        _testing.assert_onnx_program(onnx_program)

    def test_aten_unique_consecutive(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int64)
        onnx_program = torch.onnx.export(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_unique_consecutive_int32(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int32)
        onnx_program = torch.onnx.export(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_unique_consecutive_return(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x, return_inverse=True, return_counts=True)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 3, 0, 0], dtype=torch.int64)
        onnx_program = torch.onnx.export(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            dynamo=True,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_stft_1(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.stft(x, n_fft=4, return_complex=True)

        x = torch.randn(4, 16, dtype=torch.float32)

        onnx_program = torch.onnx.export(
            Model(),
            (x,),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_stft_2(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.stft(x, n_fft=4, return_complex=False)

        x = torch.randn(4, 16, dtype=torch.float32)

        onnx_program = torch.onnx.export(
            Model(),
            (x,),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_stft_3(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                window = torch.ones(16, dtype=torch.float32)
                return torch.ops.aten.stft(x, n_fft=16, window=window, return_complex=False)

        x = torch.randn(100, dtype=torch.float32)

        onnx_program = torch.onnx.export(
            Model(),
            (x,),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_aten_stft_4(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.stft(
                    x,
                    n_fft=4,
                    hop_length=1,
                    win_length=4,
                    center=True,
                    onesided=True,
                    return_complex=True,
                )

        x = torch.randn(4, 16, dtype=torch.float32)

        onnx_program = torch.onnx.export(
            Model(),
            (x,),
            dynamo=True,
            verbose=False,
        )
        _testing.assert_onnx_program(onnx_program)

    def test_my_index_put(self):
        def test(x_shape, index_list, update_shape, testname):
            with self.subTest(testname=testname):
                indices = [
                    (torch.tensor(index, dtype=torch.int64) if index is not None else None)
                    for index in index_list
                ]

                class Model(torch.nn.Module):
                    def forward(self, x, update):
                        return torch.ops.aten.index_put(x, indices, update, accumulate=True)

                x = torch.zeros(x_shape, dtype=torch.float32)
                update = torch.randn(update_shape, dtype=torch.float32)
                onnx_program = torch.onnx.export(
                    Model(),
                    (x, update),
                    input_names=["x", "update"],
                    output_names=["output"],
                    opset_version=18,
                    dynamo=True,
                )
                _testing.assert_onnx_program(onnx_program)

        # Test cases
        shape_6x6x6 = (6, 6, 6)

        # Multiple advanced indices, all 1D tensors.
        # Non-contiguous advanced indices: updates must be broadcastable to (2, 6)
        base = "non_contiguous_non_broadcast_indices_"
        test(shape_6x6x6, [[0, 1], None, [2, 3]], (2, 6), base + "no_value_broadcast")
        test(shape_6x6x6, [[0, 1], None, [2, 3]], (2, 1), base + "expand_dim2")
        test(shape_6x6x6, [[0, 1], None, [2, 3]], (1, 6), base + "expand_dim1")
        test(shape_6x6x6, [[0, 1], None, [2, 3]], (6,), base + "new_dim1")
        test(shape_6x6x6, [[0, 1], None, [2, 3]], (), base + "scalar")

        # Contiguous advanced indices versions of above tests: updates must be broadcastable to (6, 2)
        base = "contiguous_non_broadcast_indices_"
        test(shape_6x6x6, [None, [0, 1], [2, 3]], (6, 2), base + "no_value_broadcast")
        test(shape_6x6x6, [None, [0, 1], [2, 3]], (6, 1), base + "expand_dim2")
        test(shape_6x6x6, [None, [0, 1], [2, 3]], (1, 2), base + "expand_dim1")
        test(shape_6x6x6, [None, [0, 1], [2, 3]], (2,), base + "new_dim1")
        test(shape_6x6x6, [None, [0, 1], [2, 3]], (), base + "scalar")

        # Multiple advanced indices, with broadcasting among indices.
        # Contiguous advanced indices:
        # This produces index tuples [(0,2), (0, 3), (1,2), (1,3)] in shape (2,2)
        # The update values must be broadcastable to (6,2,2)
        base = "contiguous_broadcast_indices_"
        test(shape_6x6x6, [None, [[0], [1]], [2, 3]], (6, 2, 2), base + "no_value_broadcast")
        test(shape_6x6x6, [None, [[0], [1]], [2, 3]], (6, 1, 1), base + "expand_dim2_dim3")
        test(shape_6x6x6, [None, [[0], [1]], [2, 3]], (2,), base + "extend_dim1_dim2")

        # Non-contiguous advanced indices versions of above tests:
        # Here, update values must be broadcastable to (2,2,6)
        base = "non_contiguous_broadcast_indices_"
        test(shape_6x6x6, [[[0], [1]], None, [2, 3]], (2, 2, 6), base + "no_value_broadcast")
        test(shape_6x6x6, [[[0], [1]], None, [2, 3]], (1, 1, 6), base + "expand_dim1_dim2")
        test(shape_6x6x6, [[[0], [1]], None, [2, 3]], (6,), base + "extend_dim1_dim2")
