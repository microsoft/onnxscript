# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the tensor_adapters module."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import parameterized
import torch

from onnxscript.ir import tensor_adapters


def skip_if_no(module_name: str):
    """Decorator to skip a test if a module is not installed."""
    if importlib.util.find_spec(module_name) is None:
        return unittest.skip(f"{module_name} not installed")
    return lambda func: func


@skip_if_no("torch")
class TorchTensorTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (torch.bfloat16, np.uint16),
            (torch.bool, np.bool_),
            (torch.complex128, np.complex128),
            (torch.complex64, np.complex64),
            (torch.float16, np.float16),
            (torch.float32, np.float32),
            (torch.float64, np.float64),
            (torch.float8_e4m3fn, np.uint8),
            (torch.float8_e4m3fnuz, np.uint8),
            (torch.float8_e5m2, np.uint8),
            (torch.float8_e5m2fnuz, np.uint8),
            (torch.int16, np.int16),
            (torch.int32, np.int32),
            (torch.int64, np.int64),
            (torch.int8, np.int8),
            (torch.uint16, np.uint16),
            (torch.uint32, np.uint32),
            (torch.uint64, np.uint64),
            (torch.uint8, np.uint8),
        ],
    )
    def test_numpy_returns_correct_dtype(self, dtype: torch.dtype, np_dtype):
        tensor = tensor_adapters.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.numpy().dtype, np_dtype)
        self.assertEqual(tensor.__array__().dtype, np_dtype)
        self.assertEqual(np.array(tensor).dtype, np_dtype)

    @parameterized.parameterized.expand(
        [
            (torch.bfloat16),
            (torch.bool),
            (torch.complex128),
            (torch.complex64),
            (torch.float16),
            (torch.float32),
            (torch.float64),
            (torch.float8_e4m3fn),
            (torch.float8_e4m3fnuz),
            (torch.float8_e5m2),
            (torch.float8_e5m2fnuz),
            (torch.int16),
            (torch.int32),
            (torch.int64),
            (torch.int8),
            (torch.uint16),
            (torch.uint32),
            (torch.uint64),
            (torch.uint8),
        ],
    )
    def test_tobytes(self, dtype: torch.dtype):
        tensor = tensor_adapters.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.tobytes(), tensor.numpy().tobytes())


    def test_array_conversion_with_dtype(self):
        tensor = tensor_adapters.TorchTensor(torch.tensor([1.0], dtype=torch.float32))
        np_array = tensor.__array__(dtype=np.float64)
        self.assertEqual(np_array.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
