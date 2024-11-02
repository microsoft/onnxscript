# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Compatible adapters implementing the TensorProtocol interface for various framework tensor types.

This module provides public classes that implement the :class:`onnxscript.ir.TensorProtocol`
interface for various tensor types from popular deep learning frameworks.

You can use these classes to create tensors and use them in the IR graph like any other tensor.

Example::
    import torch
    from onnxscript import ir

    # Create a PyTorch tensor
    torch_tensor = torch.tensor([1, 2, 3])

    # Wrap the PyTorch tensor in a TorchTensor object
    ir_tensor = ir.tensor_adapters.TorchTensor(torch_tensor)

    # Use the IR tensor in the graph
    attr = ir.AttrTensor("x", ir_tensor)
    print(attr)
"""

# pylint: disable=import-outside-toplevel

# NOTE: DO NOT import any framework-specific modules here in the global namespace.

from __future__ import annotations

__all__ = [
    "TorchTensor",
]

import ctypes
from typing import TYPE_CHECKING, Any

import numpy.typing as npt

from onnxscript import ir

if TYPE_CHECKING:
    import torch


class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None):
        # Pass the tensor as the raw data to ir.Tensor's constructor
        import torch

        _TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
            torch.bfloat16: ir.DataType.BFLOAT16,
            torch.bool: ir.DataType.BOOL,
            torch.complex128: ir.DataType.COMPLEX128,
            torch.complex64: ir.DataType.COMPLEX64,
            torch.float16: ir.DataType.FLOAT16,
            torch.float32: ir.DataType.FLOAT,
            torch.float64: ir.DataType.DOUBLE,
            torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
            torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
            torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
            torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
            torch.int16: ir.DataType.INT16,
            torch.int32: ir.DataType.INT32,
            torch.int64: ir.DataType.INT64,
            torch.int8: ir.DataType.INT8,
            torch.uint8: ir.DataType.UINT8,
            torch.uint16: ir.DataType.UINT16,
            torch.uint32: ir.DataType.UINT32,
            torch.uint64: ir.DataType.UINT64,
        }
        super().__init__(tensor, dtype=_TORCH_DTYPE_TO_ONNX[tensor.dtype], name=name)

    def numpy(self) -> npt.NDArray:
        import torch

        self.raw: torch.Tensor
        if self.dtype == ir.DataType.BFLOAT16:
            return self.raw.view(torch.uint16).numpy(force=True)
        if self.dtype in {
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E5M2FNUZ,
        }:
            # TODO: Use ml_dtypes
            return self.raw.view(torch.uint8).numpy(force=True)
        return self.raw.numpy(force=True)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> npt.NDArray:
        del copy  # Unused, but needed for the signature
        if dtype is None:
            return self.numpy()
        return self.numpy().__array__(dtype)

    def tobytes(self) -> bytes:
        # Implement tobytes to support native PyTorch types so we can use types like bloat16
        # Reading from memory directly is also more efficient because
        # it avoids copying to a NumPy array
        import torch._subclasses.fake_tensor

        with torch._subclasses.fake_tensor.unset_fake_temporarily():  # pylint: disable=protected-access
            # Disable any fake mode so calling detach() etc. will return a real tensor
            tensor = self.raw.detach().cpu().contiguous()

        if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):  # pylint: disable=protected-access
            raise TypeError(
                f"Cannot take content out from the FakeTensor ('{self.name}'). Please replace the tensor "
                "with a tensor backed by real data using ONNXProgram.apply_weights() "
                "or save the model without initializers by setting include_initializers=False."
            )

        return bytes(
            (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                tensor.data_ptr()
            )
        )
