# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `fft` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from onnxscript import INT64
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


# def _compute_signal_size(signal: TFloat, dims: Sequence[int], last_dim_size: Optional[INT64] = None) -> INT64:
#     if last_dim_size is not None:
#         all_other_dims = dims[:-1]
#         if all_other_dims:
#             signal_size = op.ReduceProd(signal, axes=all_other_dims, keepdims=False)
#             signal_size = op.Mul(signal_size, last_dim_size)
#         else:
#             signal_size = last_dim_size
#     else:
#         signal_size = op.ReduceProd(signal, axes=dims, keepdims=False)
#     return signal_size


# def _fftn_ortho_normalization(
#     self: TFloat,
#     dims: Sequence[int],
#     forward: bool,
#     onesided: bool,
#     last_dim_size: Optional[INT64] = None,
# ) -> TFloat:
#     transformed = self

#     signal_size = _compute_signal_size(self, dims, last_dim_size)

#     for dim in dims[:-1]:
#         transformed = op.DFT(transformed, axis=dim, onesided=False)

#     # Torch computes one-sided FFT on the last dimension only.
#     if onesided:
#         transformed = op.DFT(transformed, axis=dims[-1], onesided=True)
#         # TODO: Update signal_size for one-sided FFT
#     elif last_dim_size is not None:
#         transformed = op.DFT(
#             transformed, last_dim_size, axis=dims[-1], onesided=True
#         )
#     else:
#         transformed = op.DFT(transformed, axis=dims[-1], onesided=False)


def _fftn_onnx_normalization(
    self: TFloat,
    normalization: int,
    signal_size: INT64,
) -> TFloat:
    """
    """
    # TODO: Make more efficient - there should be a faster way to recalculate everything
    # Norm values defined in https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOps.cpp#L117-L131
    # Norm modes: https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOpsUtils.h#L15-L19
    # Modes:
    # 0: no normalization (backward)
    # 1: "ortho" - divide by 1/sqrt(signal_size) (ortho)
    # 2: divide by signal_size (forward)
    if normalization == 1:
        self = op.Div(self, op.Sqrt(signal_size))
    elif normalization == 2:
        self = op.Div(self, signal_size)
    return self

def _fftn_onnx_inverse_normalization(
    self: TFloat,
    normalization: int,
    signal_size: INT64,
) -> TFloat:
    """
    """
    # TODO: Make more efficient - there should be a faster way to recalculate everything
    # Norm values defined in https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOps.cpp#L117-L131
    # Norm modes: https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOpsUtils.h#L15-L19
    # Modes:
    # 0: no normalization (backward)
    # 1: "ortho" - divide by 1/sqrt(signal_size) (ortho)
    # 2: divide by signal_size (forward)
    if normalization == 1:
        self = op.Mul(self, op.Sqrt(signal_size))
    elif normalization == 0:
        self = op.Mul(self, signal_size)
    return self

# def _fftn_onnx(
#     self: TFloat,
#     dims: Sequence[int],
#     normalization: int,
#     forward: bool,
#     onesided: bool,
#     last_dim_size: Optional[INT64] = None,
# ) -> TFloat:
#     """Standard complex to complex or real to complex FFT (forward or backward).

#     This is a private shared function for implementing the various FFT functions.

#     Args:
#         self: The input tensor.
#         dims: The dimensions to apply FFT.
#         normalization: The normalization mode.
#         forward: Whether to compute forward FFT or backward FFT.
#         onesided: Whether to compute the one-sided FFT, which retains only the
#             positive frequencies.
#         last_dim_size: The size of the last specified dimension.

#     Returns:
#         The transformed tensor.
#     """
#     # NOTE: SymInt dim is not support because DFT-17 needs a static axis

#     # If taking FFT along the 0-th dimension: Since
#     # the 0-th dimension in ONNX DFT-17 is the batch dimension (cannot take DFT over),
#     # we need to add a new dimension at the beginning to represent the batch dimension.
#     unsqueeze_first_dim = 0 in dims
#     if unsqueeze_first_dim:
#         transformed = op.Unsqueeze(self, axes=[0])
#         # Add 1 to account for the batch dimension when counting axes from the left
#         dims = [dim_ + 1 if dim_ >= 0 else dim_ for dim_ in dims]
#     else:
#         transformed = self

#     # Select inverse mode for ONNX based on the norm mode and forward/backward mode.
#     # In ONNX the only difference between inverse=True/False is the 1/n normalization applied.
#     #
#     # If normalization is 1/n and we are in backward mode, we use the inverse
#     # mode in ONNX to get the 1/n normalization.
#     inverse = normalization == 2 and not forward
#     ortho = normalization == 1

#     for dim in dims[:-1]:
#         transformed = op.DFT(transformed, axis=dim, inverse=inverse, onesided=False)

#     # Torch computes one-sided FFT on the last dimension only.
#     if onesided:
#         transformed = op.DFT(transformed, axis=dims[-1], inverse=inverse, onesided=True)
#     elif last_dim_size is not None:
#         transformed = op.DFT(
#             transformed, last_dim_size, axis=dims[-1], inverse=inverse, onesided=False
#         )
#     else:
#         transformed = op.DFT(transformed, axis=dims[-1], inverse=inverse, onesided=False)

#     if ortho or inverse:
#         normalized = _fftn_onnx_normalization(
#             transformed, ortho, dims, last_dim_size=last_dim_size
#         )
#     else:
#         normalized = transformed
#     # TODO: Merge to normalization mode and ONNX inverse mode
#     # Be sure to normalize before squeezing the batch dimension, because dims would
#     # have been shifted by 1 if the batch dimension was added.
#     if unsqueeze_first_dim:
#         # Remove the batch dimension
#         normalized = op.Squeeze(normalized, axes=[0])

#     return normalized


@torch_op("aten::_fft_c2c", trace_only=True, complex=True)
def aten__fft_c2c(
    self: TFloat, dim: Sequence[int], normalization: int, forward: bool
) -> TFloat:
    """_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor

    Standard complex to complex FFT (forward or backward).
    """

    # NOTE: SymInt dim is not supported because DFT-17 needs a static axis

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    assert(dim[2] in dim == 2, "Unexpected input size")

    signal = self
    self_rank = len(self.shape)
    signal_size = op.Size(signal)

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    transformed = signal

    for dimension in reversed(dim):
        transformed = op.DFT(transformed, axis=dimension, inverse=not forward, onesided=False)
        if forward:
            transformed = _fftn_onnx_normalization(transformed, normalization, signal_size)
        else:
            transformed = _fftn_onnx_inverse_normalization(transformed, normalization, signal_size)

    # Unsure if output format is correct
    return transformed


@torch_op("aten::_fft_c2r", trace_only=True, complex=True)
def aten__fft_c2r(
    self: TFloat,
    dim: Sequence[int],
    normalization: int,
    last_dim_size: INT64,
) -> TFloat:
    """_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor

    Complex to real inverse FFT.
    """
    assert(dim[2] in dim == 2, "Unexpected input size")

    signal = self
    self_rank = len(self.shape)
    signal_size = op.Size(signal)

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    transformed = signal
    for dimension in reversed(dim):
        transformed = op.DFT(transformed, axis=dimension, inverse=True, onesided=False)
        transformed = _fftn_onnx_inverse_normalization(transformed, normalization, signal_size)

    # Unsure if output format is correct
    transformed = op.Squeeze(transformed, axes=[-1])

    if transformed.shape[-1] < last_dim_size:
        pads = [0, last_dim_size - transformed.shape[-1]]
        mode = 'constant'
        constant_value = 0.0
        transformed = op.Pad(mode=mode, data=transformed, pads=pads, constant_value=constant_value, axes=[-1])
    elif transformed.shape[-1] > last_dim_size:
        starts = [0]*(self_rank-1)
        ends = list(self.shape)
        ends[-1] = last_dim_size
        transformed = op.Slice(data=transformed, starts=starts, ends=ends)

    return transformed


@torch_op("aten::_fft_r2c", trace_only=True)
def aten__fft_r2c(
    self: TFloat, dim: Sequence[int], normalization: int, onesided: bool
) -> TFloat:
    """_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor

    Real to complex forward FFT.
    """

    # Add a new dimension at the end
    signal = op.Unsqueeze(self, axes=[-1])
    # No need to fill the imaginary part because ONNX DFT accepts real inputs
    # https://onnx.ai/onnx/operators/onnx__DFT.html#inputs

    self_rank = len(self.shape)
    signal_size = op.Size(signal)

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    # Torch computes one-sided FFT on the last dimension only.
    transformed = op.DFT(signal, axis=dim[-1], inverse=False, onesided=onesided)
    transformed = _fftn_onnx_normalization(transformed, normalization, signal_size)

    for dimension in reversed(dim[:-1]):
        transformed = op.DFT(transformed, axis=dimension, inverse=False, onesided=False)
        transformed = _fftn_onnx_normalization(transformed, normalization, signal_size)

    # Unsure if output format is correct
    return transformed

def aten_fft_fft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_fft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_fftfreq(n: int, d: float = 1.0) -> TensorType:
    """fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_fftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_fftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    """fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_hfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_hfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_hfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_hfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_hfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ifft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ifft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ifftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ifftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    """fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ihfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ihfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ihfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_ihfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ihfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_irfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_irfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_irfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_rfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_rfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_rfftfreq(n: int, d: float = 1.0) -> TensorType:
    """fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_fft_rfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError()
