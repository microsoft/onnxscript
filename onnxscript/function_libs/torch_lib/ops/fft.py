# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `fft` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import INT64
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


def _fftn_onnx_normalization(
    self: TFloat,
    normalization: int,
    signal_size: TFloat,
) -> TFloat:
    """Normalize in forward direction."""
    # TODO: Make more efficient
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
    signal_size: TFloat,
) -> TFloat:
    """Normalize in backward direction, accounting for what op.DFT does."""
    # TODO: Make more efficient
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
    self_rank = len(self.shape)

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    unsqueeze_first_dim = 0 in dim
    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0])
        # Add 1 to account for the batch dimension when counting axes from the left
        dim = [dim_ + 1 if dim_ >= 0 else dim_ for dim_ in dim]
    else:
        transformed = self

    for dimension in reversed(dim):
        transformed = op.DFT(transformed, axis=dimension, inverse=not forward, onesided=False)
        if forward:
            transformed = _fftn_onnx_normalization(
                transformed,
                normalization,
                op.CastLike(self.shape[dimension - unsqueeze_first_dim], transformed),
            )
        else:
            transformed = _fftn_onnx_inverse_normalization(
                transformed,
                normalization,
                op.CastLike(self.shape[dimension - unsqueeze_first_dim], transformed),
            )

    if unsqueeze_first_dim:
        transformed = op.Squeeze(transformed, axes=[0])

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
    if len(dim) != 1:
        raise NotImplementedError("Only one dimension is supported for inverse FFT")

    self_rank = len(self.shape)

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    unsqueeze_first_dim = 0 in dim
    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0])
        # Add 1 to account for the batch dimension when counting axes from the left
        dim = [dim_ + 1 if dim_ >= 0 else dim_ for dim_ in dim]
    else:
        transformed = self

    for idx, dimension in enumerate(reversed(dim)):
        if idx > 0:
            transformed = op.DFT(transformed, axis=dimension, inverse=True, onesided=False)
        else:
            # Torch truncates/pads on the last dimension only. Typically, the only valid values that can be passed
            # into PyTorch are n or n//2+1, where n is self.shape[dim[-1]], but this is not always the case, so we
            # place no such restriction on the ONNX side.
            transformed = op.DFT(transformed, dft_length=last_dim_size, axis=dimension, inverse=True, onesided=False)
        transformed = _fftn_onnx_inverse_normalization(
            transformed,
            normalization,
            op.CastLike(op.Shape(transformed, start=dimension, end=dimension + 1), transformed),
        )

    if unsqueeze_first_dim:
        transformed = op.Squeeze(transformed, axes=[0])

    # Remove the imaginary part
    transformed = op.Slice(transformed, [0], [1], [-1])
    transformed = op.Squeeze(transformed, axes=[-1])

    return transformed


@torch_op("aten::_fft_r2c", trace_only=True)
def aten__fft_r2c(
    self: TFloat, dim: Sequence[int], normalization: int, onesided: bool
) -> TFloat:
    """_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor

    Real to complex forward FFT.
    """

    # No need to fill the imaginary part because ONNX DFT accepts real inputs
    # https://onnx.ai/onnx/operators/onnx__DFT.html#inputs

    self_rank = len(self.shape)

    # Add a new dimension at the end
    transformed = op.Unsqueeze(self, axes=[-1])

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    unsqueeze_first_dim = 0 in dim
    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(transformed, axes=[0])
        # Add 1 to account for the batch dimension when counting axes from the left
        dim = [dim_ + 1 if dim_ >= 0 else dim_ for dim_ in dim]

    for idx, dimension in enumerate(reversed(dim)):
        if idx > 0:
            transformed = op.DFT(transformed, axis=dimension, inverse=False, onesided=False)
        else:
            # Torch computes one-sided FFT on the last dimension only.
            transformed = op.DFT(transformed, axis=dimension, inverse=False, onesided=onesided)
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            op.CastLike(self.shape[dimension - unsqueeze_first_dim], transformed),
        )

    if unsqueeze_first_dim:
        transformed = op.Squeeze(transformed, axes=[0])

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
