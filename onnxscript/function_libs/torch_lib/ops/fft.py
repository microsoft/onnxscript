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
    signal_size: INT64,
    inverse: bool = False,
) -> TFloat:
    """Normalize in forward or backward direction."""
    # Norm values defined in https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOps.cpp#L117-L131
    # Norm modes: https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOpsUtils.h#L15-L19
    # Modes:
    # 0: no normalization (backward)
    # 1: "ortho" - divide by 1/sqrt(signal_size) (ortho)
    # 2: divide by signal_size (forward)
    signal_size = op.CastLike(signal_size, self)
    if not inverse:
        # Forward normalization
        if normalization == 1:
            self = op.Div(self, op.Sqrt(signal_size))
        elif normalization == 2:
            self = op.Div(self, signal_size)
    else:
        # Backward normalization, accounting for op.DFT already dividing by signal_size
        if normalization == 0:
            self = op.Mul(self, signal_size)
        elif normalization == 1:
            self = op.Mul(self, op.Sqrt(signal_size))
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

    unsqueeze_first_dim = 0 in dim
    # 1. Add a new dimension for the end and batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0])
        dim = [d + 1 for d in dim]
    else:
        transformed = self

    for dimension in reversed(dim):
        transformed = op.DFT(transformed, axis=dimension, inverse=not forward, onesided=False)
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            op.Shape(transformed, start=dimension, end=dimension + 1),
            not forward,
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

    Complex to real inverse FFT. Assumes that input tensor is output of previous FFT operation.
    """
    if len(dim) != 1:
        raise NotImplementedError("Only one dimension is supported for inverse FFT")

    dimension = dim[0]
    unsqueeze_first_dim = dimension == 0
    # 1. Add a new dimension for batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0])
        dimension = 1
    else:
        transformed = self

    # Torch truncates/pads on the last dimension only. Typically, the only valid values that can be passed
    # into PyTorch are n or n//2+1, where n is self.shape[dim[-1]], but this is not always the case, so we
    # place no such restriction on the ONNX side.
    transformed = op.DFT(
        transformed,
        dft_length=last_dim_size,
        axis=dimension,
        inverse=True,
        onesided=False,
    )
    transformed = _fftn_onnx_normalization(
        transformed,
        normalization,
        op.Shape(transformed, start=dimension, end=dimension + 1),
        inverse=True,
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

    unsqueeze_first_dim = 0 in dim
    # 1. Add a new dimension for the end and batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0, -1])
        dim = [d + 1 for d in dim]
    else:
        transformed = op.Unsqueeze(self, axes=[-1])

    for idx, dimension in enumerate(reversed(dim)):
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            op.Shape(transformed, start=dimension, end=dimension + 1),
            inverse=False,
        )
        if idx > 0:
            transformed = op.DFT(transformed, axis=dimension, inverse=False, onesided=False)
        else:
            # Torch computes one-sided FFT on the last dimension only.
            transformed = op.DFT(transformed, axis=dimension, inverse=False, onesided=onesided)

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
