# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `fft` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import TensorType


def aten_fft_fft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_fft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_fftfreq(n: int, d: float = 1.0) -> TensorType:
    raise NotImplementedError()


def aten_fft_fftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_fftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_fft_hfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_hfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_hfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ifft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ifft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ifftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ifftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_fft_ihfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ihfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_ihfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_irfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_irfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_irfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_rfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_fft_rfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_fft_rfftfreq(n: int, d: float = 1.0) -> TensorType:
    raise NotImplementedError()


def aten_fft_rfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()
