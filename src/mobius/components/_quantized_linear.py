# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Quantized linear layer using MatMulNBits (com.microsoft domain)."""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

# MatMulNBits packs weights into uint8 blobs.  The packed shape depends
# on bits and block_size:
#   packed_weights: [N, n_blocks, blob_size]  (uint8)
#     where n_blocks = ceil(K / block_size)
#           blob_size = block_size * bits / 8
#   scales:         [N, n_blocks]             (float16/float32)
#   zero_points:    [N, ceil(n_blocks/2)]    (uint8, optional, 4-bit packed)

_MICROSOFT_DOMAIN = "com.microsoft"


class QuantizedLinear(nn.Module):
    """Linear layer backed by the MatMulNBits custom op.

    Replaces a standard ``Linear`` for weight-only quantized models
    (GPTQ, AWQ, etc.).  The op performs::

        y = x @ dequantize(packed_weights, scales, zero_points)^T

    entirely inside a single fused kernel at inference time.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bits: Quantization bit-width (4 or 8).
        block_size: Number of elements per quantization group.
        has_zero_point: Whether asymmetric zero-point is used.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 32,
        has_zero_point: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        if block_size < 16 or (block_size & (block_size - 1)):
            raise ValueError(f"block_size must be a power of 2 >= 16, got {block_size}")

        self._bits = bits
        self._block_size = block_size
        self._k = in_features
        self._n = out_features

        n_blocks = math.ceil(in_features / block_size)
        blob_size = block_size * bits // 8

        # Packed quantized weight tensor (uint8)
        self.weight = nn.Parameter(
            [out_features, n_blocks, blob_size],
            dtype=ir.DataType.UINT8,
        )
        # Per-block scale factors
        self.scales = nn.Parameter([out_features, n_blocks])
        # Optional per-block zero points (asymmetric quantization).
        # For 4-bit, two zero-point values are packed per byte →
        # the last dimension is ceil(n_blocks / 2).
        zp_dim = math.ceil(n_blocks / 2) if bits == 4 else n_blocks
        self.zero_points = (
            nn.Parameter(
                [out_features, zp_dim],
                dtype=ir.DataType.UINT8,
            )
            if has_zero_point
            else None
        )
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Compute quantized matmul: y = x @ dequant(W).

        Args:
            op: ONNX op builder.
            x: Input tensor of shape ``[*, K]``.

        Returns:
            Output tensor of shape ``[*, N]``.
        """
        inputs: list[ir.Value | None] = [x, self.weight, self.scales]
        if self.zero_points is not None:
            inputs.append(self.zero_points)

        result = op.MatMulNBits(
            *inputs,
            K=self._k,
            N=self._n,
            bits=self._bits,
            block_size=self._block_size,
            _domain=_MICROSOFT_DOMAIN,
        )
        if self.bias is not None:
            result = op.Add(result, self.bias)
        return result


def make_quantized_linear_factory(
    bits: int = 4,
    block_size: int = 32,
    has_zero_point: bool = False,
) -> type:
    """Create a QuantizedLinear factory compatible with the linear_class pattern.

    Returns a class whose ``__init__(in_features, out_features, bias=True)``
    signature matches ``Linear`` so it can be injected via ``linear_class``
    in ``DecoderLayer``, ``Attention``, and ``MLP``.

    Args:
        bits: Quantization bit-width (typically 4).
        block_size: Number of elements per quantization group.
        has_zero_point: Whether to include zero-point parameters.

    Returns:
        A class that constructs QuantizedLinear instances.
    """

    class _Factory(QuantizedLinear):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
        ):
            super().__init__(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                bits=bits,
                block_size=block_size,
                has_zero_point=has_zero_point,
            )

    _Factory.__name__ = "QuantizedLinear"
    _Factory.__qualname__ = "QuantizedLinear"
    return _Factory
