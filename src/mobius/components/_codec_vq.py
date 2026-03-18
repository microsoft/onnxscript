# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Vector quantization components for the Qwen3-TTS codec tokenizer.

Implements the SplitResidualVectorQuantizer decode path used by
``Qwen3TTSTokenizerV2Decoder`` to convert discrete audio codes into
continuous embeddings.

At inference time the VQ codebooks are simple embedding lookups —
``preprocess_weights`` precomputes ``embedding_sum / cluster_usage``
so no runtime division is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import INT64_MAX, Embedding

if TYPE_CHECKING:
    import onnx_ir as ir


class EuclideanCodebook(nn.Module):
    """Single VQ codebook: integer codes → embedding vectors.

    After ``preprocess_weights`` precomputes ``embedding_sum / cluster_usage``,
    this is a simple Gather on the resulting embedding table.

    HF class: ``EuclideanCodebook`` in qwen_tts.

    Parameters:
        codebook_size: Number of entries (e.g. 2048).
        dim: Embedding dimension (e.g. 256).
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        # After precomputation this holds the actual embeddings
        self.embedding = Embedding(codebook_size, dim)

    def forward(self, op: builder.OpBuilder, codes: ir.Value):
        """Lookup codebook entries.

        Args:
            codes: (B, T) int64 indices.

        Returns:
            quantized: (B, T, dim) float embeddings.
        """
        return self.embedding(op, codes)


class VectorQuantization(nn.Module):
    """Single-layer VQ: codebook lookup (no per-layer projection).

    HF class: ``VectorQuantization`` in qwen_tts.

    Parameters:
        codebook_size: Number of codebook entries.
        dim: Codebook dimension (codebook_dim // 2).
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self._codebook = EuclideanCodebook(codebook_size, dim)

    def forward(self, op: builder.OpBuilder, codes: ir.Value):
        """Decode codes through one VQ layer.

        Args:
            codes: (B, T) int64 indices.

        Returns:
            quantized: (B, dim, T) transposed embeddings.
        """
        # (B, T) → (B, T, dim)
        quantized = self._codebook(op, codes)
        # Transpose to channels-first: (B, dim, T)
        return op.Transpose(quantized, perm=[0, 2, 1])


class ResidualVectorQuantization(nn.Module):
    """Multi-layer residual VQ: sum decoded outputs across layers.

    HF class: ``ResidualVectorQuantization`` in qwen_tts.

    Parameters:
        num_layers: Number of quantizer layers.
        codebook_size: Entries per codebook.
        dim: Per-layer codebook dimension.
    """

    def __init__(self, num_layers: int, codebook_size: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(codebook_size, dim) for _ in range(num_layers)]
        )

    def forward(self, op: builder.OpBuilder, codes: ir.Value):
        """Decode and sum across quantizer layers.

        Args:
            codes: (B, K, T) where K = num_layers.

        Returns:
            quantized: (B, dim, T) summed embeddings.
        """
        # Split along quantizer axis: K tensors of (B, 1, T)
        # Then squeeze to (B, T) for each layer
        quantized = None
        for i, layer in enumerate(self.layers):
            # Gather layer i with scalar index: (B, K, T) → (B, T)
            # Scalar index removes the axis, no squeeze needed
            layer_codes = op.Gather(codes, op.Constant(value_int=i), axis=1)
            # (B, dim, T)
            decoded = layer(op, layer_codes)
            if quantized is None:
                quantized = decoded
            else:
                quantized = op.Add(quantized, decoded)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    """RVQ with input/output 1x1 convolution projections.

    HF class: ``ResidualVectorQuantizer`` in qwen_tts.

    Parameters:
        num_quantizers: Number of quantizer layers.
        codebook_size: Entries per codebook.
        dim: Internal codebook dimension (codebook_dim // 2).
        input_dim: Input dimension for input_proj.
        output_dim: Output dimension for output_proj.
    """

    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        dim: int,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        # 1x1 conv projections — use _Conv1dParams to get .weight suffix
        # matching HF names: rvq_first.input_proj.weight
        self.input_proj = _Conv1dProjParams(input_dim, dim)
        self.output_proj = _Conv1dProjParams(dim, output_dim)
        self.vq = ResidualVectorQuantization(num_quantizers, codebook_size, dim)

    def forward(self, op: builder.OpBuilder, codes: ir.Value):
        """Decode codes through projected RVQ.

        Args:
            codes: (B, K, T) where K = num_quantizers.

        Returns:
            quantized: (B, output_dim, T).
        """
        # Decode: (B, K, T) → (B, dim, T)
        quantized = self.vq(op, codes)
        # Output projection: 1x1 conv (B, dim, T) → (B, output_dim, T)
        quantized = self.output_proj(op, quantized)
        return quantized


class _Conv1dProjParams(nn.Module):
    """1x1 convolution projection — called to register parameters.

    Produces HF-compatible weight names like
    ``input_proj.weight`` and ``output_proj.weight``.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter([out_dim, in_dim, 1])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[1],
            strides=[1],
            pads=[0, 0],
        )


class SplitResidualVectorQuantizer(nn.Module):
    """Split RVQ: semantic (1 layer) + acoustic (N-1 layers).

    The first quantizer handles semantic codes, the rest handle
    acoustic detail. Outputs are summed.

    HF class: ``SplitResidualVectorQuantizer`` in qwen_tts.

    Parameters:
        num_quantizers: Total number of quantizers (e.g. 16).
        codebook_size: Entries per codebook (e.g. 2048).
        codebook_dim: Full codebook dimension (e.g. 512).
            Internal VQ uses codebook_dim // 2.
    """

    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        dim = codebook_dim // 2  # Internal VQ dimension
        self.rvq_first = ResidualVectorQuantizer(
            num_quantizers=1,
            codebook_size=codebook_size,
            dim=dim,
            input_dim=codebook_dim,
            output_dim=codebook_dim,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            num_quantizers=num_quantizers - 1,
            codebook_size=codebook_size,
            dim=dim,
            input_dim=codebook_dim,
            output_dim=codebook_dim,
        )

    def forward(self, op: builder.OpBuilder, codes: ir.Value):
        """Decode all code groups and sum.

        Args:
            codes: (B, num_quantizers, T) int64.

        Returns:
            quantized: (B, codebook_dim, T) float.
        """
        # Split: first quantizer vs rest
        # codes[:, :1, :] → (B, 1, T)
        first_codes = op.Slice(
            codes,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[1]),
        )
        # codes[:, 1:, :] → (B, num_quantizers-1, T)
        # INT64_MAX as Slice end to mean "all remaining elements"
        rest_codes = op.Slice(
            codes,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[INT64_MAX]),
            op.Constant(value_ints=[1]),
        )

        # Decode each group: (B, codebook_dim, T)
        quantized_first = self.rvq_first(op, first_codes)
        quantized_rest = self.rvq_rest(op, rest_codes)

        # Sum: (B, codebook_dim, T)
        return op.Add(quantized_first, quantized_rest)
