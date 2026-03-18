# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Repack GGUF quantized blocks into ORT MatMulNBits format.

Converts raw GGUF block data for Q4_0, Q4_1, and Q8_0 quantization
types into the (weight, scales, zero_points) tensors expected by the
``com.microsoft.MatMulNBits`` operator.

GGUF block layouts (32 elements per block):
    Q4_0 (18 bytes): [fp16 scale][16B packed nibbles]
    Q4_1 (20 bytes): [fp16 scale][fp16 min][16B packed nibbles]
    Q8_0 (34 bytes): [fp16 scale][32B int8 values]

MatMulNBits expects:
    weight:      [N, n_blocks, blob_size] uint8
    scales:      [N, n_blocks]            float16
    zero_points: [N, zp_dim]             uint8 (nibble-packed for 4-bit)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 32

# GGUF quantization type IDs (from gguf.GGMLQuantizationType enum)
_GGUF_Q4_0 = 2
_GGUF_Q4_1 = 3
_GGUF_Q8_0 = 8
_GGUF_Q4_K = 12

# Block byte sizes per GGUF type
_BLOCK_BYTES = {
    _GGUF_Q4_0: 18,  # 2B scale + 16B quants
    _GGUF_Q4_1: 20,  # 2B scale + 2B min + 16B quants
    _GGUF_Q8_0: 34,  # 2B scale + 32B int8 values
    _GGUF_Q4_K: 144,  # 2B d + 2B dmin + 12B scales + 128B quants
}

# Elements per GGUF block. Q4_K uses 256-element "super-blocks"
# that decompose into 8 sub-blocks of 32 for MatMulNBits.
_GGUF_BLOCK_ELEMENTS = {
    _GGUF_Q4_0: 32,
    _GGUF_Q4_1: 32,
    _GGUF_Q8_0: 32,
    _GGUF_Q4_K: 256,
}

_SUPPORTED_TYPES = frozenset(_BLOCK_BYTES.keys())


@dataclass
class RepackedTensor:
    """MatMulNBits-compatible representation of a quantized weight.

    Attributes:
        weight: Packed uint8 blob, shape ``[N, n_blocks, blob_size]``.
        scales: Per-block scale factors, float16, shape ``[N, n_blocks]``.
        zero_points: Per-block zero points, uint8, or ``None``.
            For 4-bit: nibble-packed, shape ``[N, ceil(n_blocks/2)]``.
            For 8-bit: shape ``[N, n_blocks]``.
        block_size: Elements per quantization block (always 32 for GGUF).
        bits: Quantization bit-width (4 or 8).
    """

    weight: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray | None
    block_size: int
    bits: int


def can_repack(gguf_type: int) -> bool:
    """Return True if the GGUF type can be repacked to MatMulNBits."""
    return gguf_type in _SUPPORTED_TYPES


def repack_gguf_tensor(
    raw_data: np.ndarray,
    gguf_type: int,
    shape: tuple[int, ...],
) -> RepackedTensor:
    """Repack a GGUF quantized tensor into MatMulNBits format.

    Args:
        raw_data: Raw bytes as a uint8 numpy array (flat).
        gguf_type: GGUF quantization type ID (e.g. 2 for Q4_0).
        shape: Logical weight shape ``(N, K)`` where N = out_features,
            K = in_features.  GGUF tensors are typically stored as
            ``(N, K)`` with blocks laid out row-by-row.

    Returns:
        A ``RepackedTensor`` with MatMulNBits-compatible arrays.

    Raises:
        ValueError: If the GGUF type is unsupported or data size is wrong.
    """
    if gguf_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported GGUF type {gguf_type}. Supported: {sorted(_SUPPORTED_TYPES)}"
        )

    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape (N, K), got {shape}")

    n_out, k_in = shape
    block_bytes = _BLOCK_BYTES[gguf_type]
    gguf_block_elems = _GGUF_BLOCK_ELEMENTS[gguf_type]
    n_blocks_per_row = math.ceil(k_in / gguf_block_elems)
    total_blocks = n_out * n_blocks_per_row
    expected_bytes = total_blocks * block_bytes

    if raw_data.size != expected_bytes:
        raise ValueError(
            f"Data size mismatch: got {raw_data.size} bytes, "
            f"expected {expected_bytes} for shape {shape} "
            f"with {total_blocks} blocks x {block_bytes} bytes"
        )

    # Reshape into (total_blocks, block_bytes) then dispatch
    blocks = raw_data.reshape(total_blocks, block_bytes)

    if gguf_type == _GGUF_Q4_0:
        return _repack_q4_0(blocks, n_out, n_blocks_per_row)
    elif gguf_type == _GGUF_Q4_1:
        return _repack_q4_1(blocks, n_out, n_blocks_per_row)
    elif gguf_type == _GGUF_Q4_K:
        return _repack_q4_k(blocks, n_out, n_blocks_per_row)
    else:
        return _repack_q8_0(blocks, n_out, n_blocks_per_row)


def _reorder_nibbles_gguf_to_ort(
    gguf_packed: np.ndarray,
) -> np.ndarray:
    """Convert GGUF nibble ordering to MatMulNBits ordering.

    GGUF packs 32 elements into 16 bytes as:
        byte i: low nibble = element[i], high nibble = element[i+16]

    MatMulNBits packs as:
        byte j: low nibble = element[2j], high nibble = element[2j+1]

    Args:
        gguf_packed: uint8 array with last dim = 16 (GGUF packed bytes).

    Returns:
        uint8 array with same shape, nibbles reordered for MatMulNBits.
    """
    low = gguf_packed & 0x0F  # Elements 0..15
    high = (gguf_packed >> 4) & 0x0F  # Elements 16..31

    # Group each set of 16 nibbles into 8 pairs, pack each pair
    shape = gguf_packed.shape[:-1]
    low_pairs = low.reshape(*shape, 8, 2)
    high_pairs = high.reshape(*shape, 8, 2)

    ort_low = (low_pairs[..., 1] << 4) | low_pairs[..., 0]  # 8 bytes
    ort_high = (high_pairs[..., 1] << 4) | high_pairs[..., 0]  # 8 bytes

    return np.concatenate([ort_low, ort_high], axis=-1)  # 16 bytes


def _repack_q4_0(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q4_0 blocks.

    Q4_0 block (18 bytes): [fp16 scale (2B)][16B packed 4-bit quants]
    Dequant: (nibble - 8) * scale  →  symmetric with zero_point = 8.

    GGUF nibble ordering differs from MatMulNBits — we reorder during
    repacking.  See ``_reorder_nibbles_gguf_to_ort`` for details.
    """
    # Split scale (first 2 bytes) from quants (remaining 16 bytes)
    raw_scales = blocks[:, :2].copy()
    raw_quants = blocks[:, 2:]  # (total_blocks, 16)

    # Scales: view as fp16 → (total_blocks,) → reshape to (N, n_blocks)
    scales = raw_scales.view(np.float16).reshape(n_out, n_blocks_per_row)

    # Reorder nibbles from GGUF order to MatMulNBits order
    ort_quants = _reorder_nibbles_gguf_to_ort(raw_quants)
    weight = ort_quants.reshape(n_out, n_blocks_per_row, 16)

    # Zero points: Q4_0 is symmetric around 8
    # For MatMulNBits 4-bit: two ZPs packed per byte (low=block_i, high=block_i+1)
    # All ZPs are 8, so each packed byte = (8 << 4) | 8 = 0x88
    zp_cols = math.ceil(n_blocks_per_row / 2)
    zero_points = np.full((n_out, zp_cols), 0x88, dtype=np.uint8)
    # If odd number of blocks, the high nibble of the last byte is padding
    # ORT ignores it, but set to 0 for cleanliness
    if n_blocks_per_row % 2 == 1:
        zero_points[:, -1] = 0x08  # low nibble = 8, high nibble = 0

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=4,
    )


def _repack_q4_1(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q4_1 blocks.

    Q4_1 block (20 bytes): [fp16 scale (2B)][fp16 min (2B)][16B quants]
    Dequant: nibble * scale + min  →  asymmetric.

    MatMulNBits dequant: (nibble - zp) * scale
    So: zp = round(-min / scale), clamped to [0, 15].
    """
    raw_scales = blocks[:, :2].copy()
    raw_mins = blocks[:, 2:4].copy()
    raw_quants = blocks[:, 4:]  # (total_blocks, 16)

    scales_flat = raw_scales.view(np.float16).astype(np.float32).ravel()
    mins_flat = raw_mins.view(np.float16).astype(np.float32).ravel()

    # Compute zero points: zp = round(-min / scale), clamp to [0, 15]
    # Guard against division by zero: where scale == 0, zp = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        zp_float = np.where(
            scales_flat != 0,
            np.round(-mins_flat / scales_flat),
            0.0,
        )
    zp_uint4 = np.clip(zp_float, 0, 15).astype(np.uint8)

    # Reshape to (N, n_blocks)
    scales = scales_flat.astype(np.float16).reshape(n_out, n_blocks_per_row)
    # Reorder nibbles from GGUF order to MatMulNBits order
    ort_quants = _reorder_nibbles_gguf_to_ort(raw_quants)
    weight = ort_quants.reshape(n_out, n_blocks_per_row, 16)

    # Pack two 4-bit zero points per byte (vectorized, matching Q4_K)
    zp_2d = zp_uint4.reshape(n_out, n_blocks_per_row)
    zp_cols = math.ceil(n_blocks_per_row / 2)
    zp_padded = zp_2d
    if n_blocks_per_row % 2 == 1:
        zp_padded = np.zeros((n_out, n_blocks_per_row + 1), dtype=np.uint8)
        zp_padded[:, :n_blocks_per_row] = zp_2d
    zp_pairs = zp_padded.reshape(n_out, zp_cols, 2)
    zero_points = zp_pairs[:, :, 0] | (zp_pairs[:, :, 1] << 4)

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=4,
    )


def _repack_q8_0(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q8_0 blocks.

    Q8_0 block (34 bytes): [fp16 scale (2B)][32 x int8 values (32B)]
    Dequant: int8_val * scale  →  symmetric around 0.

    MatMulNBits dequant: (uint8_val - zp) * scale
    Convert: uint8_val = int8_val + 128, zp = 128.
    """
    raw_scales = blocks[:, :2].copy()
    raw_quants = blocks[:, 2:]  # (total_blocks, 32) as uint8

    scales = raw_scales.view(np.float16).reshape(n_out, n_blocks_per_row)

    # Convert signed int8 → unsigned uint8 by adding 128
    quants_int8 = raw_quants.view(np.int8).astype(np.int16)
    quants_uint8 = (quants_int8 + 128).astype(np.uint8)
    weight = quants_uint8.reshape(n_out, n_blocks_per_row, 32)

    # Zero points: 128 for all blocks
    zero_points = np.full((n_out, n_blocks_per_row), 128, dtype=np.uint8)

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=8,
    )


def _unpack_q4_k_scales(
    scales_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q4_K 6-bit packed scales into sub-block scales and mins.

    Each Q4_K super-block stores 8 sub-block scales and 8 sub-block mins
    packed into 12 bytes using 6-bit encoding.  The packing layout (from
    llama.cpp) uses three 4-byte groups::

        Bytes 0-3 (d):   low 6 bits → sub_scale[0..3],
                         high 2 bits → sub_scale[4..7] bits 4-5
        Bytes 4-7 (m):   low 6 bits → sub_min[0..3],
                         high 2 bits → sub_min[4..7] bits 4-5
        Bytes 8-11 (md): low 4 bits → sub_scale[4..7] bits 0-3,
                         high 4 bits → sub_min[4..7] bits 0-3

    Args:
        scales_raw: uint8 array, shape ``(n_super_blocks, 12)``.

    Returns:
        Tuple of ``(sub_scales, sub_mins)``, each ``(n_super_blocks, 8)``
        as uint8 in range [0, 63].
    """
    n = scales_raw.shape[0]
    s = scales_raw.reshape(n, 3, 4)
    d = s[:, 0, :]  # (n, 4) — bytes 0-3
    m = s[:, 1, :]  # (n, 4) — bytes 4-7
    md = s[:, 2, :]  # (n, 4) — bytes 8-11

    # Sub-scales: lower 4 from d[0..3] bits 0-5, upper 4 from md+d
    sc = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    # Sub-mins: lower 4 from m[0..3] bits 0-5, upper 4 from md+m
    mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(n, 8), mn.reshape(n, 8)


def _repack_q4_k(
    blocks: np.ndarray,
    n_out: int,
    n_super_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q4_K super-blocks into MatMulNBits sub-blocks.

    Q4_K uses 256-element super-blocks with a two-level scale hierarchy:
    ``value = d * sub_scale[i] * nibble - dmin * sub_min[i]``.  Each
    super-block is decomposed into 8 sub-blocks of 32 elements, and the
    two-level hierarchy is flattened to MatMulNBits' single-level format:
    ``value = (nibble - zero_point) * effective_scale``.

    **This flattening is lossy.**  The additive offset ``dmin * sub_min``
    is approximated as a multiplicative zero-point by dividing and
    rounding.  Rounding error is at most 0.5 * eff_scale.  Clamping
    error can be much larger when the offset exceeds 15 * eff_scale
    (i.e., when ``round(dmin * sub_min / eff_scale) > 15``).

    Args:
        blocks: uint8 array, shape ``(total_super_blocks, 144)``.
        n_out: Number of output rows (N dimension).
        n_super_blocks_per_row: Super-blocks per row.

    Returns:
        A ``RepackedTensor`` with ``block_size=32`` and ``bits=4``.
    """
    total = blocks.shape[0]

    # Parse super-block fields (144 bytes each):
    # [d: 2B fp16][dmin: 2B fp16][scales: 12B][quants: 128B]
    d_raw = blocks[:, :2].copy()
    dmin_raw = blocks[:, 2:4].copy()
    scales_raw = blocks[:, 4:16]
    qs_raw = blocks[:, 16:]  # (total, 128)

    d = d_raw.view(np.float16).astype(np.float32).ravel()  # (total,)
    dmin = dmin_raw.view(np.float16).astype(np.float32).ravel()  # (total,)

    # Unpack 6-bit sub-block scales and mins
    sub_scales, sub_mins = _unpack_q4_k_scales(scales_raw)
    sub_scales_f = sub_scales.astype(np.float32)  # (total, 8)
    sub_mins_f = sub_mins.astype(np.float32)  # (total, 8)

    # Effective per-sub-block scale: d * sub_scale[i]
    eff_scales = d[:, None] * sub_scales_f  # (total, 8)

    # Effective zero-point: zp = round(dmin * sub_min / eff_scale)
    # GGUF dequant: d * sub_scale * nibble - dmin * sub_min
    # MatMulNBits: (nibble - zp) * eff_scale = eff_scale * nibble - eff_scale * zp
    # So eff_scale * zp = dmin * sub_min → zp = dmin * sub_min / eff_scale
    numerator = dmin[:, None] * sub_mins_f  # (total, 8)
    with np.errstate(divide="ignore", invalid="ignore"):
        zp_float = np.where(
            eff_scales != 0,
            np.round(numerator / eff_scales),
            0.0,
        )
    zp_uint4 = np.clip(zp_float, 0, 15).astype(np.uint8)  # (total, 8)

    n_clamped = int(np.count_nonzero(zp_float > 15) + np.count_nonzero(zp_float < 0))
    if n_clamped > 0.05 * zp_float.size:
        logger.warning(
            "%d/%d Q4_K zero-points clamped — precision loss may be significant",
            n_clamped,
            zp_float.size,
        )

    # Unpack 4-bit quants from Q4_K layout.
    # 128 bytes = 4 groups of 32 bytes. Each group encodes two sub-blocks:
    #   byte[j] low nibble  → even sub-block element j
    #   byte[j] high nibble → odd sub-block element j
    qs = qs_raw.reshape(total, 4, 1, 32)
    shifts = np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs = (qs >> shifts) & np.uint8(0x0F)  # (total, 4, 2, 32)
    qs = qs.reshape(total, 8, 32)  # (total, 8, 32) — 8 sub-blocks

    # Repack each sub-block's 32 nibbles → 16 MatMulNBits bytes.
    # ORT format: byte[j] = (element[2j+1] << 4) | element[2j]
    pairs = qs.reshape(total, 8, 16, 2)
    ort_packed = (pairs[..., 1] << 4) | pairs[..., 0]  # (total, 8, 16)

    # Reshape to output dimensions: 8 sub-blocks per super-block
    n_sub_blocks_per_row = n_super_blocks_per_row * 8
    weight = ort_packed.reshape(n_out, n_sub_blocks_per_row, 16)
    scales_out = eff_scales.astype(np.float16).reshape(n_out, n_sub_blocks_per_row)

    # Pack two 4-bit zero-points per byte
    zp_2d = zp_uint4.reshape(n_out, n_sub_blocks_per_row)
    zp_cols = math.ceil(n_sub_blocks_per_row / 2)
    # n_sub_blocks_per_row is always a multiple of 8, so always even
    zp_padded = zp_2d
    if n_sub_blocks_per_row % 2 == 1:  # pragma: no cover — safety
        zp_padded = np.zeros((n_out, n_sub_blocks_per_row + 1), dtype=np.uint8)
        zp_padded[:, :n_sub_blocks_per_row] = zp_2d
    zp_pairs = zp_padded.reshape(n_out, zp_cols, 2)
    zero_points = zp_pairs[:, :, 0] | (zp_pairs[:, :, 1] << 4)

    return RepackedTensor(
        weight=weight,
        scales=scales_out,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=4,
    )
