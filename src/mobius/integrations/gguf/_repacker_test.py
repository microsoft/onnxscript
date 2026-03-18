# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from mobius.integrations.gguf._repacker import (
    RepackedTensor,
    _unpack_q4_k_scales,
    can_repack,
    repack_gguf_tensor,
)

_Q4_0 = 2
_Q4_1 = 3
_Q8_0 = 8
_Q4_K = 12
_BLOCK_SIZE = 32


def _make_q4_0_block(scale: float, nibbles: list[int]) -> np.ndarray:
    """Build a single Q4_0 block (18 bytes) from scale + 32 element values.

    GGUF packing: byte i = (element[i+16] << 4) | element[i]
    """
    assert len(nibbles) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    packed = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        packed[i] = (nibbles[i + 16] << 4) | nibbles[i]
    return np.concatenate([scale_bytes, packed])


def _make_q4_1_block(scale: float, minimum: float, nibbles: list[int]) -> np.ndarray:
    """Build a single Q4_1 block (20 bytes).

    GGUF packing: byte i = (element[i+16] << 4) | element[i]
    """
    assert len(nibbles) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    min_bytes = np.array([minimum], dtype=np.float16).view(np.uint8)
    packed = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        packed[i] = (nibbles[i + 16] << 4) | nibbles[i]
    return np.concatenate([scale_bytes, min_bytes, packed])


def _make_q8_0_block(scale: float, values: list[int]) -> np.ndarray:
    """Build a single Q8_0 block (34 bytes) from scale + 32 int8 values."""
    assert len(values) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    val_bytes = np.array(values, dtype=np.int8).view(np.uint8)
    return np.concatenate([scale_bytes, val_bytes])


class TestCanRepack:
    def test_supported_types(self):
        assert can_repack(_Q4_0) is True
        assert can_repack(_Q4_1) is True
        assert can_repack(_Q8_0) is True
        assert can_repack(_Q4_K) is True

    def test_unsupported_types(self):
        assert can_repack(0) is False  # F32
        assert can_repack(1) is False  # F16
        assert can_repack(6) is False  # Q5_0
        assert can_repack(99) is False


class TestRepackQ40:
    def test_single_block(self):
        """Repack a single Q4_0 block with known values."""
        nibbles = list(range(16)) + list(range(16))  # 0..15, 0..15
        block = _make_q4_0_block(scale=0.5, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        assert isinstance(result, RepackedTensor)
        assert result.bits == 4
        assert result.block_size == 32
        assert result.weight.shape == (1, 1, 16)
        assert result.scales.shape == (1, 1)
        assert result.zero_points.shape == (1, 1)
        # Scale should match
        np.testing.assert_almost_equal(result.scales[0, 0], np.float16(0.5))
        # Zero point for Q4_0 is 8 (packed: 0x08 for single block)
        assert result.zero_points[0, 0] == 0x08

    def test_two_blocks_per_row(self):
        """Two blocks per row — zero points packed into one byte."""
        nibbles = [8] * 32
        block = _make_q4_0_block(scale=1.0, nibbles=nibbles)
        # 1 row, 2 blocks → shape (1, 64)
        raw = np.concatenate([block, block])

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 64))

        assert result.weight.shape == (1, 2, 16)
        assert result.scales.shape == (1, 2)
        # 2 blocks → 1 ZP byte, both nibbles = 8 → 0x88
        assert result.zero_points.shape == (1, 1)
        assert result.zero_points[0, 0] == 0x88

    def test_multiple_rows(self):
        """Multiple output features (N=3, K=32)."""
        nibbles = [5] * 32
        block = _make_q4_0_block(scale=2.0, nibbles=nibbles)
        raw = np.tile(block, 3)  # 3 rows x 1 block

        result = repack_gguf_tensor(raw, _Q4_0, shape=(3, 32))

        assert result.weight.shape == (3, 1, 16)
        assert result.scales.shape == (3, 1)
        assert result.zero_points.shape == (3, 1)

    def test_nibble_ordering_reordered(self):
        """Verify GGUF→ORT nibble reordering.

        GGUF: byte i has element[i] (low) and element[i+16] (high).
        ORT:  byte j has element[2j] (low) and element[2j+1] (high).

        Use elements 0-15 = [0]*16, elements 16-31 = [15]*16 to make
        the reordering visible.
        """
        nibbles = [0] * 16 + [15] * 16
        block = _make_q4_0_block(scale=1.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        # ORT bytes 0-7: pairs from elements 0..15 (all 0)
        # byte j = (element[2j+1] << 4) | element[2j] = (0<<4)|0 = 0x00
        for j in range(8):
            assert result.weight[0, 0, j] == 0x00
        # ORT bytes 8-15: pairs from elements 16..31 (all 15)
        # byte j = (15 << 4) | 15 = 0xFF
        for j in range(8, 16):
            assert result.weight[0, 0, j] == 0xFF

    def test_round_trip_dequantize(self):
        """Verify repacked data dequantizes to same values as GGUF."""
        from gguf import quants

        scale = np.float16(0.25)
        nibbles = [
            3,
            7,
            0,
            15,
            8,
            10,
            1,
            14,
            5,
            9,
            2,
            12,
            6,
            11,
            4,
            13,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
        block = _make_q4_0_block(scale=float(scale), nibbles=nibbles)

        # GGUF dequantize
        gguf_deq = quants.dequantize(block.reshape(1, -1), quants.GGMLQuantizationType.Q4_0)

        # Repack and manually dequantize via MatMulNBits formula
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        # Unpack nibbles from weight
        packed = result.weight[0, 0]  # (16,)
        low = (packed & 0x0F).astype(np.float32)
        high = ((packed >> 4) & 0x0F).astype(np.float32)
        elements = np.empty(32, dtype=np.float32)
        elements[0::2] = low
        elements[1::2] = high
        # MatMulNBits dequant: (element - zp) * scale
        zp = 8.0
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - zp) * s

        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=1e-3)


class TestRepackQ41:
    def test_single_block(self):
        """Repack a Q4_1 block with known scale and min."""
        nibbles = [0] * 32
        block = _make_q4_1_block(scale=0.5, minimum=-2.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        assert result.bits == 4
        assert result.weight.shape == (1, 1, 16)
        assert result.scales.shape == (1, 1)
        np.testing.assert_almost_equal(result.scales[0, 0], np.float16(0.5))
        # zp = round(-min / scale) = round(2.0 / 0.5) = 4
        zp_low = result.zero_points[0, 0] & 0x0F
        assert zp_low == 4

    def test_zero_scale_gives_zero_zp(self):
        """When scale=0, zero_point should be 0 (no division by zero)."""
        nibbles = [7] * 32
        block = _make_q4_1_block(scale=0.0, minimum=-1.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        zp = result.zero_points[0, 0] & 0x0F
        assert zp == 0

    def test_zp_clamped_to_15(self):
        """Zero point is clamped to [0, 15] for 4-bit."""
        # min = -100, scale = 1 → zp = 100 → clamp to 15
        nibbles = [0] * 32
        block = _make_q4_1_block(scale=1.0, minimum=-100.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        zp = result.zero_points[0, 0] & 0x0F
        assert zp == 15

    def test_round_trip_dequantize(self):
        """Verify Q4_1 repacked values match GGUF dequantization."""
        from gguf import quants

        scale = np.float16(0.5)
        minimum = np.float16(-1.0)
        nibbles = [0, 5, 10, 15] * 8
        block = _make_q4_1_block(scale=float(scale), minimum=float(minimum), nibbles=nibbles)

        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q4_1,
        )

        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        # Unpack and dequantize via MatMulNBits formula
        packed = result.weight[0, 0]
        low = (packed & 0x0F).astype(np.float32)
        high = ((packed >> 4) & 0x0F).astype(np.float32)
        elements = np.empty(32, dtype=np.float32)
        elements[0::2] = low
        elements[1::2] = high

        zp = float(result.zero_points[0, 0] & 0x0F)
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - zp) * s

        # Q4_1 → MatMulNBits is lossy (zero_point quantization)
        # Allow larger tolerance
        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=0.5)


class TestRepackQ80:
    def test_single_block(self):
        """Repack a Q8_0 block with known values."""
        values = list(range(-16, 16))  # 32 int8 values
        block = _make_q8_0_block(scale=0.1, values=values)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        assert result.bits == 8
        assert result.block_size == 32
        assert result.weight.shape == (1, 1, 32)
        assert result.scales.shape == (1, 1)
        assert result.zero_points.shape == (1, 1)
        # Zero point for symmetric Q8_0 is 128
        assert result.zero_points[0, 0] == 128

    def test_int8_to_uint8_conversion(self):
        """Verify signed int8 → unsigned uint8 + 128 offset."""
        values = [-128, -1, 0, 1, 127] + [0] * 27
        block = _make_q8_0_block(scale=1.0, values=values)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        # -128 + 128 = 0, -1 + 128 = 127, 0 + 128 = 128,
        # 1 + 128 = 129, 127 + 128 = 255
        assert result.weight[0, 0, 0] == 0
        assert result.weight[0, 0, 1] == 127
        assert result.weight[0, 0, 2] == 128
        assert result.weight[0, 0, 3] == 129
        assert result.weight[0, 0, 4] == 255

    def test_round_trip_dequantize(self):
        """Verify Q8_0 repacked values match GGUF dequantization."""
        from gguf import quants

        values = [int(x) for x in np.random.randint(-128, 128, size=32)]
        scale = 0.05
        block = _make_q8_0_block(scale=scale, values=values)

        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q8_0,
        )

        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        # MatMulNBits dequant: (uint8 - 128) * scale
        elements = result.weight[0, 0].astype(np.float32)
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - 128.0) * s

        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=1e-3)


class TestEdgeCases:
    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported GGUF type"):
            repack_gguf_tensor(np.zeros(10, dtype=np.uint8), 99, (1, 32))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected 2D shape"):
            repack_gguf_tensor(np.zeros(18, dtype=np.uint8), _Q4_0, (32,))

    def test_data_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="Data size mismatch"):
            repack_gguf_tensor(np.zeros(10, dtype=np.uint8), _Q4_0, (1, 32))

    def test_multi_block_multi_row(self):
        """N=4, K=128 → 4 blocks per row."""
        n, k = 4, 128
        n_blocks = k // 32
        block_bytes = 18
        _total = n * n_blocks * block_bytes
        # Create uniform blocks with scale=1.0, all nibbles=8
        scale_bytes = np.array([1.0], dtype=np.float16).view(np.uint8)
        quant_bytes = np.full(16, 0x88, dtype=np.uint8)  # nibbles all 8
        single_block = np.concatenate([scale_bytes, quant_bytes])
        raw = np.tile(single_block, n * n_blocks)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(n, k))

        assert result.weight.shape == (4, 4, 16)
        assert result.scales.shape == (4, 4)
        # 4 blocks → 2 ZP bytes per row
        assert result.zero_points.shape == (4, 2)
        assert result.zero_points[0, 0] == 0x88
        assert result.zero_points[0, 1] == 0x88


# ---- Q4_K helpers ----


def _pack_6bit_scales(sub_scales: list[int], sub_mins: list[int]) -> np.ndarray:
    """Pack 8 sub_scales + 8 sub_mins into 12 bytes (inverse of unpack).

    This is the encoding side of the 6-bit packing used by Q4_K.
    """
    assert len(sub_scales) == 8 and len(sub_mins) == 8
    out = np.zeros(12, dtype=np.uint8)
    # Bytes 0-3 (d): low 6 bits of sc[0..3], bits 6-7 from sc[4..7]
    for i in range(4):
        out[i] = (sub_scales[i] & 0x3F) | ((sub_scales[i + 4] & 0x30) << 2)
    # Bytes 4-7 (m): low 6 bits of min[0..3], bits 6-7 from min[4..7]
    for i in range(4):
        out[4 + i] = (sub_mins[i] & 0x3F) | ((sub_mins[i + 4] & 0x30) << 2)
    # Bytes 8-11 (md): low 4 bits of sc[4..7], high 4 bits of min[4..7]
    for i in range(4):
        out[8 + i] = (sub_scales[i + 4] & 0x0F) | ((sub_mins[i + 4] & 0x0F) << 4)
    return out


def _pack_q4_k_quants(nibbles: list[int]) -> np.ndarray:
    """Pack 256 nibbles into 128 bytes in Q4_K format.

    Within each 32-byte group, byte[j] = (odd_sub_block[j] << 4) |
    even_sub_block[j].
    """
    assert len(nibbles) == 256
    nibs = np.array(nibbles, dtype=np.uint8).reshape(8, 32)
    packed = np.zeros(128, dtype=np.uint8)
    for g in range(4):
        even = nibs[2 * g]
        odd = nibs[2 * g + 1]
        packed[g * 32 : (g + 1) * 32] = (odd << 4) | even
    return packed


def _make_q4_k_block(
    d: float,
    dmin: float,
    sub_scales: list[int],
    sub_mins: list[int],
    nibbles: list[int],
) -> np.ndarray:
    """Build a single Q4_K super-block (144 bytes)."""
    d_bytes = np.array([d], dtype=np.float16).view(np.uint8)
    dmin_bytes = np.array([dmin], dtype=np.float16).view(np.uint8)
    scales_bytes = _pack_6bit_scales(sub_scales, sub_mins)
    qs_bytes = _pack_q4_k_quants(nibbles)
    return np.concatenate([d_bytes, dmin_bytes, scales_bytes, qs_bytes])


class TestUnpackQ4KScales:
    def test_simple_values(self):
        """Pack known 6-bit values and verify round-trip."""
        sc = [1, 2, 3, 4, 5, 6, 7, 8]
        mn = [10, 20, 30, 40, 50, 60, 11, 22]
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_max_6bit_values(self):
        """All values at maximum (63)."""
        sc = [63] * 8
        mn = [63] * 8
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_zero_values(self):
        """All zeros."""
        sc = [0] * 8
        mn = [0] * 8
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_batch(self):
        """Multiple super-blocks at once."""
        sc1 = [10, 20, 30, 40, 50, 60, 15, 25]
        mn1 = [5, 15, 25, 35, 45, 55, 12, 22]
        sc2 = [1, 1, 1, 1, 1, 1, 1, 1]
        mn2 = [2, 2, 2, 2, 2, 2, 2, 2]
        packed = np.stack([_pack_6bit_scales(sc1, mn1), _pack_6bit_scales(sc2, mn2)])
        got_sc, got_mn = _unpack_q4_k_scales(packed)
        np.testing.assert_array_equal(got_sc[0], sc1)
        np.testing.assert_array_equal(got_mn[0], mn1)
        np.testing.assert_array_equal(got_sc[1], sc2)
        np.testing.assert_array_equal(got_mn[1], mn2)


class TestRepackQ4K:
    def test_single_super_block_shapes(self):
        """Single Q4_K super-block → 8 MatMulNBits sub-blocks."""
        sc = [10] * 8
        mn = [5] * 8
        nibs = [7] * 256
        block = _make_q4_k_block(d=1.0, dmin=0.5, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        assert isinstance(result, RepackedTensor)
        assert result.bits == 4
        assert result.block_size == 32
        # 1 super-block → 8 sub-blocks
        assert result.weight.shape == (1, 8, 16)
        assert result.scales.shape == (1, 8)
        assert result.zero_points.shape == (1, 4)  # 8 zps / 2

    def test_effective_scales(self):
        """Verify effective scale = d * sub_scale."""
        sc = [10, 20, 30, 40, 1, 2, 3, 4]
        mn = [0] * 8  # zero mins → zp=0, no offset
        nibs = [0] * 256
        d_val = 0.5
        block = _make_q4_k_block(d=d_val, dmin=0.0, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        for i, s in enumerate(sc):
            expected = np.float16(d_val * s)
            np.testing.assert_almost_equal(result.scales[0, i], expected, decimal=2)

    def test_zero_point_computation(self):
        """Verify zp = round(dmin * sub_min / eff_scale), clamped."""
        d_val = 1.0
        dmin_val = 1.0
        # Use sub_scale=1 so eff_scale=1.0, making zp = sub_min (direct)
        sc = [1] * 8
        # sub_min values in valid 6-bit range [0, 63]
        # zp = round(dmin * mn[i] / (d * sc[i])) = mn[i], clamped to [0,15]
        mn = [3, 5, 10, 0, 1, 63, 1, 2]
        expected_zp = [3, 5, 10, 0, 1, 15, 1, 2]  # 63 clamped to 15
        nibs = [0] * 256
        block = _make_q4_k_block(
            d=d_val, dmin=dmin_val, sub_scales=sc, sub_mins=mn, nibbles=nibs
        )
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        # Unpack 4-bit zero points from packed bytes
        zp_packed = result.zero_points[0]  # (4,) bytes
        zps = []
        for byte_val in zp_packed:
            zps.append(int(byte_val) & 0x0F)
            zps.append((int(byte_val) >> 4) & 0x0F)
        assert zps == expected_zp

    def test_zero_effective_scale_gives_zero_zp(self):
        """When d=0 or sub_scale=0, zp should be 0 (no division by 0)."""
        sc = [0] * 8
        mn = [50] * 8
        nibs = [8] * 256
        block = _make_q4_k_block(d=1.0, dmin=1.0, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        # All zps should be 0
        assert np.all(result.zero_points == 0)

    def test_round_trip_dequantize(self):
        """Compare repacked MatMulNBits dequant against gguf native."""
        from gguf import quants

        d_val = np.float16(0.01)
        dmin_val = np.float16(0.005)
        sc = [10, 20, 30, 40, 15, 25, 35, 45]
        mn = [5, 10, 15, 20, 8, 12, 18, 22]
        rng = np.random.RandomState(42)
        nibs = rng.randint(0, 16, size=256).tolist()

        block = _make_q4_k_block(
            d=float(d_val),
            dmin=float(dmin_val),
            sub_scales=sc,
            sub_mins=mn,
            nibbles=nibs,
        )

        # GGUF native dequantization (reference)
        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q4_K,
        ).ravel()  # (256,)

        # Repack to MatMulNBits
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        # Manually dequantize from MatMulNBits format
        ort_deq = np.empty(256, dtype=np.float32)
        zp_packed = result.zero_points[0]
        for sb in range(8):
            packed = result.weight[0, sb]  # (16,) uint8
            low = (packed & 0x0F).astype(np.float32)
            high = ((packed >> 4) & 0x0F).astype(np.float32)
            elements = np.empty(32, dtype=np.float32)
            elements[0::2] = low
            elements[1::2] = high
            s = result.scales[0, sb].astype(np.float32)
            byte_idx = sb // 2
            if sb % 2 == 0:
                zp = float(zp_packed[byte_idx] & 0x0F)
            else:
                zp = float((zp_packed[byte_idx] >> 4) & 0x0F)
            ort_deq[sb * 32 : (sb + 1) * 32] = (elements - zp) * s

        # Lossy: allow tolerance proportional to effective scale
        # Typical max error ≈ 0.5 * max(eff_scale)
        max_eff_scale = float(d_val) * max(sc)
        atol = max_eff_scale * 0.6
        np.testing.assert_allclose(ort_deq, gguf_deq, atol=atol)

    def test_multiple_rows_and_super_blocks(self):
        """N=2 rows, K=512 → 2 super-blocks per row."""
        sc = [10] * 8
        mn = [5] * 8
        nibs = [7] * 256
        block = _make_q4_k_block(d=0.1, dmin=0.05, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        # 2 rows x 2 super-blocks = 4 blocks total
        raw = np.tile(block, 4)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(2, 512))

        # 2 super-blocks/row x 8 sub-blocks/super = 16 sub-blocks/row
        assert result.weight.shape == (2, 16, 16)
        assert result.scales.shape == (2, 16)
        assert result.zero_points.shape == (2, 8)  # 16 zps / 2

    def test_d_zero_produces_zero_output(self):
        """When d=0, all effective scales and zero points are zero.

        This covers pruned layers where an entire super-block contributes
        nothing (all-zero weights).
        """
        sc = [10, 20, 30, 40, 15, 25, 35, 45]
        mn = [5, 10, 15, 20, 8, 12, 18, 22]
        nibs = [7] * 256
        block = _make_q4_k_block(d=0.0, dmin=0.0, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        # All effective scales should be zero (d * sub_scale = 0)
        np.testing.assert_array_equal(result.scales, 0)
        # All zero points should be zero (guarded against div-by-zero)
        np.testing.assert_array_equal(result.zero_points, 0)

    def test_q4_k_in_can_repack(self):
        """Q4_K (type 12) is recognized as repackable."""
        assert can_repack(12) is True
