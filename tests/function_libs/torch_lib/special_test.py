# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Focused TorchLib tests for special functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from onnxscript.function_libs.torch_lib.ops import special
from tests.function_libs.torch_lib import ops_test_common


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((np.float16, 2e-3, 0), (np.float32, 1.3e-6, 0), (np.float64, 2e-14, 0)),
)
def test_erfcx_ort_matches_torch_at_boundaries_and_special_values(dtype, rtol, atol):
    """Exercises each approximation range and the low-precision working path."""

    values = np.array(
        [
            -np.inf,
            -30.0,
            -26.0,
            -12.0,
            -8.001,
            -8.0,
            -7.999,
            -1.001,
            -1.0,
            -0.999,
            -0.0,
            0.0,
            0.999,
            1.0,
            1.001,
            7.999,
            8.0,
            8.001,
            12.0,
            30.0,
            np.inf,
            np.nan,
        ],
        dtype=dtype,
    )
    with np.errstate(over="ignore"):
        expected = torch.special.erfcx(torch.from_numpy(values).float()).numpy().astype(dtype)
    if dtype is np.float64:
        expected = torch.special.erfcx(torch.from_numpy(values)).numpy()
    actual = ops_test_common.graph_executor("test_erfcx", [torch.from_numpy(expected)])(
        special.aten_special_erfcx, (values,), {}
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    assert np.isposinf(actual[0])
    assert actual[-2] == 0
    assert np.isnan(actual[-1])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_erfcx_ort_has_correct_large_positive_asymptote(dtype):
    """Checks the reciprocal tail form that prevents polynomial overflow."""

    values = np.array([8.0, 12.0, 30.0, np.finfo(dtype).max], dtype=dtype)
    expected = torch.special.erfcx(torch.from_numpy(values)).numpy()
    actual = ops_test_common.graph_executor("test_erfcx_tail", [torch.from_numpy(expected)])(
        special.aten_special_erfcx, (values,), {}
    )[0]

    rtol = 1.3e-6 if dtype is np.float32 else 2e-14
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)
    np.testing.assert_allclose(values[-1] * actual[-1], 1 / math.sqrt(math.pi), rtol=rtol)


@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
@pytest.mark.parametrize("shape", ((), (0,), (2, 0, 3), (2, 3)))
def test_erfcx_ort_preserves_shape_and_dtype(dtype, shape):
    values = np.ones(shape, dtype=dtype)
    expected = torch.special.erfcx(torch.from_numpy(values).double()).numpy().astype(dtype)
    actual = ops_test_common.graph_executor("test_erfcx_shape", [torch.from_numpy(expected)])(
        special.aten_special_erfcx, (values,), {}
    )[0]
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-3 if dtype is np.float16 else 1e-6)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_erfcx_ort_across_approximation_intervals(dtype):
    # Include adjacent representable values at each piecewise boundary.
    boundaries = np.array([-8, -1, 0, 1, 8], dtype=dtype)
    values = np.concatenate(
        [
            np.linspace(-9, 0, 257, dtype=dtype),
            np.linspace(0, 1, 257, dtype=dtype),
            np.linspace(1, 8, 257, dtype=dtype),
            np.geomspace(8, 1e30 if dtype is np.float32 else 1e300, 257).astype(dtype),
            boundaries,
            np.nextafter(boundaries, -np.inf),
            np.nextafter(boundaries, np.inf),
        ]
    )
    expected = torch.special.erfcx(torch.from_numpy(values)).numpy()
    actual = ops_test_common.graph_executor(
        "test_erfcx_intervals", [torch.from_numpy(expected)]
    )(special.aten_special_erfcx, (values,), {})[0]
    np.testing.assert_allclose(
        actual, expected, rtol=1.3e-6 if dtype is np.float32 else 2e-14, atol=0
    )
