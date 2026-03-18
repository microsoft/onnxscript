# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical comparison utilities for integration testing."""

from __future__ import annotations

import numpy as np


def assert_logits_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Assert that two logit tensors are numerically close.

    Args:
        actual: ONNX model output logits.
        expected: Reference (PyTorch) model output logits.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If shapes differ or values are not close.
    """
    assert actual.shape == expected.shape, (
        f"Shape mismatch: ONNX {actual.shape} vs reference {expected.shape}"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg="Logits differ between ONNX and reference model",
    )


def assert_generation_match(
    actual_ids: list[int] | np.ndarray,
    expected_ids: list[int] | np.ndarray,
) -> None:
    """Assert that two generated token sequences are identical.

    Args:
        actual_ids: Token IDs from ONNX generation.
        expected_ids: Token IDs from reference generation.

    Raises:
        AssertionError: If sequences differ.
    """
    actual = list(actual_ids) if not isinstance(actual_ids, list) else actual_ids
    expected = list(expected_ids) if not isinstance(expected_ids, list) else expected_ids
    assert actual == expected, (
        f"Generated tokens differ:\n  ONNX:      {actual}\n  Reference:  {expected}"
    )
