# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Numerical cases the global-max logcumsumexp formula gets wrong."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from onnxscript.function_libs.torch_lib.ops import core
from tests.function_libs.torch_lib.ops_test_common import graph_executor


def _assert_matches(values: torch.Tensor, dim: int) -> None:
    expected = torch.logcumsumexp(values, dim=dim)
    actual = graph_executor("logcumsumexp", [expected])(
        core.aten_logcumsumexp, (values.detach().numpy(), dim), {}
    )[0]
    np.testing.assert_allclose(
        actual,
        expected.detach().numpy(),
        rtol=1e-4 if values.dtype != torch.float16 else 1e-2,
        atol=1e-4 if values.dtype != torch.float16 else 1e-2,
        equal_nan=True,
    )


class LogCumSumExpTest(unittest.TestCase):
    def test_prefixes_far_below_the_global_max(self):
        # ReduceMax of the whole axis makes exp(-200 - 100) and exp(-100 - 100)
        # underflow, so the old formula returned [-inf, -inf, 100].
        _assert_matches(torch.tensor([-200.0, -100.0, 100.0]), 0)

    def test_logaddexp_keeps_both_terms(self):
        _assert_matches(torch.tensor([0.5, 0.5, 1.0]), 0)

    def test_negative_dim_and_non_zero_axis(self):
        values = torch.tensor([[1.0, 100.0, -200.0], [-50.0, -40.0, -30.0]])
        _assert_matches(values, 1)
        _assert_matches(values, -1)
        _assert_matches(values, 0)

    def test_non_finite_values(self):
        _assert_matches(torch.tensor([float("-inf"), float("-inf"), 3.0]), 0)
        _assert_matches(torch.tensor([float("inf"), float("inf")]), 0)
        _assert_matches(torch.tensor([1.0, float("nan"), 2.0]), 0)

    def test_empty_scan_axis(self):
        _assert_matches(torch.tensor([], dtype=torch.float32), 0)
        _assert_matches(torch.zeros(0, 2), 1)
        _assert_matches(torch.zeros(2, 0), 0)

    def test_scalar(self):
        _assert_matches(torch.tensor(4.0), 0)

    def test_float64_wide_range(self):
        _assert_matches(torch.tensor([-1000.0, -900.0, 10.0], dtype=torch.float64), 0)

    def test_float16_prefixes_below_the_max(self):
        # float16 exp underflows well before the float32 case.
        _assert_matches(torch.tensor([-8.0, -4.0, 8.0], dtype=torch.float16), 0)
