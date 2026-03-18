# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for rewrite rule tests."""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx_ir as ir

from mobius._testing.ort_inference import OnnxModelSession


def count_ops(model: ir.Model) -> Counter:
    """Count op types in the model's main graph."""
    return Counter(n.op_type for n in model.graph)


def fill_random_weights(model: ir.Model) -> None:
    """Fill all uninitialised parameters with random float32 values."""
    for init in model.graph.initializers.values():
        if init.const_value is None:
            shape = list(init.shape)
            init.const_value = ir.Tensor(np.random.randn(*shape).astype(np.float32))


def make_prefill_feeds(session: OnnxModelSession, seq_len: int = 3) -> dict[str, np.ndarray]:
    """Create dummy prefill feeds with past_seq_len=0."""
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([list(range(1, seq_len + 1))], dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for inp in session._session.get_inputs():
        if inp.name not in feeds:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            # No past context for prefill
            if "past_key" in inp.name or "past_value" in inp.name:
                shape[2] = 0
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    return feeds
