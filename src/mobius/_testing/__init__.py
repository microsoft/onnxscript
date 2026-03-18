# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for mobius tests."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal.builder import GraphBuilder

from mobius._configs import ArchitectureConfig
from mobius._constants import OPSET_VERSION
from mobius._testing.code_paths import (
    CODE_PATH_INDICATORS as CODE_PATH_INDICATORS,
)
from mobius._testing.code_paths import (
    detect_code_paths as detect_code_paths,
)
from mobius._testing.code_paths import (
    detect_code_paths_from_config as detect_code_paths_from_config,
)
from mobius._testing.code_paths import (
    get_all_code_path_labels as get_all_code_path_labels,
)
from mobius._testing.golden import (
    GoldenRef as GoldenRef,
)
from mobius._testing.golden import (
    GoldenTestCase as GoldenTestCase,
)
from mobius._testing.golden import (
    Tolerances as Tolerances,
)
from mobius._testing.golden import (
    discover_test_cases as discover_test_cases,
)
from mobius._testing.golden import (
    golden_path_for_case as golden_path_for_case,
)
from mobius._testing.golden import (
    has_golden as has_golden,
)
from mobius._testing.golden import (
    load_golden_ref as load_golden_ref,
)
from mobius._testing.golden import (
    load_test_case as load_test_case,
)
from mobius._testing.golden import (
    load_tolerances as load_tolerances,
)
from mobius._testing.golden import (
    save_golden_ref as save_golden_ref,
)
from mobius._testing.parity import (
    ParityReport as ParityReport,
)
from mobius._testing.parity import (
    ParityResult as ParityResult,
)
from mobius._testing.parity import (
    compare_golden as compare_golden,
)
from mobius._testing.parity import (
    compare_synthetic as compare_synthetic,
)


def create_test_builder():
    """Create a GraphBuilder with a fresh graph for testing.

    Returns:
        Tuple of (builder, op, graph).
    """
    graph = ir.Graph(
        [],
        [],
        nodes=[],
        name="test_graph",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    op = builder.op
    return builder, op, graph


def create_test_input(
    builder: GraphBuilder,
    name: str,
    shape: list,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> ir.Value:
    """Create a test input value and add it to the graph."""
    value = ir.Value(
        name=name,
        shape=ir.Shape(shape),
        type=ir.TensorType(dtype),
    )
    builder.graph.inputs.append(value)
    return value


def count_op_type(graph: ir.Graph, op_type: str) -> int:
    """Count the number of nodes with the given op_type in the graph."""
    count = 0
    for node in graph:
        if node.op_type == op_type:
            count += 1
    return count


def make_config(**overrides) -> ArchitectureConfig:
    """Create a minimal test config with sensible defaults."""
    defaults = dict(
        vocab_size=100,
        max_position_embeddings=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        head_dim=16,
        pad_token_id=0,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)
