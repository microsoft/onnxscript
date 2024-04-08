# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utilities for comparing IR graphs."""

from __future__ import annotations

from onnxscript.ir import _core

# NOTE(justinchuby): We need to ensure a graph has valid inputs and outputs
# NOTE(justinchuby): A graph may be specified with a set of inputs and outputs


def topologically_equal(graph1: _core.Graph, graph2: _core.Graph) -> bool:
    """Return true if the two graphs are topologically equivalent, without considering initializers.

    Args:
        graph1: The first graph to compare.
        graph2: The second graph to compare.

    Returns:
        True if the graphs are equal, False otherwise.
    """
    raise NotImplementedError()
