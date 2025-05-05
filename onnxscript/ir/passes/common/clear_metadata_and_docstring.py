# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Clear all metadata and docstring from the model, graphs, nodes, and functions."""

from __future__ import annotations

__all__ = [
    "ClearMetadataAndDocStringPass",
]

import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


class ClearMetadataAndDocStringPass(ir.passes.InPlacePass):
    """Clear all metadata and docstring from the model, graphs, nodes, and functions."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # 0. TODO: Should we clean model metadata and docstring?

        # 1. Clean up the graph and the belonged nodes metadata properties
        modified = self._clear_graph_or_function_metadata_and_docstring(model.graph)

        # 2. Clean up all of the functions metadata properties
        for function in model.functions.values():
            modified = (
                self._clear_graph_or_function_metadata_and_docstring(function) or modified
            )
        return ir.passes.PassResult(model, modified=modified)

    def _clear_graph_or_function_metadata_and_docstring(
        self,
        graph_or_function: ir.Graph | ir.Function,
    ) -> bool:
        """Clear metadata and docstring from the graph or function."""
        checked_graphs_or_functions: set[ir.Graph | ir.Function] = set()
        modified = False
        # Clean up all of the nodes metadata properties
        for node in ir.traversal.RecursiveGraphIterator(graph_or_function):
            if node.metadata_props:
                modified = True
                logger.debug("Removed metadata from %s nodes", node.name)
            node.metadata_props.clear()
            node.doc_string = None

            # Clean up the owning graph/function metadata properties
            # and doc_string if the graph/function is not already checked
            assert node.graph is not None
            if node.graph not in checked_graphs_or_functions and (
                node.graph.metadata_props or node.graph.doc_string
            ):
                modified = True
                logger.debug("Removed metadata from %s graph/function", node.graph.name)
                node.graph.metadata_props.clear()
                node.graph.doc_string = None
                checked_graphs_or_functions.add(node.graph)
        return modified
