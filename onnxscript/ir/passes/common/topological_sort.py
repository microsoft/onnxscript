# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pass for topologically sorting the graphs."""

from __future__ import annotations

__all__ = [
    "TopologicalSortPass",
]


from onnxscript import ir


class TopologicalSortPass(ir.passes.InPlacePass):
    """Topologically sort graphs and functions in a model."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        original_nodes = list(model.graph)
        model.graph.sort()
        sorted_nodes = list(model.graph)
        for function in model.functions.values():
            original_nodes.extend(function)
            function.sort()
            sorted_nodes.extend(function)

        # Compare node orders to determine if any changes were made
        modified = False
        for node, new_node in zip(original_nodes, sorted_nodes):
            if node is not new_node:
                modified = True
                break
        return ir.passes.PassResult(model=model, modified=modified)
