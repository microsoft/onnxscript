# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Passes for extracting subgraphs from a graph."""

from __future__ import annotations

import itertools

__all__ = [
    "ExtractGraphPass",
]

import logging
from collections.abc import Collection

from onnxscript import ir

logger = logging.getLogger(__name__)


def _find_subgraph_bounded_by_values(
    graph: ir.Graph, inputs: Collection[ir.Value], outputs: Collection[ir.Value]
) -> tuple[list[ir.Node], list[ir.Value]]:
    """Finds the subgraph bounded by the given inputs and outputs.

    Args:
        graph: The graph to search.
        inputs: The inputs to the subgraph.
        outputs: The outputs of the subgraph.

    Returns:
        A list of nodes in the subgraph and the initializers used.
    """
    all_nodes = []
    value_stack: list[ir.Value] = [*outputs]
    visited_nodes: set[ir.Node] = set()
    visited_values: set[ir.Value] = set(inputs)
    initializers = []
    while value_stack:
        value = value_stack.pop()
        if value in visited_values:
            continue
        if value.name in graph.initializers:
            # Record the initializer
            assert value.const_value is not None
            initializers.append(value)
        visited_values.add(value)
        if (node := value.producer()) is not None:
            if node not in visited_nodes:
                visited_nodes.add(node)
                all_nodes.append(node)
                for input in node.inputs:
                    if input not in visited_values and input is not None:
                        value_stack.append(input)
    return all_nodes, initializers


class ExtractGraphPass(ir.passes.PassBase):
    """This pass extracts a subgraph from the given graph."""
    # This pass does not modify the model in place
    in_place = False
    # This pass destroys the input model
    destructive = True

    def __init__(self, input_names: Collection[str], output_names: Collection[str]) -> None:
        """Extracts sub-model from an ONNX model.

        The sub-model is defined by the names of the input and output tensors *exactly*.

        Args:
            input_names: The names of the inputs to extract. Must be deduplicated.
            output_names: The names of the outputs to extract. Must be deduplicated.
        """
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        values = ir.convenience.create_value_mapping(model.graph)
        inputs = [values[name] for name in self.input_names]
        outputs = [values[name] for name in self.output_names]
        extracted_nodes, initializers = _find_subgraph_bounded_by_values(
            model.graph, inputs, outputs
        )

        model.graph.remove(extracted_nodes)
        # Create inputs for the new graph as the old inputs are owned by the old nodes
        new_inputs = []
        for input in inputs:
            new_inputs.append(
                ir.Value(
                    name=input.name,
                    shape=input.shape,
                    type=input.type,
                    doc_string=input.doc_string,
                    const_value=input.const_value,
                )
            )
        ir.convenience.replace_all_uses_with(inputs, new_inputs)

        new_model = ir.Model(
            ir.Graph(
                new_inputs,
                outputs,
                nodes=extracted_nodes,
                initializers=initializers,
                doc_string=model.graph.doc_string,
                opset_imports=model.graph.opset_imports,
                name=model.graph.name,
                metadata_props=model.graph.metadata_props,
            ),
            ir_version=model.ir_version,
            producer_name=model.producer_name,
            producer_version=model.producer_version,
            domain=model.domain,
            model_version=model.model_version,
            doc_string=model.doc_string,
            functions=tuple(model.functions.values()),
            meta_data_props=model.metadata_props,
        )
        return ir.passes.PassResult(new_model, modified=True)

    def requires(self, model: ir.Model) -> None:
        # All inputs and outputs can be found in the model
        values = ir.convenience.create_value_mapping(model.graph)
        input_names_not_found = sorted(set(self.input_names) - set(values.keys()))
        if input_names_not_found:
            raise ir.passes.PreconditionError(
                f"Input names not found in the model: {input_names_not_found}"
            )
        output_names_not_found = sorted(set(self.output_names) - set(values.keys()))
        if output_names_not_found:
            raise ir.passes.PreconditionError(
                f"Output names not found in the model: {output_names_not_found}"
            )

        # All inputs and outputs must have type and shape
        for name in itertools.chain(self.input_names, self.output_names):
            value = values[name]
            if value.type is None:
                logger.warning(
                    "Value %%%s does not have a type: '%r'. "
                    "Consider setting its type or running shape inference first.",
                    name, value
                )
            if value.shape is None:
                logger.warning(
                    "Value %%%s does not have a shape: '%r'. "
                    "Consider setting its shape or running shape inference first.",
                    name, value
                )
        # TODO(justinchuby): Make sure the subgraph is completely bounded by inputs and outputs
