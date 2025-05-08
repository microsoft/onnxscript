# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Lift constants to initializers."""

from __future__ import annotations

__all__ = [
    "AddInitializersToInputsPass",
    "LiftConstantsToInitializersPass",
    "LiftSubgraphInitializersToMainGraphPass",
    "RemoveInitializersFromInputsPass",
]

import logging

import numpy as np

from onnxscript import ir

logger = logging.getLogger(__name__)


class LiftConstantsToInitializersPass(ir.passes.InPlacePass):
    """Lift constants to initializers.

    Attributes:
        lift_all_constants: Whether to lift all Constant nodes, including those that does not contain a tensor attribute (e.g. with value_ints etc.)
            Default to False, where only Constants with the ``value`` attribute are lifted.
        size_limit: The minimum size of the tensor to be lifted. If the tensor contains
            number of elements less than size_limit, it will not be lifted. Default is 16.
    """

    def __init__(self, lift_all_constants: bool = False, size_limit: int = 16):
        super().__init__()
        self.lift_all_constants = lift_all_constants
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            assert node.graph is not None
            if node.op_type != "Constant" or node.domain not in ("", "onnx.ai"):
                continue
            if node.outputs[0].is_graph_output():
                logger.debug(
                    "Constant node '%s' is used as output, so it can't be lifted.", node.name
                )
                continue
            constant_node_attribute = set(node.attributes.keys())
            if len(constant_node_attribute) != 1:
                logger.debug(
                    "Invalid constant node '%s' has more than one attribute", node.name
                )
                continue

            attr_name, attr_value = next(iter(node.attributes.items()))
            initializer_name = node.outputs[0].name
            assert initializer_name is not None
            assert isinstance(attr_value, ir.Attr)
            tensor = self._constant_node_attribute_to_tensor(
                node, attr_name, attr_value, initializer_name
            )
            if tensor is None:
                # The reason of None is logged in _constant_node_attribute_to_tensor
                continue
            # Register an initializer with the tensor value
            initializer = ir.Value(
                name=initializer_name,
                shape=tensor.shape,  # type: ignore[arg-type]
                type=ir.TensorType(tensor.dtype),
                const_value=tensor,
            )
            assert node.graph is not None
            node.graph.register_initializer(initializer)
            # Replace the constant node with the initializer
            ir.convenience.replace_all_uses_with(node.outputs[0], initializer)
            node.graph.remove(node, safe=True)
            count += 1
            logger.debug(
                "Converted constant node '%s' to initializer '%s'", node.name, initializer_name
            )
        if count:
            logger.debug("Lifted %s constants to initializers", count)
        return ir.passes.PassResult(model, modified=bool(count))

    def _constant_node_attribute_to_tensor(
        self, node, attr_name: str, attr_value: ir.Attr, initializer_name: str
    ) -> ir.TensorProtocol | None:
        """Convert constant node attribute to tensor."""
        if not self.lift_all_constants and attr_name != "value":
            logger.debug(
                "Constant node '%s' has non-tensor attribute '%s'", node.name, attr_name
            )
            return None

        tensor: ir.TensorProtocol
        if attr_name == "value":
            tensor = attr_value.as_tensor()
        elif attr_name == "value_int":
            tensor = ir.tensor(
                attr_value.as_int(), dtype=ir.DataType.INT64, name=initializer_name
            )
        elif attr_name == "value_ints":
            tensor = ir.tensor(
                attr_value.as_ints(), dtype=ir.DataType.INT64, name=initializer_name
            )
        elif attr_name == "value_float":
            tensor = ir.tensor(
                attr_value.as_float(), dtype=ir.DataType.FLOAT, name=initializer_name
            )
        elif attr_name == "value_floats":
            tensor = ir.tensor(
                attr_value.as_floats(), dtype=ir.DataType.FLOAT, name=initializer_name
            )
        elif attr_name in ("value_string", "value_strings"):
            tensor = ir.StringTensor(
                np.array(attr_value.value, dtype=np.bytes_), name=initializer_name
            )
        else:
            raise ValueError(
                f"Unsupported constant node '{node.name}' attribute '{attr_name}'"
            )

        if tensor.size < self.size_limit:
            logger.debug(
                "Tensor from node '%s' has less than %s elements",
                node.name,
                self.size_limit,
            )
            return None
        return tensor


class LiftSubgraphInitializersToMainGraphPass(ir.passes.InPlacePass):
    """Lift subgraph initializers to main graph.

    This pass lifts the initializers of a subgraph to the main graph.
    It is used to ensure that the initializers are available in the main graph
    for further processing or optimization.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        registered_initializer_names: dict[str, int] = {}
        for graph in model.graphs():
            if graph is model.graph:
                continue
            for name in tuple(graph.initializers):
                initializer = graph.initializers[name]
                if initializer.is_graph_input():
                    # Skip the ones that are also graph inputs
                    logger.debug(
                        "Initializer '%s' is also a graph input, so it can't be lifted",
                        initializer.name,
                    )
                    continue
                # Remove the initializer from the subgraph
                graph.initializers.pop(name)
                # To avoid name conflicts, we need to rename the initializer
                # to a unique name in the main graph
                if name in registered_initializer_names:
                    name_count = registered_initializer_names[name]
                    initializer.name = f"{name}_{name_count}"
                    registered_initializer_names[name] = name_count + 1
                else:
                    assert initializer.name is not None
                    registered_initializer_names[initializer.name] = 1
                model.graph.register_initializer(initializer)
                count += 1
                logger.debug(
                    "Lifted initializer '%s' from subgraph '%s' to main graph",
                    initializer.name,
                    graph.name,
                )
        return ir.passes.PassResult(model, modified=bool(count))


class RemoveInitializersFromInputsPass(ir.passes.InPlacePass):
    """Remove initializers from inputs.

    This pass finds all graph inputs that have a const_value and removes them from the graph.inputs list.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        for graph in model.graphs():
            initializers = set(graph.initializers.values())
            new_inputs = []
            for input_value in graph.inputs:
                if input_value in initializers:
                    count += 1
                else:
                    new_inputs.append(input_value)
            graph.inputs.clear()
            graph.inputs.extend(new_inputs)
        logger.info("Removed %s initializers from graph inputs", count)
        return ir.passes.PassResult(model, modified=bool(count))


class AddInitializersToInputsPass(ir.passes.InPlacePass):
    """Add initializers to inputs.

    This pass finds all initializers and adds them to the graph.inputs list if they are not already present.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        for graph in model.graphs():
            inputs_set = set(graph.inputs)
            for initializer in graph.initializers.values():
                if initializer not in inputs_set:
                    graph.inputs.append(initializer)
                    count += 1
        logger.info("Added %s initializers to graph inputs", count)
        return ir.passes.PassResult(model, modified=bool(count))
