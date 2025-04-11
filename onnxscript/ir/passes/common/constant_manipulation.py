# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Lift constants to initializers."""

from __future__ import annotations

__all__ = [
    "LiftConstantsToInitializersPass",
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
    """

    def __init__(self, lift_all_constants: bool = False):
        super().__init__()
        self._lift_all_constants = lift_all_constants

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            assert node.graph is not None
            if node.op_type != "Constant" or node.domain not in ("", "onnx.ai"):
                continue
            if node.outputs[0] in node.graph.outputs:
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
            assert isinstance(node.graph, ir.Graph)
            node.graph.register_initializer(initializer)
            # Replace the constant node with the initilizer
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
    ) -> ir.Tensor | None:
        """Convert constant node attribute to tensor."""
        if not self._lift_all_constants and attr_name != "value":
            logger.debug(
                "Constant node '%s' has non-tensor attribute '%s'", node.name, attr_name
            )
            return None

        # Dispatch table for attribute-to-tensor conversion
        tensor_converters = {
            "value": lambda: attr_value.as_tensor(),  # pylint: disable=unnecessary-lambda`
            "value_int": lambda: ir.tensor(
                attr_value.as_int(), dtype=ir.DataType.INT64, name=initializer_name
            ),
            "value_ints": lambda: ir.tensor(
                attr_value.as_ints(), dtype=ir.DataType.INT64, name=initializer_name
            ),
            "value_float": lambda: ir.tensor(
                attr_value.as_float(), dtype=ir.DataType.FLOAT, name=initializer_name
            ),
            "value_floats": lambda: ir.tensor(
                attr_value.as_floats(), dtype=ir.DataType.FLOAT, name=initializer_name
            ),
            "value_string": lambda: ir.StringTensor(
                np.array(attr_value.value, dtype=np.bytes_), name=initializer_name
            ),
            "value_strings": lambda: ir.StringTensor(
                np.array(attr_value.value, dtype=np.bytes_), name=initializer_name
            ),
        }
        converter = tensor_converters.get(attr_name)
        if converter is None:
            logger.debug(
                "Unsupported constant node attribute '%s' in node '%s'", attr_name, node.name
            )
            return None

        return converter()  # type: ignore[return-value]
