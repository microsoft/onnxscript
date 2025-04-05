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
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Convert constant nodes in main graph to initializers."""
        count = 0
        for node in model.graph:
            if node.op_type != "Constant" or node.domain not in ("", "onnx.ai"):
                continue

            allowed_constant_attributes = {
                "value",
                "value_int",
                "value_ints",
                "value_float",
                "value_floats",
                "value_string",
                "value_strings",
            }
            constant_node_attribute = set(node.attributes.keys())
            if len(constant_node_attribute) != 1:
                logger.debug(
                    "Invalid constant node '%s' has more than one attribute", node.name
                )
                continue
            if constant_node_attribute not in allowed_constant_attributes:
                logger.debug("Invalid constant node '%s' has unsupported attribute", node.name)
                continue

            initializer_name = node.outputs[0].name
            assert initializer_name is not None
            # The value of attribute can only be ir.Attr, as
            # ir.RefAttr is only defined in Functions.
            attr_value = node.attributes[constant_node_attribute]
            if constant_node_attribute == "value":
                tensor = attr_value.as_tensor()  # type: ignore[union-attr]
            elif constant_node_attribute == "value_int":
                tensor = ir.Tensor(
                    np.array(attr_value.as_int(), dtype=np.int64), name=initializer_name
                )
            elif constant_node_attribute == "value_ints":
                tensor = ir.Tensor(
                    np.array(attr_value.as_ints(), dtype=np.int64), name=initializer_name
                )
            elif constant_node_attribute == "value_float":
                tensor = ir.Tensor(
                    np.array(attr_value.as_float(), dtype=np.float32), name=initializer_name
                )
            elif constant_node_attribute == "value_floats":
                tensor = ir.Tensor(
                    np.array(attr_value.as_floats(), dtype=np.float32), name=initializer_name
                )
            elif constant_node_attribute == "value_string":
                tensor = ir.Tensor(
                    np.array(attr_value.as_string(), dtype=np.object_), name=initializer_name
                )
            elif constant_node_attribute == "value_strings":
                tensor = ir.Tensor(
                    np.array(attr_value.as_strings(), dtype=np.object_), name=initializer_name
                )
            else:
                logger.debug("Invalid constant node '%s' has unsupported attribute", node.name)
                continue
            # Register an initializer with the tensor value
            initializer = ir.Value(
                name=initializer_name,
                shape=tensor.shape,  # type: ignore[arg-type]
                type=ir.TensorType(tensor.dtype),
                const_value=tensor,
            )
            # TODO(titaiwang): Is it possible that the initializer name has
            # been taken?
            model.graph.register_initializer(initializer)
            # Replace the constant node with the initilizer
            ir.convenience.replace_all_uses_with(node.outputs[0], initializer)
            model.graph.remove(node, safe=True)
            count += 1
            logger.info(
                "Converted constant node '%s' to initializer '%s'", node.name, initializer_name
            )
        if count:
            logger.info("Lifted %s constants to initializers", count)
        return ir.passes.PassResult(model, modified=bool(count))
