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
            tensor = _constant_node_attribute_to_tensor(
                attr_name, attr_value, initializer_name
            )
            if tensor is None:
                logger.debug(
                    "Invalid constant node '%s' has unsupported attribute value", node.name
                )
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


def _constant_node_attribute_to_tensor(
    attr_name: str, attr_value: ir.Attr, initializer_name: str
) -> ir.Tensor | None:
    """Convert constant node attribute to tensor."""
    if attr_name == "value":
        tensor = attr_value.as_tensor()  # type: ignore[union-attr]
    elif attr_name == "value_int":
        tensor = ir.Tensor(
            np.array(attr_value.as_int(), dtype=np.int64), name=initializer_name
        )
    elif attr_name == "value_ints":
        tensor = ir.Tensor(
            np.array(attr_value.as_ints(), dtype=np.int64), name=initializer_name
        )
    elif attr_name == "value_float":
        tensor = ir.Tensor(
            np.array(attr_value.as_float(), dtype=np.float32), name=initializer_name
        )
    elif attr_name == "value_floats":
        tensor = ir.Tensor(
            np.array(attr_value.as_floats(), dtype=np.float32), name=initializer_name
        )
    elif attr_name == "value_string":
        tensor = ir.Tensor(
            np.array(attr_value.as_string(), dtype=np.object_), name=initializer_name
        )
    elif attr_name == "value_strings":
        tensor = ir.Tensor(
            np.array(attr_value.as_strings(), dtype=np.object_), name=initializer_name
        )
    else:
        tensor = None
    return tensor  # type: ignore[return-value]
