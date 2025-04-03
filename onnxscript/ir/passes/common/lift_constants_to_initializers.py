# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Lift constants to initializers."""

from __future__ import annotations

__all__ = [
    "LiftConstantsToInitializersPass",
]

import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


class LiftConstantsToInitializersPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Convert constant nodes in main graph to initializers."""
        count = 0
        for node in model.graph:
            if node.op_type != "Constant" or node.domain not in ("", "onnx.ai"):
                continue
            if "value" not in node.attributes:
                logger.debug("Constant node '%s' has no 'value' attribute", node.name)
                continue
            # The value of attribute can only be ir.Attr, as
            # ir.RefAttr is only defined in Functions.
            tensor = node.attributes["value"].as_tensor()  # type: ignore[union-attr]
            # Register an initializer with the tensor value
            initializer_name = node.outputs[0].name
            assert initializer_name is not None
            initializer = ir.Value(
                name=initializer_name,
                shape=tensor.shape,  # type: ignore[arg-type]
                type=ir.TensorType(tensor.dtype),
                const_value=tensor,
            )
            # TODO(titaiwang): Is it possible that the initializer name has
            # been taken?
            model.graph.initializers[initializer_name] = initializer
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
