# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shape inference pass using onnx.shape_inference."""

from __future__ import annotations

__all__ = [
    "ShapeInferencePass",
    "infer_shapes",
]

import logging

import onnx

from onnxscript import ir

logger = logging.getLogger(__name__)

# Temporarily remove initializers larger than this size to keep model size down
# for the onnx.shape_inference call because it needs to serialize the model
_BIG_TENSOR_SIZE_LIMIT = 1000  # 1KB


class ShapeInferencePass(ir.passes.FunctionalPass):
    """This pass performs shape inference on the graph."""

    def __init__(
        self, check_type: bool = True, strict_mode: bool = True, data_prop: bool = True
    ) -> None:
        """Initialize the shape inference pass.

        Args:
            check_type: If True, check the types of the inputs and outputs.
            strict_mode: If True, use strict mode for shape inference.
            data_prop: If True, use data propagation for shape inference.
        """
        super().__init__()
        self.check_type = check_type
        self.strict_mode = strict_mode
        self.data_prop = data_prop

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Store the original initializer values so they can be restored
        initializer_values = tuple(model.graph.initializers.values())
        tensors = {v.name: v.const_value for v in initializer_values}
        original_inputs_len = len(model.graph.inputs)
        initializer_names = {v.name for v in initializer_values}

        # Turn the initializers into inputs and clear the initializers
        # to limit the model size
        for initializer in initializer_values:
            # Make sure the initializer has its shape/type set
            assert initializer.const_value is not None
            if initializer.shape is None:
                initializer.shape = initializer.const_value.shape  # type: ignore[assignment]
            if initializer.dtype is None:
                initializer.dtype = initializer.const_value.dtype
            if initializer not in model.graph.inputs:
                model.graph.inputs.append(initializer)
            if initializer.const_value.nbytes > _BIG_TENSOR_SIZE_LIMIT:
                # Temporarily remove the initializer value to reduce model size
                # for onnx.shape_inference
                initializer.const_value = None
                assert initializer.name is not None
                model.graph.initializers.pop(initializer.name)

        # Perform shape inference
        try:
            proto = ir.serde.serialize_model(model)
            value_infos = {info.name: info for info in proto.graph.value_info}
            inferred_proto = onnx.shape_inference.infer_shapes(
                proto,
                check_type=self.check_type,
                strict_mode=self.strict_mode,
                data_prop=self.data_prop,
            )
            inferred_value_infos = {
                info.name: info for info in inferred_proto.graph.value_info
            }
            inferred_model = ir.serde.deserialize_model(inferred_proto)

        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Shape inference failed. The model is not modified", exc_info=True)
            return ir.passes.PassResult(model, modified=False)
        finally:
            # Restore the original initializer values so the model is unchanged
            for initializer in initializer_values:
                if initializer.name in initializer_names:
                    initializer.const_value = tensors[initializer.name]
                    model.graph.register_initializer(initializer)

            # Restore the original inputs
            inputs = model.graph.inputs[:original_inputs_len]
            model.graph.inputs.clear()
            model.graph.inputs.extend(inputs)

        # Add the original initializer tensors to the new (inferred) model
        for new_input in inferred_model.graph.inputs:
            # Assign the tensors back to the initializers
            if new_input.name in initializer_names:
                new_input.const_value = tensors[new_input.name]
                inferred_model.graph.register_initializer(new_input)

        # Remove the inputs that were added
        new_inputs = inferred_model.graph.inputs[:original_inputs_len]
        inferred_model.graph.inputs.clear()
        inferred_model.graph.inputs.extend(new_inputs)

        return ir.passes.PassResult(
            inferred_model, modified=value_infos != inferred_value_infos
        )


def infer_shapes(
    model: ir.Model,
    *,
    check_type: bool = True,
    strict_mode: bool = True,
    data_prop: bool = True,
) -> ir.Model:
    """Perform shape inference on the model.

    Args:
        model: The model to perform shape inference on.
        check_type: If True, check the types of the inputs and outputs.
        strict_mode: If True, use strict mode for shape inference.
        data_prop: If True, use data propagation for shape inference.

    Returns:
        The model with shape inference applied.
    """
    return ShapeInferencePass(
        check_type=check_type, strict_mode=strict_mode, data_prop=data_prop
    )(model).model
