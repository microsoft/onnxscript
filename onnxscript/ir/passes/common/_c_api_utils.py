"""Utilities for interfacing with onnx C APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from onnxscript import ir

if TYPE_CHECKING:
    import onnx


logger = logging.getLogger(__name__)
# Temporarily remove initializers larger than this size to keep model size down
# for the onnx.shape_inference call because it needs to serialize the model
_BIG_TENSOR_SIZE_LIMIT = 1000  # 1KB


def call_onnx_api(
    func: Callable[[onnx.ModelProto], onnx.ModelProto],
    model: ir.Model,
    merge_func: Callable[[ir.Model, onnx.ModelProto], tuple[ir.Model, bool]],
) -> tuple[ir.Model, bool]:
    """Call an ONNX C API function by temporarily removing initializers.
    This is necessary because the ONNX C API does not support large models
    with initializers that have large tensor values.

    Args:
        func: Partially applied function that takes a model proto and returns a model proto.
        model: The IR model to pass to the API function.
        merge_func: Function that merges IR model with information from the model proto.

    Returns:
        A tuple containing the modified model and a boolean indicating whether the model was modified.
    """

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

    try:
        proto = ir.serde.serialize_model(model)
        result_proto = func(proto)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("Call to %s failed. The model is not modified", func, exc_info=True)
        return (model, False)
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

    # Merge the result with the original model
    return merge_func(model, result_proto)
