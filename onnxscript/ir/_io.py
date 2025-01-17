# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load and save ONNX models."""

from __future__ import annotations

__all__ = ["load", "save"]

import os

import onnx

from onnxscript.ir import _core, _external_data, serde


def load(path: str | os.PathLike, format: str | None = None) -> _core.Model:
    """Load an ONNX model from a file.

    Args:
        path: The path to the ONNX file.
        format: The format of the file (e.g. protobuf, textproto, json, etc.).
            If None, the format is inferred from the file extension.

    Returns:
        The loaded model.
    """
    # Do not use ONNX to load external data because the IR handles external data
    # by doing memory mapping directly.
    proto = onnx.load(path, format=format, load_external_data=False)
    model = serde.deserialize_model(proto)
    base_dir = os.path.dirname(path)
    # Set the base directory for external data to the directory of the ONNX file
    # so that relative paths are resolved correctly.
    _external_data.set_base_dir(model.graph, base_dir)
    return model


def save(
    model: _core.Model,
    path: str | os.PathLike,
    format: str | None = None,
    external_data: str | os.PathLike | None = None,
    modify_model: bool = False,
) -> None:
    """Save an ONNX model to a file.

    Args:
        model: The model to save.
        path: The path to save the model to. E.g. "model.onnx".
        format: The format of the file (e.g. ``protobuf``, ``textproto``, ``json``, etc.).
            If None, the format is inferred from the file extension.
        external_data: The relative path to save external data to. When specified,
            all initializers in the model will be converted to external data and
            saved to the specified directory. If None, all tensors will be saved unmodified.
            That is, if a tensor in the model is already external, it will be saved
            with the same external information; if the tensor is not external,
            it will be serialized in the ONNX Proto message.
        modify_model: Whether to modify the model in place when :param:`external_data` is ``True``.
            If ``False``, the model will be kept unmodified after saving. If ``True``, the model's
            initializers will reference the newly created external data file.
            If the external data path is currently referenced by an initializer in the model,
            ``modify_model`` must be set to ``True`` to maintain model correctness.

    Raises:
        ValueError: If the external data path is an absolute path.
        ValueError: If the external data path is currently referenced by an initializer
            and :param:`modify_model` is set to False.
    """
    if external_data is not None:
        if os.path.isabs(external_data):
            raise ValueError(
                f"The external data path must be a relative to the ONNX file path, not '{external_data}'."
            )
        base_dir = os.path.dirname(path)

        # Filter out the uninitialized initializer values
        initializer_values = [
            v for v in model.graph.initializers.values() if v.const_value is not None
        ]

        # Store the original initializer values so they can be restored if modify_model=False
        tensors = [v.const_value for v in initializer_values]

        # Check that we are not overwriting the external data path that is currently
        # referenced by an initializer if we are not modifying the model
        for value in initializer_values:
            tensor = value.const_value
            if isinstance(tensor, _core.ExternalTensor) and os.path.samefile(
                tensor.path, os.path.join(base_dir, external_data)
            ):
                if not modify_model:
                    raise ValueError(
                        f"The external data path is currently referenced by an initializer ('{value}'). "
                        "Model will be incorrect if modify_model=False, because the original reference will "
                        "be invalid after the external data is overwritten. You can set modify_model=True, or "
                        "choose a different `external_data` path that is not currently referenced by the model."
                    )

        model = _external_data.to_external_data(model, base_dir, external_data)
        proto = serde.serialize_model(model)
        onnx.save(proto, path, format=format)

        if not modify_model:
            # Restore the original initializer values so the model is unchanged
            for initializer, tensor in zip(initializer_values, tensors):
                initializer.const_value = tensor

    else:
        proto = serde.serialize_model(model)
        onnx.save(proto, path, format=format)
