# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing the IR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from onnx_ir import tape

if TYPE_CHECKING:
    import onnx_ir as ir


# A type representing the domains/versions used in creating nodes in IR.
UsedOpsets = set[tuple[str, Optional[int]]]


class Builder(tape.Tape):
    """An extension of the tape that provides a more convenient API for constructing the IR."""

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)
        if isinstance(outputs, Sequence):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, int)
            num_outputs = outputs

        if num_outputs == 1:
            value = super().op(
                op_type, inputs=inputs, attributes=kwargs, domain=domain, version=version
            )
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = super().op_multi_out(
            op_type,
            inputs=inputs,
            attributes=kwargs,
            domain=domain,
            version=version,
            num_outputs=num_outputs,
        )
        if isinstance(outputs, Sequence):
            for value, name in zip(values, outputs):
                value.name = name
        return values
