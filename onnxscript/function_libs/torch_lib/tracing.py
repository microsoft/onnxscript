"""Module for trace_only functions."""
from __future__ import annotations

import inspect
from typing import Callable, Optional
import typing
import onnx
from onnxscript import type_annotation
import onnxscript
from onnxscript._internal import version_utils

_ONNX_OP_SCHEMA_WRITABLE = not version_utils.onnx_older_than("1.14")

class TraceOnlyFunction:
    """TraceOnlyFunction.

    Attributes:
        name: Name of the op. E.g. "aten::add".
        func: Function.
    """

    def __init__(self, opset: onnxscript.values.Opset, func: Callable):
        self._opset = opset
        self._func = func
        self._opschema: Optional[onnx.defs.OpSchema] = None
        # Set the signature of the class to function's
        self.__signature__ = inspect.signature(func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self):
        return f"TraceOnlyFunction({self!r})"

    @property
    def name(self) -> str:
        """Return the name of the op."""
        return self._func.__name__

    @property
    def source(self) -> str:
        """Return the source of the op."""
        return inspect.getsource(self._func)

    @property
    def opschema(self) -> Optional[onnx.defs.OpSchema]:
        """Return the opschema."""

        if self._opschema is not None:
            return self._opschema

        if not _ONNX_OP_SCHEMA_WRITABLE:
            return None

        # TODO: Construct inputs and outputs and attrs

        # Find all distinct types in the inputs and outputs
        distinct_types = {arg.typeinfo for arg in function_ir.inputs}.union(
            {arg.typeinfo for arg in function_ir.outputs}
        )
        # Create a mapping from type to a unique name
        type_to_constraint = {}
        for i, type_ in enumerate(distinct_types):
            name = f"T{i}"
            type_to_constraint[type_] = onnxscript.values.TypeConstraint(
                name=type_annotation.get_type_constraint_name(type_) or name,
                allowed_types=type_annotation.pytype_to_input_strings(type_),
            )

        formal_inputs = [
            onnx.defs.OpSchema.FormalParameter(
                arg.name,
                type_to_constraint[arg.typeinfo].name,
                param_option=(
                    onnx.defs.OpSchema.FormalParameterOption.Optional
                    if type_annotation.is_optional(arg.typeinfo)
                    else onnx.defs.OpSchema.FormalParameterOption.Single
                ),
                # TODO(justinchu): Check this is_homogeneous thing
                is_homogeneous=True,
            )
            for arg in function_ir.inputs
        ]
        formal_outputs = [
            onnx.defs.OpSchema.FormalParameter(
                arg.name,
                type_to_constraint[arg.typeinfo].name,
                param_option=(
                    onnx.defs.OpSchema.FormalParameterOption.Optional
                    if type_annotation.is_optional(arg.typeinfo)
                    else onnx.defs.OpSchema.FormalParameterOption.Single
                ),
                # TODO(justinchu): Check this is_homogeneous thing
                is_homogeneous=True,
            )
            for arg in function_ir.outputs
        ]

        self._opschema = onnx.defs.OpSchema(
            self.name,
            self.opset.domain,
            since_version=self.opset.version,
            doc=self._func.__doc__ or "",
            inputs=formal_inputs,
            outputs=formal_outputs,
            type_constraints=[
                constraint.as_tuple() for constraint in type_to_constraint.values()
            ],
            attributes=[
                *[
                    onnx.defs.OpSchema.Attribute(
                        attr.name,
                        type=onnx.defs.OpSchema.AttrType(attr.type),
                    )
                    for attr in function_ir.attrs
                    if not attr.has_default
                ],
                *[
                    onnx.defs.OpSchema.Attribute(
                        attr.name,
                        default_value=attr.attr_proto,
                    )
                    for attr in function_ir.attrs
                    if attr.has_default
                ],
            ],
        )

        return self._opschema
