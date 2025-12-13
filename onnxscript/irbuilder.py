# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa: TID251
from __future__ import annotations

import logging
import warnings
from typing import Any, Sequence, Union

import onnx
import onnx_ir as ir

import onnxscript.type_annotation
from onnxscript import values
from onnxscript.sourceinfo import SourceInfo

logger = logging.getLogger("onnxscript")


def select_ir_version(version: int, domain: str = "") -> int:
    """Selects a suitable ONNX ir_version for a given opset version."""
    if domain == "":
        domain = "ai.onnx"
    if (domain, version) not in onnx.helper.OP_SET_ID_VERSION_MAP:
        return max(
            v for k, v in onnx.helper.OP_SET_ID_VERSION_MAP.items() if k[0] == "ai.onnx"
        )
    return onnx.helper.OP_SET_ID_VERSION_MAP[domain, version]


TypeAnnotationValue = onnxscript.type_annotation.TypeAnnotationValue


class IRFunction(ir.Function):
    """Represents a function in the IR."""

    def __init__(self, name: str, domain: str = "") -> None:
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name=name)
        super().__init__(domain, name, graph=graph, attributes=[])
        self.ordered_inputs_and_attrs: list[Union[ir.Value, ir.Attr]] = []

        # a dictionary of nested function-definitions
        self.nested_functions: dict[str, IRFunction] = {}
        self.outer_scope_variables: dict[Any, Any] = {}

    @property
    def docstring(self) -> str:
        """Returns the docstring of this function."""
        return self.doc_string or ""

    @property
    def assigned_names(self) -> Sequence[str]:
        """Returns the list of variables assigned to by this function."""
        return [v.name for n in self for v in n.outputs]

    @property
    def attrs(self) -> Sequence[ir.Attr]:
        return [attr for attr in self.ordered_inputs_and_attrs if isinstance(attr, ir.Attr)]

    def append_node(self, node: ir.Node) -> None:
        count = len(self)
        node.name = f"n{count}"
        self.append(node)
        domain = node.domain
        version = node.version
        if domain not in self.opset_imports:
            self.opset_imports[domain] = version
        else:
            existing_version = self.opset_imports[domain]
            if existing_version != version:
                warnings.warn(
                    f"Version conflict: domain: {domain!r}, "
                    f"versions {existing_version} and {version} used.",
                    category=UserWarning,
                    stacklevel=2,
                )

    def append_parameter(self, parameter: ir.Value | ir.Attr) -> None:
        self.ordered_inputs_and_attrs.append(parameter)
        if isinstance(parameter, ir.Value):
            self.inputs.append(parameter)
        else:
            if not isinstance(parameter, ir.Attr):
                raise TypeError(f"Expected ir.Value or ir.Attr, got {type(parameter)}")
            self.attributes.add(parameter)

    def append_output(self, var: ir.Value) -> None:
        self.outputs.append(var)

    def add_nested_function(self, fun: IRFunction) -> None:
        self.nested_functions[fun.name] = fun

    def get_called_functions(self) -> dict[str, onnx.FunctionProto]:
        called_functions: dict[str, values.OnnxFunction] = {}

        def visit(function_ir: IRFunction):
            for node in ir.traversal.RecursiveGraphIterator(function_ir.graph):
                callee = node.meta.get("callee", None)
                if isinstance(callee, values.OnnxFunction):
                    add(callee)

        def add(f: values.OnnxFunction):
            if f.name in called_functions:
                return
            called_functions[f.name] = f
            visit(f.function_ir)

        visit(self)

        return {name: f.to_function_proto() for name, f in called_functions.items()}

    def to_graph_proto(self) -> onnx.GraphProto:
        """Converts this instance into a `onnx.GraphProto`."""
        return ir.to_proto(self.graph)

    def to_function_proto(self) -> onnx.FunctionProto:
        """Converts this instance into a `onnx.FunctionProto`."""
        return ir.to_proto(self)


# IRBuilder: abstracts out details of the IR in the python-to-IR converter


def set_type_info(value: ir.Value, typeinfo: TypeAnnotationValue) -> None:
    """Sets the type information on an IR value."""
    try:
        type_and_shape = ir.from_proto(typeinfo.to_type_proto())
        value.type = type_and_shape.type
        value.shape = type_and_shape.shape
    except AttributeError:
        pass
    value.meta["typeinfo"] = typeinfo


def make_value(
    varname: str, typeinfo: TypeAnnotationValue, sourceinfo: SourceInfo
) -> ir.Value:
    value = ir.Value(name=varname)
    value.meta.setdefault("sourceinfo", sourceinfo)
    if typeinfo is not None:
        set_type_info(value, typeinfo)
    return value


class IRBuilder:
    def __init__(self):
        self.functions = {}
