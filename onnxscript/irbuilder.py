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

