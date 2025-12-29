# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import warnings
from typing import Any, Sequence, Union

import onnx
import onnx_ir as ir

from onnxscript._internal import type_annotation, values

logger = logging.getLogger("onnxscript")

TypeAnnotationValue = type_annotation.TypeAnnotationValue


class IRFunction(ir.Function):
    """Represents a function in the IR."""

    def __init__(self, name: str, domain: str = "") -> None:
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name=name)
        super().__init__(domain, name, graph=graph, attributes=[])
        self.ordered_inputs_and_attrs: list[Union[ir.Value, ir.Attr]] = []

        # A dictionary of nested function-definitions: when an onnxscript function outer_f
        # is translated, and it contains a nested function inner_f, then the inner function
        # is translated and stored here. It will be used in any subsequent concrete execution
        # of outer_f. Such nested functions are used in two different ways: it can be converted
        # into a GraphProto to be stored as a graph-valued attribute of a node; alternatively,
        # in a python-based execution mode, it can be called as a python function. It serves
        # to enable a python-based debugging experience for higher-order functions such as Scan
        # and SequenceMap.
        self.nested_functions: dict[str, IRFunction] = {}

        # For nested functions, this dictionary maps outer-scope (python) variable names
        # to their corresponding translated values.
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
