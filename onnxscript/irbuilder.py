# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import io
import logging
import warnings
from typing import Any, Optional, Protocol, Sequence, Union

import onnx
from onnx import ValueInfoProto, helper
from onnx.defs import onnx_opset_version

import onnxscript
from onnxscript import type_annotation as ta
from onnxscript import values
from onnxscript._internal import version_utils
from onnxscript.onnx_types import ONNXType
from onnxscript.sourceinfo import SourceInfo

from dataclasses import make_dataclass
import ast

import spox
from spox._public import argument as SpoxArgument

import spox.opset.ai.onnx.v17 as spox_operators
from spox._type_system import Type as SpoxType
from spox._var import Var as SpoxVar
from spox._attributes import _Ref as SpoxAttrRef
from spox._attributes import Attr as SpoxAttr
from spox._attributes import AttrInt64
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._node import Node as SpoxNode
from spox._node import OpType
from spox._scope import Scope as SpoxScope
from spox._scope import ScopeSpace

from spox._function import Function as SpoxFunction
import spox._function
from spox._graph import Graph as SpoxGraph
from spox._graph import results as GraphInit

import numpy as np

# A simple IR (Function, Stmt, Attr, Var):

logger = logging.getLogger("onnxscript")


def _format(seq: Sequence[Any], prefix: str, sep: str, suffix: str, formatter=str):
    """Formats a sequence of objects into a string."""
    return prefix + sep.join([formatter(x) for x in seq]) + suffix


def select_ir_version(version: int, domain: str = ""):
    """Selects a suitable ONNX ir_version for a given opset version."""
    if domain == "":
        domain = "ai.onnx"
    if (domain, version) not in helper.OP_SET_ID_VERSION_MAP:
        return max(v for k, v in helper.OP_SET_ID_VERSION_MAP.items() if k[0] == "ai.onnx")
    return helper.OP_SET_ID_VERSION_MAP[domain, version]


class IRType:
    def __init__(self):
        self.onnx_type = onnx.TypeProto()

    def to_type_proto(self):
        return self.onnx_type

    def __repr__(self) -> str:
        return "IRType()"


class IRTensorType(IRType):
    def __init__(self, elem_type: onnx.TensorProto.DataType) -> None:
        super().__init__()
        self.onnx_type.tensor_type.elem_type = elem_type

    def __repr__(self) -> str:
        return f"IRTensorType({self.onnx_type.tensor_type.elem_type})"


class IRTypeLike(Protocol):
    def to_type_proto(self) -> onnx.TypeProto:
        """Converts IR type representation to onnx.TypeProto"""

# wrapper around spox class instance
# implementation will wrap a spox type instance

class IRVar(SpoxVar):
    """A variable (representing a formal parameter)."""

    def __init__(
            self, varname: str,
            typeinfo: IRTypeLike,
            sourceinfo: SourceInfo,
            source_op: Optional[SpoxNode] = None,
        ) -> None:
        if not isinstance(varname, str):
            raise ValueError(f"varname must be a string not {type(varname)!r}.")
        self.name = varname
        self.info = sourceinfo
        self.typeinfo = typeinfo

        """Wrap IRVar and initialize SpoxVar"""
        if typeinfo is not None:
            type_ = SpoxType()._from_onnx(typeinfo.to_type_proto())
        else:
            type_ = SpoxType()
        #assert sourceinfo is not None or source_op is not None
        if sourceinfo and isinstance(sourceinfo.ast_node, ast.arg):
            node_ = SpoxArgument(type_)
        else:
            node_ = None
        if source_op:
            node_ = source_op
        super().__init__(node_, type_, None)
        self._rename(varname)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.typeinfo!r})"

    def typed_str(self):
        return f"{self.name} : {self.typeinfo}"

    def _to_value_info(self, use_default_type: bool = True):
        """Converts the content of this class into :class:`onnx.ValueInfoProto`.

        Args:
            use_default_type: if True, use a default type if an explicit type
                is not known. Otherwise, returns a ValueInfoProto without type.

        Returns:
            an instance of :class:`onnx.ValueInfoProto`
        """
        if self.name is None:
            raise ValueError(self.info.msg("name cannot be None."))
        value_info_proto = ValueInfoProto()
        value_info_proto.name = self.name
        if self.typeinfo is not None:
            value_info_proto.type.CopyFrom(self.typeinfo.to_type_proto())
        elif use_default_type:
            value_info_proto.type.CopyFrom(IRType().to_type_proto())
        return value_info_proto

    def to_value_info(self, use_default_type: bool = True):
        spox_value_info_proto = self.type._to_onnx_value_info(self.name)
        assert spox_value_info_proto == self._to_value_info(use_default_type)
        return spox_value_info_proto

def _opt_var_to_str(x):
    return "" if x is None else str(x)


# TODO: Wrap around other SpoxAttr classes such as AttrFloat, AttrString etc
# Only wrapped around AttrInt64 for dummy testing purpose
class IRAttributeValue(AttrInt64):
    """An attribute value (representing an actual parameter).

    Attributes:
        name: The name of the attribute.
        type: The type of the attribute.
        attr_proto: The attribute proto.
    """

    def __init__(self, attrproto: onnx.AttributeProto) -> None:
        if not attrproto.ref_attr_name:
            value = helper.get_attribute_value(attrproto)
        else:
            dummy_tens = AttrInt64(0)
            value = SpoxAttrRef(dummy_tens, attrproto.ref_attr_name)
        super().__init__(value)
        self.attr_proto = attrproto

    def __str__(self):
        if self.attr_proto.HasField("ref_attr_name"):
            return f"{self.attr_proto.name} = @{self.attr_proto.ref_attr_name}"
        # self.name + " = " + self.value
        return helper.printable_attribute(self.attr_proto)

    @property
    def name(self) -> str:
        return self.attr_proto.name

    @property
    def type(self) -> onnx.AttributeProto.AttributeType:
        return self.attr_proto.type


# TODO: Add this class equivalent to spox
@dataclasses.dataclass(frozen=True)
class IRAttributeParameter:
    """An attribute parameter (representing a formal parameter).

    It may or may not carry a default value.

    Attributes:
        name: The name of the attribute.
        type: The type of the attribute.
        default_value: The default value of the attribute.
        has_default: Whether the attribute has a default value.
        attr_proto: The attribute proto.
    """

    name: str
    type: onnx.AttributeProto.AttributeType
    default_value: str | int | float | None = None

    # TODO(justinchuby): Validate the default_value is the same type as specified in AttributeType.

    def __str__(self):
        if self.has_default:
            return helper.printable_attribute(self.attr_proto)
        # TODO(justinchuby): Include a readable type name.
        return self.name

    @property
    def has_default(self):
        return self.default_value is not None

    @property
    def attr_proto(self) -> onnx.AttributeProto:
        if not self.has_default:
            raise ValueError(
                "Attribute has no default value. Only attributes with default "
                "values can be converted to AttributeProto."
            )
        if version_utils.onnx_older_than("1.14.1"):
            # Argument 'attr_type' was added after version 1.14.0.
            return helper.make_attribute(self.name, self.default_value)
        # pylint: disable=unexpected-keyword-arg
        return helper.make_attribute(self.name, self.default_value, attr_type=self.type)  # type: ignore[call-arg]
        # pylint: enable=unexpected-keyword-arg


class IRStmt(SpoxNode):
    def __init__(
        self,
        result: Sequence[str],
        callee: values.Op,
        args: Sequence[Optional[str]],
        attrs: Sequence[IRAttributeValue],
        sub_functions=None,
    ) -> None:
        if not isinstance(callee, values.Op):
            raise TypeError(f"Unexpected type {type(callee)} for callee.")
        self.result = result
        self.callee = callee
        self.args = args
        self._attrs = attrs
        self.functions = sub_functions or {}

        """Wrap IRStmt and initialize SpoxNode"""
        spox_attrs = self.convert_to_spox_attrs(attrs)
        spox_inputs = self.create_spox_inputs(args)
        spox_outputs = self.create_spox_outputs(result)
        super().__init__(spox_attrs, spox_inputs, spox_outputs)
        self.op_type = OpType(callee._name, callee._opset.domain, callee._opset.version)
        # Populate spox scope variable
        what = {}
        for i in self.inputs:
            what[i] = str(i)
        for i in self.outputs:
            what[i] = str(i)
        self.scope = SpoxScope()
        vscope = ScopeSpace(name_of=what)
        self.scope.var = vscope
        nscope = ScopeSpace()
        self.scope.node = nscope

    def convert_to_spox_attrs(self, attrs) -> BaseAttributes:
        spox_attrs = BaseAttributes()
        fields = []
        for a in attrs:
            # TODO: Set tensor type based on Attr Tensor Type
            # Testing only int64 types for dummy testing
            fields.append((a.attr_proto.name, AttrInt64))
        spox_attrs.__class__ = make_dataclass('Attributes', fields=fields, bases=(BaseAttributes,))
        for a in attrs:
            setattr(spox_attrs, a.attr_proto.name, a)
        return spox_attrs

    def create_spox_inputs(self, args) -> BaseInputs:
        spox_inputs = BaseInputs()
        fields = []
        for a in args:
            fields.append((a, IRVar))
        spox_inputs.__class__ = make_dataclass('Inputs', fields=fields, bases=(BaseInputs,))
        for a in args:
            # TODO: Set tensor type based on Var Type
            # Testing only float types for dummy testing
            irtype = IRTensorType(1)
            irvar = IRVar(a, irtype, None)
            setattr(spox_inputs, a, irvar)
        return spox_inputs

    def create_spox_outputs(self, args) -> BaseOutputs:
        spox_inputs = BaseInputs()
        fields = []
        for a in args:
            fields.append((a, IRVar))
        spox_inputs.__class__ = make_dataclass('Outputs', fields=fields, bases=(BaseOutputs,))
        for a in args:
            # TODO: Set tensor type based on Var Type
            # Testing only float types for dummy testing
            irtype = IRTensorType(1)
            irvar = IRVar(a, irtype, None)
            setattr(spox_inputs, a, irvar)
        return spox_inputs

    def __str__(self):
        if isinstance(self.result, str):
            logger.debug("unexpected str type for self.result where type(self)=%r", type(self))
        lhs = ", ".join(self.result)
        attrs = ""
        if self._attrs:
            attrs = _format(self._attrs, "<", ", ", ">")

        args = _format(self.args, "(", ", ", ")", _opt_var_to_str)
        domain = self.callee.opset.domain
        opname = self.callee.name
        callee = f"{domain}.{opname}" if (domain != "") else opname
        return f"{lhs} = {callee} {attrs}{args}"

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: %s", type(self), str(self))

    def _to_node_proto(self, node_name: str) -> onnx.NodeProto:
        n = helper.make_node(
            self.callee.name,
            [_opt_var_to_str(x) for x in self.args],
            [str(x) for x in self.result],
            domain=self.callee.opset.domain,
            name=node_name,
        )
        for a in self._attrs:
            n.attribute.append(a.attr_proto)
        return n

    def to_node_proto(self, node_name: str) -> onnx.NodeProto:
        self.scope.node.__setitem__(self, node_name)
        spox_n = self.to_onnx(self.scope)[0]
        assert spox_n == self._to_node_proto(node_name)
        return spox_n

    @property
    def output_names(self) -> Sequence[str]:
        """Returns the list of variables assigned to by this statement."""
        return [str(x) for x in self.result]


class IRFunction(SpoxFunction):
    """Represents a function in the IR."""

    def __init__(self, name: str, domain: str = "") -> None:
        self.domain = domain
        self.name = name
        self._outputs: list[IRVar] = []
        self.stmts: list[IRStmt] = []
        self.called_functions: dict[str, onnx.FunctionProto] = {}
        self.docstring: str = ""
        # a dictionary of nested function-definitions
        self.nested_functions: dict[str, IRFunction] = {}
        self.outer_scope_variables: dict[Any, Any] = {}
        self.ordered_inputs_and_attrs: list[Union[IRVar, IRAttributeParameter]] = []

    @property
    def assigned_names(self) -> Sequence[str]:
        """Returns the list of variables assigned to by this function."""
        return [v for stmt in self.stmts for v in stmt.output_names]

    @property
    def _inputs(self) -> Sequence[IRVar]:
        return [var for var in self.ordered_inputs_and_attrs if isinstance(var, IRVar)]

    @property
    def _attrs(self) -> Sequence[IRAttributeParameter]:
        return [
            attr
            for attr in self.ordered_inputs_and_attrs
            if isinstance(attr, IRAttributeParameter)
        ]

    def __str__(self):
        attrs = _format(self._attrs, "<", ", ", ">") if self._attrs else ""
        inputs = _format([x.typed_str() for x in self._inputs], "(", ", ", ")")
        outputs = _format([x.typed_str() for x in self._outputs], "(", ", ", ")")
        stmts = _format(self.stmts, "\n{\n   ", "\n   ", "\n}\n")
        return f"{self.name} {attrs}{inputs} => {outputs}{stmts}"

    def append_docstring(self, docstring):
        self.docstring += docstring

    def append_stmt(self, stmt: IRStmt) -> None:
        self.stmts.append(stmt)

    def append_input(self, name: IRVar) -> None:
        self.ordered_inputs_and_attrs.append(name)

    def append_output(self, name: IRVar) -> None:
        self._outputs.append(name)

    def add_attr_parameter(self, attr: IRAttributeParameter) -> None:
        self.ordered_inputs_and_attrs.append(attr)

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            st = io.StringIO()
            for s in self.stmts:
                for attr in s.attrs:
                    if attr.attr_proto.HasField("g"):
                        st.write(helper.printable_graph(attr.attr_proto.g))
                        st.write("\n")

    def add_called_function(self, fun: values.OnnxFunction) -> None:
        for name, fct in fun.function_ir.called_functions.items():
            if name in self.called_functions:
                continue
            self.called_functions[name] = fct
        if fun.name in self.called_functions:
            # Already added.
            return
        try:
            proto = fun.to_function_proto()
        except (TypeError, AttributeError) as e:
            raise TypeError(f"Issue with type f{type(fun)}.") from e
        self.called_functions[fun.name] = proto

    def add_nested_function(self, fun: IRFunction) -> None:
        self.nested_functions[fun.name] = fun

    def to_model_proto(
        self,
        functions=None,
        io_types: Optional[ONNXType] = None,
        input_types: Optional[Sequence[ONNXType]] = None,
        output_types: Optional[Sequence[ONNXType]] = None,
        **kwargs,
    ) -> onnx.ModelProto:
        """Converts this instance into a `onnx.ModelProto`.

        Args:
            functions: A list of functions to include in the model.
                By default, all functions called at least once are included.
            io_types: When specified, all the inputs/outputs of the model
                are set to be of this type.
            input_types: When specified, all the inputs of the model
                are set to be of the corresponding type in this list.
            output_types: When specified, all the outputs of the model
                are set to be of the corresponding type in this list.
            kwargs: Additional parameters given to function :func:`onnx.helper.make_model`.

        Returns:
            An instance of :class:`onnx.ModelProto`.
        """
        graph, sub_functions = self.to_graph_and_functions(use_default_type=False)
        if io_types is not None:
            for input in graph.input:
                if not input.HasField("type"):
                    input.type.CopyFrom(io_types.to_type_proto())
            for output in graph.output:
                if not output.HasField("type"):
                    output.type.CopyFrom(io_types.to_type_proto())
        if input_types is not None:
            for input, type in zip(graph.input, input_types):
                input.type.CopyFrom(type.to_type_proto())
        if output_types is not None:
            for output, type in zip(graph.output, output_types):
                output.type.CopyFrom(type.to_type_proto())
        if functions is None:
            functions = sub_functions.values()
        else:

            def to_proto(f):
                if isinstance(f, onnx.FunctionProto):
                    return f
                if isinstance(f, onnxscript.OnnxFunction):
                    return f.to_function_proto()
                raise TypeError("Expected a value of type FunctionProto of OnnxFunction")

            functions = [to_proto(f) for f in functions]

        opsets = {}
        for n in self.stmts:
            if n.callee.opset.domain not in opsets:
                opsets[n.callee.opset.domain] = n.callee.opset.version
        if "" not in opsets:
            # No operator is using the standard opset.
            # A default value is given.
            opsets[""] = onnx_opset_version()
        for proto in functions:
            if proto.domain not in opsets:
                opsets[proto.domain] = 1

        if "ir_version" not in kwargs:
            kwargs["ir_version"] = select_ir_version(opsets[""])
        opset_imports = [
            onnx.helper.make_opsetid(domain, version) for domain, version in opsets.items()
        ]

        return helper.make_model(
            graph, opset_imports=opset_imports, functions=functions, **kwargs
        )

    def _to_graph_and_functions(
        self, use_default_type: bool = True
    ) -> tuple[onnx.GraphProto, dict[str, onnx.FunctionProto]]:
        """Converts this instance into a `onnx.GraphProto` and a map from
        function-name to `onnx.FunctionProto`.

        Args:
            use_default_type: if True, the function uses a default type
                for inputs and outputs that do not have a type

        Returns:
            a pair of a :class:`onnx.GraphProto` and list of :class:`onnx.FunctionProto`
        """
        called_functions: dict[str, onnx.FunctionProto] = {}
        for s in self.stmts:
            called_functions.update(s.functions)
        called_functions.update(self.called_functions)
        graph = helper.make_graph(
            [s.to_node_proto(f"n{i}") for i, s in enumerate(self.stmts)],
            self.name,
            [x.to_value_info(use_default_type) for x in self._inputs],
            [y.to_value_info(use_default_type) for y in self._outputs],
        )
        return graph, called_functions

    def to_graph_and_functions(
        self, use_default_type: bool = True
    ) -> tuple[onnx.GraphProto, dict[str, onnx.FunctionProto]]:
        built_graph = IRGraph(self.name, self)
        graph = built_graph.to_onnx()
        called_functions: dict[str, onnx.FunctionProto] = {}
        for s in self.stmts:
            called_functions.update(s.functions)
        called_functions.update(self.called_functions)
        #return self._to_graph_and_functions()
        return graph, called_functions

    def to_graph_proto(self, use_default_type: bool = True) -> onnx.GraphProto:
        """Converts this instance into a `onnx.GraphProto`.

        Args:
            use_default_type: if True, the function uses a default type
                for inputs and outputs that do not have a type

        Returns:
            an instance of :class:`onnx.GraphProto`
        """
        graph, _ = self.to_graph_and_functions(use_default_type=use_default_type)
        return graph

    def get_opset_import(self) -> dict[str, int]:
        func_opset_imports = {}
        for s in self.stmts:
            if s.callee.opset.domain not in func_opset_imports:
                func_opset_imports[s.callee.opset.domain] = s.callee.opset.version
            elif func_opset_imports[s.callee.opset.domain] != s.callee.opset.version:
                warnings.warn(
                    f"There is a version conflict in domain: {s.callee.opset.domain!r}, "
                    f"with {self.name!r}.",
                    category=UserWarning,
                    stacklevel=1,
                )
        return func_opset_imports

    def convert_to_spox_attrs(self, attrs) -> BaseAttributes:
        spox_attrs = BaseAttributes()
        fields = []
        '''for a in attrs:
            # TODO: Works for attrs with default_value only
            # Testing only int64 types for dummy testing
            fields.append((a.attr_proto.name, AttrInt64))
        spox_attrs.__class__ = make_dataclass('Attributes', fields=fields, bases=(BaseAttributes,))
        for a in attrs:
            setattr(spox_attrs, a.attr_proto.name, IRAttributeValue(a.attr_proto))'''
        for a in attrs:
            fields.append((a.name, SpoxAttrRef))
        spox_attrs.__class__ = make_dataclass('Attributes', fields=fields, bases=(BaseAttributes,))
        for a in attrs:
            dummy_tens = AttrInt64(0)
            value = SpoxAttrRef(dummy_tens, a.name)
            setattr(spox_attrs, a.name, value)
        return spox_attrs

    def _to_function_proto(self) -> onnx.FunctionProto:
        """Converts this instance into a `onnx.FunctionProto`.

        Note: Default values for attributes are an experimental feature in ONNX.
        Conversion ignores default values for attributes if the ONNX version installed
        doesn't support it.
        """
        opsets = self.get_opset_import()
        nodes = [s.to_node_proto(f"n{i}") for i, s in enumerate(self.stmts)]
        for n in nodes:
            if n.domain not in opsets:
                opsets[n.domain] = 1  # TODO: how to get n.version?
        opset_imports = [
            onnx.helper.make_opsetid(domain, version) for domain, version in opsets.items()
        ]

        attribute_names = [attr.name for attr in self._attrs if not attr.has_default]

        f = helper.make_function(
            self.domain,
            self.name,
            inputs=[x.name for x in self._inputs],
            outputs=[y.name for y in self._outputs],
            nodes=nodes,
            opset_imports=opset_imports,  # TODO
            attributes=attribute_names,
            doc_string=self.docstring,
        )
        if hasattr(onnx.FunctionProto, "attribute_proto"):
            f.attribute_proto.extend(
                [attr.attr_proto for attr in self._attrs if attr.has_default]
            )
        return f

    def to_function_proto(self):
        self.func = IRGraph(self.name, self)
        fn_attrs = self.convert_to_spox_attrs(self._attrs)
        fn_inputs, fn_outputs = self.func.dcs

        spox_fn = SpoxFunction(fn_attrs, fn_inputs, fn_outputs)
        spox_fn.op_type = OpType(
            self.name, self.domain, self.get_opset_import()
        )
        setattr(spox_fn, 'func_inputs', fn_inputs)
        setattr(spox_fn, 'func_outputs', fn_outputs)
        setattr(spox_fn, 'func_attrs', dict())
        setattr(spox_fn, 'func_graph', self.func.graph)
        #return self._to_function_proto()
        return spox_fn.to_onnx_function()


class IRGraph(SpoxGraph):
    def __init__(
        self,
        name: str,
        onnx_function: IRFunction,
        domain: str = "",
    ):
        self.graph = None
        self.fn = onnx_function
        self.sub_functions = None
        self.domain = domain
        self.name = name
        self.args = []
        self.results = {}
        self.dcs = []

        self.build_graph()
        super().__init__(
            self.results,
            _name = self.name,
            _arguments = tuple(self.args),
            _extra_opset_req = set(()),
        )

    def _prepare_inputs_outputs_for_spox(self):
        """Create dataclasses for spox inputs/outputs"""
        # Inputs
        spox_inputs = BaseInputs()
        fields = [(inp.name, SpoxVar) for inp in self.fn._inputs]
        spox_inputs.__class__ = make_dataclass('_FuncInputs', fields=fields, bases=(BaseInputs,))
        for inp in self.fn._inputs:
            spox_arg = SpoxArgument(inp.type)
            setattr(spox_inputs, inp.name, spox_arg)
            self.args.append(spox_arg)
        # Outputs
        spox_outputs = BaseOutputs()
        fields = [(out.name, SpoxVar) for out in self.fn._outputs]
        spox_outputs.__class__ = make_dataclass('_FuncOutputs', fields=fields, bases=(BaseOutputs,))
        return spox_inputs, spox_outputs

    def chain_graph_nodes(
        self,
        graph_inputs: BaseInputs,
        graph_outputs: BaseOutputs,
    ):
        var_stack = graph_inputs.get_fields()
        node_stack = {}
        for stmt in self.fn.stmts:
            # create inputs
            nd_inputs = BaseInputs()
            fields = [(d.name, SpoxVar) for d in stmt.dependencies]
            nd_inputs.__class__ = make_dataclass('Inputs', fields=fields, bases=(BaseInputs,))
            for d in stmt.dependencies:
                setattr(nd_inputs, d.name, var_stack[d.name])

            # create outputs
            nd_outputs = BaseOutputs()
            fields = [(d.name, SpoxVar) for d in stmt.dependents]
            nd_outputs.__class__ = make_dataclass('Outputs', fields=fields, bases=(BaseOutputs,))
            for d in stmt.dependents:
                type_ = SpoxType()._from_onnx(d.typeinfo.to_type_proto())
                var = SpoxVar(stmt, type_)
                setattr(nd_outputs, d.name, var)

            # fill outputs with _op information
            spox_node = SpoxNode(stmt.attrs, nd_inputs, nd_outputs)
            spox_node.op_type = OpType(stmt.callee._name, stmt.callee._opset.domain, stmt.callee._opset.version)
            for d in stmt.dependents:
                type_ = SpoxType()._from_onnx(d.typeinfo.to_type_proto())
                var = SpoxVar(spox_node, type_)
                setattr(spox_node.outputs, d.name, var)
                var_stack[d.name] = var

            node_stack[stmt.callee.name] = spox_node
        for o in self.fn._outputs:
            setattr(graph_outputs, o.name, var_stack[o.name])
            self.results[o.name] = var_stack[o.name]
        self.dcs = [graph_inputs, graph_outputs]
        return graph_outputs

    def build_graph(self):
        graph_inputs, graph_outputs = self._prepare_inputs_outputs_for_spox()
        chained_outs = self.chain_graph_nodes(graph_inputs, graph_outputs)
        # Build graph object
        self.graph = spox._graph.results(**chained_outs.get_vars()).with_arguments(
            *graph_inputs.get_fields().values()
        ).with_name(self.name)

# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:
    def __init__(self):
        self.functions = {}

    def new_function(self, name: str, domain: str = "", register: bool = False):
        if register and (domain, name) in self.functions:
            raise RuntimeError(f"Function '{name}' already exists in domain '{domain}'.")
        function = IRFunction(name, domain)
        if register:
            self.functions[domain, name] = function
        return function

    def add_docstring(self, fn: IRFunction, docstring: str):
        fn.append_docstring(docstring)

    def add_stmt(
        self,
        fn: IRFunction,
        results: Sequence[str],
        callee: values.Op,
        args: Sequence[Optional[str]],
        attrs: Sequence[IRAttributeValue],
        sub_functions=None,
    ) -> None:
        stmt = IRStmt(results, callee, args, attrs, sub_functions=sub_functions)
        fn.append_stmt(stmt)

    def add_input(
        self, fn: IRFunction, varname: str, type: IRTypeLike, info: SourceInfo
    ) -> None:
        var = IRVar(varname, type, info)
        fn.append_input(var)

    def add_attr_parameter(
        self,
        fn: IRFunction,
        varname: str,
        attribute_type: onnx.AttributeProto.AttributeType,
        default_value: int | float | str | None,
    ) -> None:
        fn.add_attr_parameter(IRAttributeParameter(varname, attribute_type, default_value))

    def add_output(self, fn: IRFunction, varname: str, typeinfo, sourceinfo) -> None:
        var = IRVar(varname, typeinfo, sourceinfo)
        fn.append_output(var)

    def make_attr(self, attrname: str, attrval: Any) -> IRAttributeValue:
        return IRAttributeValue(helper.make_attribute(attrname, attrval))

    def make_attr_ref(self, attrname: str, refname: str, pytype: type) -> IRAttributeValue:
        proto = onnx.AttributeProto()
        proto.name = attrname
        proto.ref_attr_name = refname
        attr_type = ta.pytype_to_attrtype(pytype)
        assert attr_type is not None
        proto.type = attr_type
        return IRAttributeValue(proto)
