# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from io import StringIO
import onnx
import onnx.helper as helper
from onnx.defs import onnx_opset_version
from . import type_annotation as ta
from .values import OnnxFunction, Opset

# A simple IR (Function, Stmt, Attr, Var):

logger = logging.getLogger("onnx-script")


def format(list, prefix, sep, suffix, formatter=str):
    return prefix + sep.join([formatter(x) for x in list]) + suffix


class Type:

    def __init__(self):
        self.onnx_type = onnx.TypeProto()

    def to_type_proto(self):
        return self.onnx_type

    def __repr__(self) -> str:
        return "Type()"


class TensorType(Type):

    def __init__(self, elem_type) -> None:
        tp = onnx.TypeProto()
        tp.tensor_type.elem_type = elem_type
        self.onnx_type = tp

    def __repr__(self) -> str:
        return "TensorType(%d)" % self.onnx_type.tensor_type.elem_type


class Var:
    def __init__(self, varname, typeinfo, info) -> None:
        self.name = varname
        self.info = info
        self.typeinfo = typeinfo

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.typeinfo)

    def typed_str(self):
        return self.name + " : " + str(self.typeinfo)

    def to_value_info(self, enforce_typed=False, default_type=None):
        """
        Converts the content of this class into :class:`onnx.ValueInfoProto`.

        :param enforce_typed: if True, the function raises an exception if
            the type of an input or output is not specified (no annotation)
            unless *io_types* defined a default value to use
        :param default_type: defines a default value for missing input and output type,
            this is only used if *enforce_typed* is True
        :return: an instance of :class:`onnx.ValueInfoProto`
        """
        if self.typeinfo is None:
            if enforce_typed:
                if default_type is None:
                    raise TypeError(self.info.msg(
                        "Variable %r is missing an annotation and default_type "
                        "is not specified." % self.name))
                return helper.make_value_info(self.name, default_type.to_type_proto())
            return helper.make_value_info(self.name, Type().to_type_proto())
        tp = self.typeinfo.to_type_proto()
        # if (not tp.tensor_type.HasField('shape')):
        #     # TODO: temporary patch to export a function as a graph
        #     tp = helper.make_tensor_type_proto(tp.tensor_type.elem_type, [10])
        return helper.make_value_info(self.name, tp)


def opt_var_to_str(x):
    return "" if x is None else str(x)


class Attr:
    def __init__(self, attrproto) -> None:
        self.attr_proto = attrproto

    def __str__(self):
        if (self.attr_proto.HasField("ref_attr_name")):
            return self.attr_proto.name + " = @" + self.attr_proto.ref_attr_name
        # self.name + " = " + self.value
        return helper.printable_attribute(self.attr_proto)


class Stmt:
    def __init__(self, result, module, opname, args, attrs, sub_functions=None) -> None:
        if not isinstance(module, Opset):
            raise TypeError(f"Unexpected type {type(module)} for module.")
        if not isinstance(opname, str):
            raise TypeError(f"Unexpected type {type(opname)} for opname.")
        self.result = result
        self.module = module
        self.opname = opname
        self.args = args
        self.attrs = attrs
        self.functions = sub_functions or {}

    def __str__(self):
        if (isinstance(self.result, str)):
            logger.debug("unexpected str type for self.result where type(self)=%r",
                         type(self))
        lhs = ", ".join(self.result)
        attrs = ""
        if (self.attrs):
            attrs = format(self.attrs, "<", ", ", ">")

        args = format(self.args, "(", ", ", ")", opt_var_to_str)
        module = str(self.module)
        callee = module + "." + self.opname if (module != '') else self.opname
        return lhs + " = " + callee + " " + attrs + args

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: %s", type(self), str(self))

    def to_node_proto(self):
        if not isinstance(self.module.domain, str):
            raise TypeError("Unexpected type %r for self.module." % type(self.module))
        n = helper.make_node(self.opname,
                             [opt_var_to_str(x) for x in self.args],
                             [str(x) for x in self.result],
                             domain=self.module.domain)
        for a in self.attrs:
            n.attribute.append(a.attr_proto)
        return n


class Function:
    def __init__(self, name, domain="") -> None:
        self.domain = domain
        self.name = name
        self.inputs = []
        self.outputs = []
        self.stmts = []
        self.attrs = []
        self.attr_protos = []
        self.functions = {}
        self.docstring = ""

    def __str__(self):
        attrs = format(self.attrs, "<", ", ", ">") if self.attrs else ""
        attr_protos = format(self.attr_protos, "<", ", ", ">") if self.attr_protos else ""
        inputs = format([x.typed_str() for x in self.inputs], "(", ", ", ")")
        outputs = format([x.typed_str() for x in self.outputs], "(", ", ", ")")
        stmts = format(self.stmts, "\n{\n   ", "\n   ", "\n}\n")
        return (self.name + " " + attrs + attr_protos + inputs + " => " + outputs + stmts)

    def append_docstring(self, docstring):
        self.docstring += docstring

    def append_stmt(self, stmt):
        self.stmts.append(stmt)

    def append_input(self, name):
        self.inputs.append(name)

    def append_output(self, name):
        self.outputs.append(name)

    def append_attr(self, attr):
        self.attrs.append(attr)

    def append_attr_proto(self, attr):
        self.attr_protos.append(attr)

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            st = StringIO()
            for s in self.stmts:
                for attr in s.attrs:
                    if attr.attr_proto.HasField("g"):
                        st.write(helper.printable_graph(attr.attr_proto.g))
                        st.write("\n")

    def append_function(self, opf):
        for name, fct in opf.function_ir.functions.items():
            if name in self.functions:
                continue
            self.functions[name] = fct
        if opf.name in self.functions:
            # Already added.
            return
        try:
            proto = opf.to_function_proto(opf.opset)
        except (TypeError, AttributeError) as e:
            raise TypeError(f"Issue with type f{type(opf)}.") from e
        self.functions[opf.name] = proto

    def to_model_proto(self, functions=None, io_types=None, **kwargs):
        """
        Converts the content of this class into a `onnx.ModelProto`.

        :param functions: list of functions to include in the model,
            by default, all functions called at least once are included
        :param io_types: many functions are written without any type specification
            so they can be type agnostic. However, ModelProto requires the inputs
            and outputs to be strongly typed. When an input or an output has no type,
            this default value is used.
        :param kwargs: additional parameters given to function :func:`onnx.helper.make_model`
        :return: an instance of :class:`onnx.ModelProto`
        """
        graph, sub_functions = self.to_graph_proto(enforce_typed=True, io_types=io_types)
        if functions is None:
            functions = sub_functions.values()
        else:
            def to_proto(f):
                if isinstance(f, onnx.FunctionProto):
                    return f
                if isinstance(f, OnnxFunction):
                    return f.to_function_proto()
                raise TypeError("Expected a value of type FunctionProto of OnnxFunction")
            functions = [to_proto(f) for f in functions]

        opsets = {}
        for n in self.stmts:
            if n.module.domain not in opsets:
                opsets[n.module.domain] = n.module.version
        if '' not in opsets:
            # No operator is using the standard opset.
            # A default value is given.
            opsets[''] = onnx_opset_version()
        for proto in functions:
            if proto.domain not in opsets:
                opsets[proto.domain] = 1

        opset_imports = [onnx.helper.make_opsetid(domain, version)
                         for domain, version in opsets.items()]

        return helper.make_model(graph, opset_imports=opset_imports,
                                 functions=functions, **kwargs)

    def to_graph_proto(self, enforce_typed=False, io_types=None):
        """
        Converts the content of this class into a `onnx.GraphProto`.

        :param enforce_typed: if True, the function raises an exception if
            the type of an input or output is not specified (no annotation)
            unless *io_types* defined a default value to use
        :param io_types: defines a default value for missing input and output type,
            this is only used if *enforce_typed* is True
        :return: an instance of :class:`onnx.GraphProto`
        """
        sub_functions = {}
        for s in self.stmts:
            sub_functions.update(s.functions)
        sub_functions.update(self.functions)
        graph = helper.make_graph(
            [s.to_node_proto() for s in self.stmts],
            self.name,
            [x.to_value_info(enforce_typed, default_type=io_types) for x in self.inputs],
            [y.to_value_info(enforce_typed, default_type=io_types) for y in self.outputs])
        return graph, sub_functions

    def to_function_proto(self, domain):
        """
        Converts a function into a *FunctionProto* after it is parsed
        by the converter.

        .. warning:: About default values

            Default values for attributes are introduced in onnx==1.13.0.
            If an earlier version of onnx is installed, it ignores the default
            values of the function arguments.
        """
        opsets = {'': 15}
        if domain != '':
            opsets[domain.domain] = domain.version
        else:
            opsets = opsets.copy()
        nodes = [s.to_node_proto() for s in self.stmts]
        for n in nodes:
            if n.domain not in opsets:
                opsets[n.domain] = 1  # TODO: how to get n.version?
        opset_imports = [onnx.helper.make_opsetid(domain, version)
                         for domain, version in opsets.items()]

        # attribute_proto is introduced in version onnx==1.13.0.
        # If this attribute is available, onnx-script uses it to
        # default values for attributes. The function has then two
        # lists, one list for attributes without default values,
        # another one for attributes with default values.
        # If this *attribute_proto* is not available,
        # all attributes with a default value are moved to the first
        # list, default values are removed.
        # TODO: remove this when onnx==1.13.0 is released.
        if hasattr(onnx.FunctionProto, 'attribute_proto'):
            atts = [a.name for a in self.attrs]
        else:
            atts = ([a.name for a in self.attrs] +
                    [a.attr_proto.name for a in self.attr_protos])

        f = helper.make_function(
            self.domain,
            self.name,
            inputs=[x.name for x in self.inputs],
            outputs=[y.name for y in self.outputs],
            nodes=nodes,
            opset_imports=opset_imports,  # TODO
            attributes=atts,
            doc_string=self.docstring)
        if hasattr(onnx.FunctionProto, 'attribute_proto'):
            f.attribute_proto.extend([a.attr_proto for a in self.attr_protos])
        return f

# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:

    def __init__(self):
        self.functions = {}

    def new_function(self, name, domain="", register=False):
        if register and (domain, name) in self.functions:
            raise RuntimeError(f"Function '{name}' already exists in domain '{domain}'.")
        fct = Function(name, domain)
        if register:
            self.functions[domain, name] = fct
        return fct

    def add_docstring(self, fn, docstring):
        fn.append_docstring(docstring)

    def add_stmt(self, fn, results, module, opname, args, attrs, sub_functions=None):
        s = Stmt(results, module, opname, args, attrs, sub_functions=sub_functions)
        fn.append_stmt(s)

    def add_input(self, fn, varname, type, info):
        v = Var(varname, type, info)
        fn.append_input(v)

    def add_attr(self, fn, varname, type, info, default_value=None):
        if default_value is not None:
            a = Attr(helper.make_attribute(varname, default_value))
            fn.append_attr_proto(a)
        else:
            v = Var(varname, type, info)
            fn.append_attr(v)

    def add_output(self, fn, varname, type, info):
        v = Var(varname, type, info)
        fn.append_output(v)

    def attr(self, attrname, attrval):
        if (isinstance(attrval, Function)):
            attrval = str(attrval)  # TODO
        return Attr(helper.make_attribute(attrname, attrval))

    def attr_ref(self, attrname, refname, pytype):
        a = onnx.AttributeProto()
        a.name = attrname
        a.ref_attr_name = refname
        a.type = ta.pytype_to_attrtype_map[pytype]  # onnx.AttributeProto.FLOAT
        return Attr(a)
        # TODO: attr_type?
