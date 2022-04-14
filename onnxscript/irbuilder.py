# SPDX-License-Identifier: Apache-2.0

import logging
from io import StringIO
import onnx
import onnx.helper as helper
from . import type_annotation as ta
from .values import Opset

# A simple IR (Function, Stmt, Attr, Var):

logger = logging.getLogger("onnx-script")


def format(list, prefix, sep, suffix):
    return prefix + sep.join([str(x) for x in list]) + suffix


class Type:
    def __init__(self) -> None:
        # TODO
        tp = onnx.TypeProto()
        tp.tensor_type.elem_type = onnx.TensorProto.FLOAT
        self.onnx_type = tp
        # helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [10])

    def to_type_proto(self):
        return self.onnx_type

    def __str__(self) -> str:
        return "SomeType"


class Var:
    def __init__(self, varname, typeinfo=None) -> None:
        self.name = varname
        self.typeinfo = typeinfo

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.typeinfo)

    def typed_str(self):
        return self.name + " : " + str(self.typeinfo)

    def to_value_info(self):
        tp = self.typeinfo.to_type_proto()
        # if (not tp.tensor_type.HasField('shape')):
        #     # TODO: temporary patch to export a function as a graph
        #     tp = helper.make_tensor_type_proto(tp.tensor_type.elem_type, [10])
        return helper.make_value_info(self.name, tp)


class Attr:
    def __init__(self, attrproto) -> None:
        self.attr_proto = attrproto

    def __str__(self):
        if (self.attr_proto.HasField("ref_attr_name")):
            return self.attr_proto.name + " = @" + self.attr_proto.ref_attr_name
        # self.name + " = " + self.value
        return helper.printable_attribute(self.attr_proto)


class Stmt:
    def __init__(self, result, module, opname, args, attrs) -> None:
        if not isinstance(module, Opset):
            raise TypeError(f"Unexpected type {type(module)} for module.")
        self.result = result
        self.module = module
        self.opname = opname
        self.args = args
        self.attrs = attrs
        self.domain = self.module.domain

    def __str__(self):
        if (isinstance(self.result, str)):
            logger.debug("unexpected str type for self.result where type(self)=%r",
                         type(self))
        lhs = ", ".join(self.result)
        attrs = ""
        if (self.attrs):
            attrs = format(self.attrs, "<", ", ", ">")

        args = format(self.args, "(", ", ", ")")
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
                             [str(x) for x in self.args],
                             [str(x) for x in self.result],
                             domain=self.domain)
        for a in self.attrs:
            n.attribute.append(a.attr_proto)
        return n


class Function:
    def __init__(self, name, domain="") -> None:
        if domain == "":
            raise ValueError("Domain cannot be null.")
        self.domain = domain
        self.name = name
        self.inputs = []
        self.outputs = []
        self.stmts = []
        self.attrs = []
        self.functions = {}
        self.docstring = ""

    def __str__(self):
        attrs = format(self.attrs, "<", ", ", ">") if self.attrs else ""
        inputs = format([x.typed_str() for x in self.inputs], "(", ", ", ")")
        outputs = format([x.typed_str() for x in self.outputs], "(", ", ", ")")
        stmts = format(self.stmts, "\n{\n   ", "\n   ", "\n}\n")
        return (self.name + " " + attrs + inputs + " => " + outputs + stmts)

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

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            st = StringIO()
            for s in self.stmts:
                for attr in s.attrs:
                    if attr.attr_proto.HasField("g"):
                        st.write(helper.printable_graph(attr.attr_proto.g))
                        st.write("\n")

    def append_function(self, opf):
        name = opf.name
        if name in self.functions:
            # Already added.
            return
        proto = opf.to_function_proto()
        self.functions[name] = proto

    def model_functions(self):
        """
        The ONNX implementation may rely on additional functions
        stored in `self.functions`. This method returns it.
        """
        return self.functions

    @staticmethod
    def retrieve_all_opsets(graph):
        domains = set()
        for node in graph.node:
            domains.add(node.domain)
            if len(node.attribute) == 0:
                continue
            for at in node.attribute:
                if at.name not in {'then_branch', 'else_branch', 'body'}:
                    continue
                sub_domains = Function.retrieve_all_opsets(at.g)
                domains |= sub_domains
        return domains

    def to_model_proto(self, opsets=None, functions=None, **kwargs):
        if opsets is None:
            opsets = {'': 15}
        elif isinstance(opsets, int):
            opsets = {'': opsets}
        else:
            opsets = opsets.copy()
        for n in self.stmts:
            if n.module.domain not in opsets:
                opsets[n.module.domain] = n.module.version
        graph = self.to_graph_proto()

        all_opsets = Function.retrieve_all_opsets(graph)
        for op in all_opsets:
            if op not in opsets:
                opsets[op] = 1

        opset_imports = [onnx.helper.make_opsetid(domain, version)
                         for domain, version in opsets.items()]
        functions = [] if functions is None else list(functions)
        # TODO: the following is incomplete. we need to do this iteratively.
        functions.extend(self.functions.values())
        return helper.make_model(graph, opset_imports=opset_imports,
                                 functions=functions, **kwargs)

    def to_graph_proto(self):
        return helper.make_graph([s.to_node_proto() for s in self.stmts],
                                 self.name,
                                 [x.to_value_info() for x in self.inputs],
                                 [y.to_value_info() for y in self.outputs])

    def to_function_proto_with_opset_imports(self, domain="", func_opset_imports=[]):
        # TODO: Ideally, in the long term, we should infer func_opset_imports
        # from the set of calls within the function itself.
        return helper.make_function(domain,
                                    self.name,
                                    inputs=[x.name for x in self.inputs],
                                    outputs=[y.name for y in self.outputs],
                                    nodes=[s.to_node_proto() for s in self.stmts],
                                    opset_imports=func_opset_imports,
                                    attributes=[a.name for a in self.attrs],
                                    doc_string=self.docstring)

    def to_function_proto(self, domain):
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
        return helper.make_function(
            self.domain,
            self.name,
            inputs=[x.name for x in self.inputs],
            outputs=[y.name for y in self.outputs],
            nodes=nodes,
            opset_imports=opset_imports,  # TODO
            attributes=[a.name for a in self.attrs],
            doc_string=self.docstring
        )

# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:
    def new_function(self, name, domain=""):
        return Function(name, domain)

    def add_docstring(self, fn, docstring):
        fn.append_docstring(docstring)

    def add_stmt(self, fn, results, module, opname, args, attrs):
        s = Stmt(results, module, opname, args, attrs)
        fn.append_stmt(s)

    def add_input(self, fn, varname, type):
        v = Var(varname, type)
        fn.append_input(v)

    def add_attr(self, fn, varname, type):
        v = Var(varname, type)
        fn.append_attr(v)

    def add_output(self, fn, varname, type):
        v = Var(varname, type)
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
