# SPDX-License-Identifier: Apache-2.0

import logging
from io import StringIO
import onnx
import onnx.helper as helper
from . import type_annotation as ta

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
    def __init__(self, varname, type=None) -> None:
        self.name = varname
        self.type = type

    def __str__(self):
        return self.name

    def typed_str(self):
        return self.name + " : " + str(self.type)

    def to_value_info(self):
        tp = self.type.to_type_proto()
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
        self.result = result
        self.module = module
        self.opname = opname
        self.args = args
        self.attrs = attrs

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
        n = helper.make_node(self.opname,
                             [str(x) for x in self.args],
                             [str(x) for x in self.result])
        for a in self.attrs:
            n.attribute.append(a.attr_proto)
        return n


class Function:
    def __init__(self, name) -> None:
        self.name = name
        self.inputs = []
        self.outputs = []
        self.stmts = []
        self.attrs = []

    def __str__(self):
        attrs = format(self.attrs, "<", ", ", ">") if self.attrs else ""
        inputs = format([x.typed_str() for x in self.inputs], "(", ", ", ")")
        outputs = format([x.typed_str() for x in self.outputs], "(", ", ", ")")
        stmts = format(self.stmts, "\n{\n   ", "\n   ", "\n}\n")
        return (self.name + " " + attrs + inputs + " => " + outputs + stmts)

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            st = StringIO()
            for s in self.stmts:
                for attr in s.attrs:
                    if attr.attr_proto.HasField("g"):
                        st.write(helper.printable_graph(attr.attr_proto.g))
                        st.write("\n")
            logger.debug("%s: %s", type(self), st.getvalue())

    def to_graph_proto(self):
        return helper.make_graph([s.to_node_proto() for s in self.stmts],
                                 self.name,
                                 [x.to_value_info() for x in self.inputs],
                                 [y.to_value_info() for y in self.outputs])

# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:
    def new_function(self, name):
        return Function(name)

    def add_stmt(self, fn, results, module, opname, args, attrs):
        s = Stmt(results, module, opname, args, attrs)
        fn.stmts.append(s)

    def add_input(self, fn, varname, type):
        v = Var(varname, type)
        fn.inputs.append(v)

    def add_attr(self, fn, varname, type):
        v = Var(varname, type)
        fn.attrs.append(v)

    def add_output(self, fn, varname, type):
        v = Var(varname, type)
        fn.outputs.append(v)

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
