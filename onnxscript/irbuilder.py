# SPDX-License-Identifier: Apache-2.0

import logging
from io import StringIO
import onnx
import onnx.helper as helper
from . import type_annotation as ta

# A simple IR (Function, Stmt, Attr, Var):

logger = logging.getLogger("onnx-script")


def format(args, prefix, sep, suffix):
    return prefix + sep.join([str(x) for x in args]) + suffix


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


class Result:
    """
    A Result refers to an onnx object, whether it is an input,
    an output, an intermediate result.
    """
    def __init__(self, name, typeinfo=None, value=None):
        if not isinstance(name, str):
            raise TypeError("name must be a string not %r." % type(name))
        self.name = name
        self.typeinfo = typeinfo
        self.value = value

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.is_py_const():
            return 'C(%r)' % self.name
        return 'R(%r)' % self.name

    def is_py_const(self):
        if self.value is None:
            return False
        return isinstance(self.value, (int, float))


class Var:
    def __init__(self, result, typeinfo=None) -> None:
        if not isinstance(result, Result):
            raise TypeError("Unexpected type %r." % type(result))
        self.name = result
        self.typeinfo = typeinfo

    def __str__(self):
        return str(self.name)

    def typed_str(self):
        return str(self.name) + " : " + str(self.typeinfo)

    def to_value_info(self):
        tp = self.typeinfo.to_type_proto()
        # if (not tp.tensor_type.HasField('shape')):
        #     # TODO: temporary patch to export a function as a graph
        #     tp = helper.make_tensor_type_proto(tp.tensor_type.elem_type, [10])
        return helper.make_value_info(str(self.name), tp)


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
        if any(map(lambda r: not isinstance(r, Result), args)):
            logger.error('Stmt:args:%r%r', args, list(map(type, args)))
            raise TypeError("args must be a list of Result.")
        if any(map(lambda r: not isinstance(r, Result), result)):
            logger.error('Stmt:result:%r:%r', result, list(map(type, result)))
            raise TypeError("result must be a list of Result.")
        self.result = result
        self.module = module
        self.opname = opname
        self.args = args
        self.attrs = attrs

    def __str__(self):
        if (isinstance(self.result, str)):
            logger.debug("unexpected str type for self.result where type(self)=%r",
                         type(self))
        lhs = ", ".join(map(str, self.result))
        attrs = ""
        if (self.attrs):
            attrs = format(self.attrs, "<", ", ", ">")

        args = format(self.args, "(", ", ", ")")
        module = str(self.module)
        callee = module + "." + self.opname if (module != '') else self.opname
        return lhs + " = " + callee + " " + attrs + args

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Stmt: %s: %s", type(self), str(self))

    def to_node_proto(self):
        # check one input is a constant
        args = self.args
        cast_like = None
        if self.opname in {'Add', 'Sub', 'Div', 'Mul', 'Mod'}:
            if len(args) == 2 and args[0].is_py_const() != args[1].is_py_const():
                index = 0 if self.args[0].is_py_const() else 1
                new_name = '%s_CASTLIKE' % args[0]  # choose a unique name
                logger.debug("Stmt.to_node_proto:CastLike(%s, %s) -> %s",
                             args[index], args[1 - index], new_name)
                cast_like = helper.make_node('CastLike',
                                             [str(args[index]), str(args[1 - index])],
                                             [new_name])
                args = list(args)
                args[index] = new_name

        n = helper.make_node(self.opname,
                             [str(x) for x in args],
                             [str(x) for x in self.result])
        for a in self.attrs:
            n.attribute.append(a.attr_proto)
        if cast_like is None:
            return [n]
        return [cast_like, n]


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

    def append_stmt(self, stmt):
        logger.debug('F(%s):stmt:%s', self.name, stmt)
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
            logger.debug("Function %r: %s: %s", self.name, type(self), st.getvalue())

    def to_graph_proto(self):
        nodes = []
        for s in self.stmts:
            nodes.extend(s.to_node_proto())
        return helper.make_graph(nodes, self.name,
                                 [x.to_value_info() for x in self.inputs],
                                 [y.to_value_info() for y in self.outputs])

    def __repr__(self):
        return 'Function(%r)' % self.name

# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:
    def new_function(self, name):
        return Function(name)

    def add_stmt(self, fn, results, module, opname, args, attrs):
        s = Stmt(results, module, opname, args, attrs)
        fn.append_stmt(s)

    def add_input(self, fn, varname, typeinfo):
        v = Var(varname, typeinfo)
        fn.append_input(v)

    def add_attr(self, fn, varname, typeinfo):
        v = Var(varname, typeinfo)
        fn.append_attr(v)

    def add_output(self, fn, varname, typeinfo):
        v = Var(varname, typeinfo)
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
