# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import ast
import logging
import sys
from enum import IntEnum
from typing import Any, Dict, List, NoReturn, Optional, Union

import numpy
import onnx
from onnx import helper, numpy_helper

import onnxscript
from onnxscript import analysis, autocast, irbuilder, onnx_types, sourceinfo
from onnxscript import type_annotation as ta
from onnxscript import values

use_subscript = sys.version_info[:2] >= (3, 9)
if use_subscript:
    _ast_Subscript = ast.Subscript  # noqa: N816
else:
    _ast_Subscript = (ast.Subscript, ast.Index)  # type: ignore[misc,assignment]  # noqa: N816

logger = logging.getLogger("onnx-script")


# Python-to-IR converter:


def not_allowed(construct):
    return f"{construct}not supported."


class TranslationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def warn(msg):
    logger.warning(msg)


def fail(msg) -> NoReturn:
    raise TranslationError(msg)


def fail_if(cond, msg):
    if cond:
        raise TranslationError(msg)


def ignore(cond, msg):
    if cond:
        warn(msg)


# Utility to convert a python value to TensorProto:
def py_type_to_onnx_type(pytype: type, info: sourceinfo.SourceInfo):
    if pytype is bool:
        return onnx.TensorProto.BOOL
    if pytype is int:
        return onnx.TensorProto.INT64
    if pytype is float:
        return onnx.TensorProto.FLOAT
    if pytype is str:
        return onnx.TensorProto.STRING
    fail(info.msg(f"Tensor conversion of element of type {pytype} is not implemented"))


def pyvalue_to_tensor(
    tensor_name: str, pyvalue, converter, info: sourceinfo.SourceInfo
):  # pylint: disable=unused-argument
    if isinstance(pyvalue, numpy.ndarray):
        return numpy_helper.from_array(pyvalue, tensor_name)
    if isinstance(pyvalue, list):
        if len(pyvalue) == 0:
            fail(info.msg("Cannot convert an empty list to tensor"))
        pytype = type(pyvalue[0])
        if not all(isinstance(e, pytype) for e in pyvalue):
            fail(info.msg("Cannot convert an list with elements of different types to tensor"))
        return helper.make_tensor(
            tensor_name,
            py_type_to_onnx_type(pytype, info),
            [len(pyvalue)],
            pyvalue,
        )
    onnx_type = py_type_to_onnx_type(type(pyvalue), info)
    if onnx_type is onnx.TensorProto.BOOL:
        return helper.make_tensor(tensor_name, onnx_type, [], [int(pyvalue)])
    if onnx_type is onnx.TensorProto.STRING:
        return helper.make_tensor(tensor_name, onnx_type, [], vals=[pyvalue.encode("utf-8")])

    return helper.make_tensor(tensor_name, onnx_type, [], [pyvalue])


# map from python operators to ONNX ops
primop_map = {
    ast.Add: "Add",
    ast.And: "And",
    ast.BitAnd: "And",
    ast.BitOr: "Or",
    ast.Div: "Div",
    ast.Eq: "Equal",
    ast.Gt: "Greater",
    ast.GtE: "GreaterOrEqual",
    ast.Lt: "Less",
    ast.LtE: "LessOrEqual",
    ast.MatMult: "MatMul",
    ast.Mod: "Mod",
    ast.Mult: "Mul",
    ast.Not: "Not",
    ast.NotEq: "NotEqual",
    ast.Or: "Or",
    ast.Pow: "Pow",
    ast.Sub: "Sub",
    ast.USub: "Neg",
}


class ConverterExpressionKind(IntEnum):
    ANY = 0
    CONST = 1


class ConverterExpression:
    def __init__(self, name: Optional[Union[str, List[str]]], kind: ConverterExpressionKind):
        self.name = name
        self.kind = kind

    def is_const(self):
        return self.kind == ConverterExpressionKind.CONST

    def __str__(self):
        return self.name


class Converter:
    """Main class to translate python code into ONNX operators.

    Args:
        ir_builder: convert AST node into ONNX structures, if None,
            class :class:`onnxscript.irbuilder.IRBuilder` is used

    The class uses logger `onnx-script`. Logging can be enabled with the following code:

    ::

        import logging
        logging.basicConfig(level=logging.DEBUG)

    Or if you need to enable only the logger used by this module:

    ::

        import logging
        logger = logging.getLogger('onnx-script')
        logger.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        logger.addHandler(console)
    """

    def __init__(
        self,
        ir_builder=None,
        opset=None,
        global_names=None,
        source=None,
        default_opset=None,
    ):
        self.ir_builder = ir_builder or irbuilder.IRBuilder()
        self.source = source
        if global_names is not None:
            # We make a copy in case function eval modifies it.
            self.globals = global_names.copy()
        self.this_module = opset
        self.default_opset_ = default_opset

    @property
    def default_opset(self):
        if self.default_opset_ is None:
            raise RuntimeError(
                "default_opset must be specified in script for functions "
                "that do not contain any use of an ONNX opset."
            )
        return self.default_opset_

    def set_default_opset(self, opset, node):
        if opset.domain != "":
            return
        if self.default_opset_ is not None:
            if (
                opset.domain != self.default_opset_.domain
                or opset.version != self.default_opset_.version
            ):
                self.fail(
                    node, f"Two distincts opset were used ({opset} != {self.default_opset_})."
                )
        else:
            self.default_opset_ = opset

    def find_onnx_opset(self, node: ast.AST) -> Optional[values.Opset]:
        """Find the (first) ONNX opset used in the function, if any."""
        # Search for a Call expression of form "op.OpName(...)"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                opset_expr = node.func.value
                if isinstance(opset_expr, ast.Name):
                    if opset_expr.id in self.globals:
                        opset = self.globals[opset_expr.id]
                        if isinstance(opset, values.Opset) and opset.domain == "":
                            return opset
        for child in ast.iter_child_nodes(node):
            res = self.find_onnx_opset(child)
            if res is not None:
                return res
        return None

    def init_function_translation(self):
        """Initialize self for translating a new (top-level) function."""
        self.outer = []
        self.current_fn = None
        self.nextvar = 0
        self.used_vars = set()
        self.locals: List[Dict[Any, Any]] = [{}]

    def source_of(self, node: ast.AST) -> sourceinfo.SourceInfo:
        return sourceinfo.SourceInfo(node, self.source, self.current_fn.name)

    def message(self, node: ast.AST, error_msg: str) -> str:
        """Constructs an error message containing source information about an ast node."""
        return self.source_of(node).msg(error_msg)

    def warn(self, node: ast.AST, error_msg: str) -> None:
        warn(self.message(node, error_msg))

    def fail(self, node: ast.AST, error_msg: str) -> NoReturn:
        fail(self.message(node, error_msg))

    # Name resolution and namescopes: This component handles the following aspects:
    # * Name-scopes are different in Python and the generated ONNX:
    #   - Control-flow blocks (a loop body or the then-or-else block of an if-stmt)
    #     form part of the same name-scope in Python, but will be mapped to a nested
    #     name-scope (as a sub-graph) in ONNX.
    # * Script-time name-value tracking: Name lookup during script-time returns
    #   statically-known information about the value the name will have at runtime.
    def enter_scope(self, name, parent_node):
        """Enter a control-flow block (a loop body or if-then-else branch).
        The block is translated into a nested-scope in ONNX.
        """
        self.outer.insert(0, self.current_fn)
        self.current_fn = self.ir_builder.new_function(name)
        self.locals.insert(0, {})
        logger.debug("Converter:enter_scope:%d:node:%s", len(self.locals), type(parent_node))

    def exit_scope(self):
        """Exit from a control-flow block (a loop body or if-then-else branch)."""
        logger.debug("Converter:exit_scope:%d", len(self.locals))
        graph = self.current_fn
        self.current_fn = self.outer.pop(0)
        self.locals.pop(0)
        return graph

    def current_scope(self):
        return self.locals[0]

    def bind(self, name, val):
        logger.debug("Converter:bind:%s", name)
        self.locals[0][name] = val

    def lookup(self, name, info, raise_exception=True):
        for scope in self.locals:
            if name in scope:
                return scope[name]
        if name in self.globals:
            return self.globals[name]
        if raise_exception:
            raise ValueError(info.msg(f"Unbound name: {name}."))
        return None

    def generate_unique_name(self, candidate: str = "tmp") -> str:
        # TODO(justinchuby): Can we reduce the O complexity of this function?
        r = candidate
        while r in self.used_vars:
            r = f"{candidate}_{self.nextvar}"
            self.nextvar = self.nextvar + 1
        self.used_vars.add(r)
        return r

    def to_onnx_attr_ref(self, val: values.AttrRef, info: Optional[sourceinfo.SourceInfo]):
        pytype = val.typeinfo
        attrtype = ta.pytype_to_attrtype(pytype)
        attrname = None
        if attrtype is onnx.AttributeProto.FLOAT:
            attrname = "value_float"
        elif attrtype is onnx.AttributeProto.INT:
            attrname = "value_int"
        elif attrtype is onnx.AttributeProto.STRING:
            attrname = "value_string"
        elif attrtype is onnx.AttributeProto.INTS:
            attrname = "value_ints"
        else:
            msg = f"Unsupported attribute type {pytype!r}."
            fail(info.msg(msg) if info else msg)
        return self.ir_builder.make_attr_ref(attrname, val.value, pytype)

    def to_onnx_var(self, val, target=None, info: Optional[sourceinfo.SourceInfo] = None):
        if isinstance(val, values.AttrRef):
            # promote attribute to value
            result = self.generate_unique_name(target or "tmp")
            attr = self.to_onnx_attr_ref(val, info)
            self.emit([result], values.Op(self.default_opset, "Constant"), [], [attr])
            return ConverterExpression(result, ConverterExpressionKind.CONST)
        if isinstance(val, values.Dynamic):
            return val.value
        # Assume value is a python-value convertible to a tensor
        # TODO: check if value is convertible to a TensorProto, so that we can
        # produce a better error message otherwise
        return self.emit_const(val, target or "tmp", info)

    def py_var_to_onnx_var(self, py_var, info: sourceinfo.SourceInfo):
        return self.to_onnx_var(self.lookup(py_var, info), target=py_var, info=info)

    def emit_docstring(self, docstring):
        self.ir_builder.add_docstring(self.current_fn, docstring)

    def emit(self, outputs, callee, inputs, attrs, sub_functions=None):
        self.ir_builder.add_stmt(
            self.current_fn,
            outputs,
            callee,
            inputs,
            attrs,
            sub_functions,
        )

    def emit_loop(self, outputs, callee, inputs, attrs, info, sub_functions=None):
        def rename(x):
            r = self.generate_unique_name(x)
            self.bind(x, values.Dynamic(r, values.DynamicKind.Output, info))
            return r

        onnx_inputs = inputs
        onnx_outputs = [rename(x) for x in outputs]
        self.emit(
            onnx_outputs,
            values.Op(self.default_opset, callee),
            onnx_inputs,
            attrs,
            sub_functions=sub_functions,
        )

    def emit_const(self, pyvalue, suggested_name, info):
        ovar = self.generate_unique_name(suggested_name)
        tensor = pyvalue_to_tensor(ovar, pyvalue, self, info)
        attr = self.ir_builder.make_attr("value", tensor)
        self.emit([ovar], values.Op(self.default_opset, "Constant"), [], [attr])
        return ConverterExpression(ovar, ConverterExpressionKind.CONST)

    def emit_copy(self, original_var: str, suggested_name: str) -> str:
        """Emits a copy statement, using the ONNX Identity operator."""
        new_var = self.generate_unique_name(suggested_name)
        self.emit([new_var], values.Op(self.default_opset, "Identity"), [original_var], [])
        return new_var

    def is_constant_expr(self, node):
        if isinstance(node, ast.UnaryOp):
            if self.is_constant_expr(node.operand):
                return True
            return False
        if isinstance(
            node,
            (
                ast.Call,
                ast.BinOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Num,
                ast.Str,
                ast.Attribute,
                ast.List,
                ast.Load,
                ast.NameConstant,
                ast.Constant,
                ast.Str,
            ),
        ):
            return all(self.is_constant_expr(c) for c in ast.iter_child_nodes(node))
        return False

    def eval_constant_expr(self, expr):
        """Evaluates a sub-expression that is assumed to represent a constant value.
        The expression can refer only to global names (inherited from the scope
        where the script is evaluated) and cannot refer to local names defined
        within the script.) Further, these expressions are assumed to be constants.
        Thus, any subsequent mutation of any state/variables (used in computing
        this constant value) will potentially lead to unexpected behavior (such
        as divergence between eager-mode execution and evaluation of the ONNX
        function.)
        """
        # TODO: assert (self.is_constant_expr(expr))
        # TODO: Refine types
        locals: dict[Any, Any] = {}
        expr = ast.Expression(expr)
        cpl = compile(expr, filename="<ast>", mode="eval")
        try:
            return eval(cpl, self.globals, locals)
        except NameError as e:
            raise NameError(
                self.message(
                    expr,
                    f"Missing names, globals contains {list(self.globals)!r}, "
                    f"locals {list(locals)!r}.",
                )
            ) from e

    def translate_attr(self, attr_name, expr):
        """Translate an attribute-value specification of the form `attr_name=<expr>`
        in a call to an op. expr is an AST. The following cases are supported:
        * Expr evaluates to a script-time constant (a python-value) that can be mapped
        into an ONNX attribute value, or
        * Expr must be an attribute-reference, that is a name representing an
        attribute-parameter of a containing function.
        """
        if isinstance(expr, ast.Name):
            val = self.lookup(expr.id, self.source_of(expr))
            if isinstance(val, values.AttrRef):
                return self.ir_builder.make_attr_ref(attr_name, val.value, val.typeinfo)
            if isinstance(val, irbuilder.IRFunction):
                # Check that outer-scope variables referenced by function have same value
                # at function-definition site and use-as-attribute site, to avoid errors.
                for pyvar, previous in val.outer_scope_variables:
                    current = self.lookup(pyvar, self.source_of(expr))
                    if current.value != previous.value:
                        self.fail(
                            expr,
                            f"Outer scope variable {pyvar} referenced by function "
                            f"{expr.id!r} modified.",
                        )

                # Create GraphProto attribute
                val = val.to_graph_proto()
            return self.ir_builder.make_attr(attr_name, val)
        return self.ir_builder.make_attr(attr_name, self.eval_constant_expr(expr))

    def translate_docstring(self, node):
        if hasattr(node.value, "value"):
            # python 3.8+
            return self.emit_docstring(node.value.value)
        if hasattr(node.value, "s"):
            # python 3.7
            return self.emit_docstring(node.value.s)
        raise TypeError(
            f"Unexpected type {type(node)!r} for node. Unsupoorted version of python."
        )

    def translate_expr(self, node, target="tmp") -> ConverterExpression:
        """Expression-translation generates "IR statements/nodes" that compute the value of
        the expression into a target-variable, and returns the variable that is
        assigned this value.
        """
        if isinstance(node, ast.Call):
            r = self.translate_call_expr(node)
        elif isinstance(node, (ast.BinOp, ast.BitAnd, ast.BitOr)):
            r = self.translate_bin_op_expr(node)
        elif isinstance(node, ast.BoolOp):
            r = self.translate_bool_op_expr(node)
        elif isinstance(node, ast.UnaryOp):
            r = self.translate_unary_op_expr(node)
        elif isinstance(node, ast.Compare):
            r = self.translate_compare_expr(node)
        elif isinstance(node, ast.Name):
            r = self.translate_name_expr(node)
        elif isinstance(node, _ast_Subscript):
            r = self.translate_subscript_expr(node)
        elif self.is_constant_expr(node):
            r = self.emit_const(self.eval_constant_expr(node), target, self.source_of(node))
        else:
            raise ValueError(
                self.message(node, f"Unsupported expression type {type(node)!r}.")
            )
        if isinstance(r, ConverterExpression):
            return r
        if isinstance(r, tuple):
            callee, args, attrs = r
            if isinstance(target, str):
                result = self.generate_unique_name(target)
                self.emit([result], callee, args, attrs)
                return ConverterExpression(result, ConverterExpressionKind.ANY)
            results = [self.generate_unique_name(x) for x in target]
            self.emit(results, callee, args, attrs)
            return ConverterExpression(results, ConverterExpressionKind.ANY)
        return ConverterExpression(r, ConverterExpressionKind.ANY)

    def translate_opt_expr(self, node, target="tmp"):
        """Translation of an expression where "None" is permitted.

        (eg., for an optional argument)
        None is represented as a NameConstant in Python 3.7 and Constant in Python 3.9.
        """
        if isinstance(node, (ast.NameConstant, ast.Constant)) and (node.value is None):
            return ConverterExpression(None, ConverterExpressionKind.ANY)
        return self.translate_expr(node, target)

    def translate_subscript_expr(self, node):
        """List of supported syntaxes is below.
        `A` is a tensor or an expression equivalent to a tensor.

        ::

            A[:, 1]
            A[:2, 0]
            A[:2, :1]
            A[2:0:-1]
            A[1:]
            A[:2]
            A[1:-1]
            A[1:2]
            A[-1]
            A[0]
            A[:0:-1]

        *i* is a tensor holding one integer.

        ::

            A[i]
            A[i+1:i+2]

        Fully supported for python 3.9+.

        ::

            A[i:i+j, k]

        Not supported:

        ::

            A[::-1]
        """
        var = self.translate_expr(node.value)
        var_name = var.name

        info = self.source_of(node.slice if use_subscript else node)

        def _get_arg(node_arg, axis, zero, one, default_value=None):
            if node_arg is None:
                if default_value is None:
                    return "", None
                # The default value for the extremities depends on the step.
                # This one is usually positive unless it is a constant,
                # otherwise it is unknown.
                if default_value == "begin":
                    return zero.name, None
                if default_value == "begin_":
                    self.fail(
                        node,
                        "`?::-1` cannot be expressed with ONNX, `?:0:-1` misses "
                        "the first line, `:-1:-1` returns an empty tensor.",
                    )
                if default_value == "end":
                    shape_name = self.generate_unique_name(f"{var_name}_shape")
                    self.emit(
                        [shape_name],
                        values.Op(self.default_opset, "Shape"),
                        [var_name],
                        [],
                    )
                    dim_name = self.generate_unique_name(f"{shape_name}_dim")
                    self.emit(
                        [dim_name],
                        values.Op(self.default_opset, "Gather"),
                        [shape_name, axis.name],
                        [],
                    )
                    return dim_name, None
                raise RuntimeError(f"Unexpected default value {default_value!r}.")

            name = self.translate_expr(node_arg).name
            reshaped = self.generate_unique_name(f"{name}_reshaped")
            self.emit(
                [reshaped],
                values.Op(self.default_opset, "Reshape"),
                [name, one.name],
                [],
            )
            if self.is_constant_expr(node_arg):
                cst = self.eval_constant_expr(node_arg)
            else:
                cst = None
            return reshaped, cst

        def _get_slice_input(node_slice, axis, zero, one):
            step_name, cst = _get_arg(node_slice.step, axis, zero, one)
            if cst is not None and cst < 0:
                # handling [::-1]
                def_a, def_b = "end", "begin_"
            else:
                def_a, def_b = "begin", "end"
            lower_name, _ = _get_arg(node_slice.lower, axis, zero, one, default_value=def_a)
            upper_name, _ = _get_arg(node_slice.upper, axis, zero, one, default_value=def_b)
            inputs = [var_name, lower_name, upper_name, axis.name]
            if step_name != "":
                inputs.append(step_name)
            return inputs

        if use_subscript:
            node_slice = node.slice
        else:
            node_slice = getattr(node.slice, "value", None)

        if self.is_constant_expr(node_slice):
            # A[i], i is an integer
            index = self.eval_constant_expr(node_slice)
            var_index = self.emit_const([index], "subscript_index", info)
            tmp = self.generate_unique_name(f"{var_name}_gather")
            self.emit(
                [tmp],
                values.Op(self.default_opset, "Gather"),
                [var_name, var_index.name],
                [],
            )
            axis = self.emit_const([0], "subscript_axis", info)
            inputs = [tmp, axis.name]
            return values.Op(self.default_opset, "Squeeze"), inputs, []

        if isinstance(node.slice, ast.Slice):
            # A[a:b], a, b are expressions equivalent to integers
            one = self.emit_const([1], "one", info)
            axis = self.emit_const([0], "subscript_axis", info)
            inputs = _get_slice_input(node.slice, axis, axis, one)
            return values.Op(self.default_opset, "Slice"), inputs, []

        if isinstance(node.slice, ast.Tuple) or (
            not use_subscript and isinstance(node.slice, ast.ExtSlice)
        ):
            # A[a:b, c:d, e], a, b, c, d, e are expressions equivalent to integers
            # tuple can be any length
            if isinstance(node.slice, ast.Tuple):
                elts = node.slice.elts
            else:
                elts = node.slice.dims
            one = self.emit_const([1], "one", info)
            zero = None
            starts = []
            ends = []
            axes = []
            steps = []
            squeezed_axes = []
            for axis, elt in enumerate(elts):
                if self.is_constant_expr(elt) or (
                    not use_subscript and isinstance(elt, ast.Index)
                ):
                    # if the tuple contains a constant, it is replaced
                    # by a slice and processed like any other slice
                    element = None
                    if use_subscript:
                        index = self.eval_constant_expr(elt)
                    else:
                        try:
                            index = self.eval_constant_expr(elt.value)
                        except NameError:
                            element = elt
                    if element is None:
                        squeezed_axes.append(axis)
                        kwargs = dict(
                            lineno=getattr(elt, "lineno", node.lineno),
                            col_offset=getattr(elt, "col_offset", node.col_offset),
                        )
                        element = ast.Slice(
                            ast.Constant(index, **kwargs),
                            ast.Constant(index + 1, **kwargs),
                            ast.Constant(1, **kwargs),
                        )
                else:
                    element = elt

                var_axis = self.emit_const([axis], f"ax{axis}", info)
                if axis == 0:
                    zero = var_axis

                if isinstance(element, ast.Slice):
                    # process slice index
                    inputs = _get_slice_input(element, var_axis, zero, one)
                    starts.append(inputs[1])
                    ends.append(inputs[2])
                    axes.append(var_axis.name)
                    steps.append(inputs[4] if len(inputs) > 4 else one.name)
                    continue

                # not a constant, not a slice -> an expression
                squeezed_axes.append(axis)
                index = self.translate_expr(element).name
                starts.append(index)
                index_1 = self.generate_unique_name(f"{var_name}_end")
                self.emit([index_1], values.Op(self.default_opset, "Add"), [index, one], [])
                ends.append(index_1)
                axes.append(var_axis.name)
                steps.append(one.name)

            attr = self.ir_builder.make_attr("axis", 0)
            start_name = self.generate_unique_name(f"{var_name}_start")
            self.emit([start_name], values.Op(self.default_opset, "Concat"), starts, [attr])

            end_name = self.generate_unique_name(f"{var_name}_end")
            self.emit([end_name], values.Op(self.default_opset, "Concat"), ends, [attr])

            axes_name = self.generate_unique_name(f"{var_name}_axis")
            self.emit([axes_name], values.Op(self.default_opset, "Concat"), axes, [attr])

            steps_name = self.generate_unique_name(f"{var_name}_step")
            self.emit([steps_name], values.Op(self.default_opset, "Concat"), steps, [attr])
            if squeezed_axes:
                sliced_name = self.generate_unique_name(f"{var_name}sliced")
                self.emit(
                    [sliced_name],
                    values.Op(self.default_opset, "Slice"),
                    [var_name, start_name, end_name, axes_name, steps_name],
                    [],
                )
                squeezed_axis = self.emit_const(squeezed_axes, f"squeezed_ax{axis}", info)
                return (
                    values.Op(self.default_opset, "Squeeze"),
                    [sliced_name, squeezed_axis],
                    [],
                )
            return (
                values.Op(self.default_opset, "Slice"),
                [var_name, start_name, end_name, axes_name, steps_name],
                [],
            )

        # A[i], i is an expression equivalent to an integer
        var_index = self.translate_expr(node_slice)
        tmp = self.generate_unique_name(f"{var_name}_gather")
        self.emit(
            [tmp],
            values.Op(self.default_opset, "Gather"),
            [var_name, var_index.name],
            [],
        )
        axis = self.emit_const([0], "subscript_axis", info)
        return values.Op(self.default_opset, "Squeeze"), [tmp, axis.name], []

    def translate_call_expr(self, node):
        """Translates a call-expression.
        For now, the handling of positional and named arguments is slightly different
        from standard Python. We implicitly map named arguments to ONNX attributes, and
        positional arguments to ONNX inputs.
        """
        callee = self.translate_callee_expr(node.func)
        args = [self.translate_opt_expr(x) for x in node.args]
        args = autocast.static_cast_inputs(self, callee.get_schema(), *args)
        attrs = [self.translate_attr(x.arg, x.value) for x in node.keywords]
        return callee, args, attrs

    def _cast_like_binary_expression(self, op, left, right):
        schema = op.get_schema()
        return autocast.static_cast_inputs(self, schema, left, right)

    def translate_bool_op_expr(self, node: ast.BoolOp) -> ConverterExpression:
        if isinstance(node.op, ast.And):
            op = values.Op(self.default_opset, "And")
        elif isinstance(node.op, ast.Or):
            op = values.Op(self.default_opset, "Or")
        else:
            raise ValueError(self.message(node, f"Unsupported operator {node.op!r}."))

        expr = self.translate_expr(node.values[0])
        for operand in node.values[1:]:
            left, right = self._cast_like_binary_expression(
                op, expr, self.translate_expr(operand)
            )
            ovar = self.generate_unique_name()
            self.emit([ovar], op, [left, right], [])
            expr = ConverterExpression(ovar, ConverterExpressionKind.ANY)
        return expr

    def translate_bin_op_expr(self, node: ast.BinOp):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(self.message(node, f"Unsupported operator {op!r}."))

        attr = []
        if isinstance(node.op, ast.Mod) and self.is_constant_expr(node.right):
            # specific case X % f where f is a float.
            # attribute fmod=1 is added in that case.
            cst = self.eval_constant_expr(node.right)
            if isinstance(cst, float):
                attr = [self.ir_builder.make_attr("fmod", 1)]

        op = values.Op(self.default_opset, primop_map[op])
        left, right = self._cast_like_binary_expression(
            op, self.translate_expr(node.left), self.translate_expr(node.right)
        )
        return op, [left, right], attr

    def translate_unary_op_expr(self, node):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(self.message(node, self).msg(f"Unsupported operator {op!r}."))
        if self.is_constant_expr(node.operand):
            # This function changed the constant node.operand
            # and returns it. The function calling this one
            # should intercept this call and replace node
            # by node.operand.
            # This mechanism does not handle somthing like `(-(-5))`.
            if hasattr(node.operand, "value"):
                # python 3.8+
                val = node.operand.value
            elif hasattr(node.operand, "n"):
                # python 3.7
                val = float(node.operand.n)
            else:
                raise TypeError(
                    f"Unable to guess constant value from type {type(node.operand)!r} "
                    f"and attributes {dir(node.operand)!r}."
                )
            if op == ast.USub:
                cst = ast.Constant(-val, lineno=node.lineno, col_offset=node.col_offset)
                return self.translate_expr(cst)
            if op == ast.UAdd:
                return self.translate_expr(node.operand)
        opname = primop_map[op]
        operand = self.translate_expr(node.operand)
        return values.Op(self.default_opset, opname), [operand], []

    def translate_compare_expr(self, node):
        # TODO: handle multiple comparisons in one expression
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        op = type(node.ops[0])
        if op not in primop_map:
            raise ValueError(self.message(node, f"Unsupported operator {op!r}."))
        opname = primop_map[op]
        left = self.translate_expr(node.left)
        right = self.translate_expr(node.comparators[0])

        # NotEqual is not a standard ONNX op, and needs to be translated into
        # an Equal op/node followed by a Not op/node.
        op = values.Op(self.default_opset, opname if opname != "NotEqual" else "Equal")
        left, right = self._cast_like_binary_expression(op, left, right)
        if opname == "NotEqual":
            tmp = self.generate_unique_name()
            self.emit([tmp], op, [left, right], [])
            not_op = values.Op(self.default_opset, "Not")
            return not_op, [tmp], []

        return op, [left, right], []

    def translate_name_expr(self, node):
        return self.py_var_to_onnx_var(node.id, self.source_of(node))

    def translate_opset_expr(self, node) -> values.Opset:  # pylint: disable=R1710
        """Return an Opset"""
        if isinstance(node, ast.Name):
            val = self.lookup(node.id, self.source_of(node), raise_exception=False)
            if isinstance(val, values.Opset):
                return val
            self.fail(node, f"'{node.id}' is not an instance of type Opset but {type(val)}.")
        elif isinstance(node, ast.Attribute):
            self.fail(node, "Nested module unimplemented.")  # TODO
        else:
            self.fail(node, "Invalid opset expression.")

    def translate_callee_expr(self, node) -> values.Op:  # pylint: disable=R1710
        """Return an Op"""
        if isinstance(node, ast.Attribute):
            module = self.translate_opset_expr(node.value)
            self.set_default_opset(module, node)
            opname = node.attr
            if opname in module:
                return values.Op(module, node.attr)
            warn(f"'{opname}' is not a known op in '{module}'")
            return values.Op(module, node.attr)
        if isinstance(node, ast.Name):
            function_name = node.id
            found = self.lookup(function_name, self.source_of(node), raise_exception=False)
            if isinstance(found, onnxscript.OnnxFunction):
                self.current_fn.add_called_function(found)
                return found
            if isinstance(found, values.Op):
                return found
            if not found:
                if function_name not in self.default_opset:
                    warn(
                        f"Unknown function name {function_name!r}. "
                        f"The ONNX graph may not work."
                    )
                return values.Op(self.default_opset, function_name)
        self.fail(node, "Invalid callee")

    def translate_stmt(self, node, index_of_stmt=None):
        """Statement translation: A single Python statement is mapped into a
        sequence of IR statements.
        """
        if isinstance(node, ast.Assign):
            return self.translate_assign_stmt(node)
        if isinstance(node, ast.AnnAssign):
            return self.translate_assign_stmt(node)
        if isinstance(node, ast.Return):
            if index_of_stmt is not None:
                return self.translate_return_stmt(node)
            raise ValueError(
                self.message(
                    node, "Return statements are not permitted inside control-flow statements."
                )
            )
        if isinstance(node, ast.If):
            return self.translate_if_stmt(node)
        if isinstance(node, (ast.For, ast.While)):
            return self.translate_loop_stmt(node)
        if isinstance(node, ast.Expr):
            if index_of_stmt == 0 and hasattr(node, "value"):
                if hasattr(node.value, "value") and isinstance(node.value.value, str):
                    # python 3.8+
                    return self.translate_docstring(node)
                if hasattr(node.value, "s") and isinstance(node.value.s, str):
                    # python 3.7
                    return self.translate_docstring(node)
        if isinstance(node, ast.FunctionDef):
            return self.translate_nested_function_def(node)
        if analysis.is_print_call(node):
            return None
        raise ValueError(self.message(node, f"Unsupported statement type {type(node)!r}."))

    def translate_assign_stmt(self, stmt: Union[ast.Assign, ast.AnnAssign]):
        def assign(lhs, rhs):
            info = self.source_of(lhs)
            if isinstance(lhs, ast.Name):
                lhs = lhs.id
                t = self.translate_expr(rhs, lhs).name
                if isinstance(stmt, ast.AnnAssign):
                    var = values.Dynamic(
                        t,
                        values.DynamicKind.Intermediate,
                        info,
                        typeinfo=self.eval_constant_expr(stmt.annotation),
                    )
                else:
                    var = values.Dynamic(t, values.DynamicKind.Intermediate, info)
                self.bind(lhs, var)
            elif isinstance(lhs, ast.Tuple):

                def id(x):
                    assert isinstance(x, ast.Name)
                    return x.id

                ids = [id(x) for x in lhs.elts]
                onnxids = self.translate_expr(rhs, ids).name
                for x, y in zip(ids, onnxids):
                    self.bind(x, values.Dynamic(y, values.DynamicKind.Intermediate, info))
            else:
                fail("Unsupported construct in LHS of assignment.")

        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        else:
            targets = [stmt.target]
        if len(targets) != 1:
            self.fail(stmt, "Multi-assignment not supported.")
        lhs = targets[0]
        rhs = stmt.value
        if isinstance(rhs, ast.Tuple):
            if not isinstance(lhs, ast.Tuple):
                self.fail(lhs, f"Left term must be a tuple not {type(lhs)!r}.")
            if len(lhs.elts) != len(rhs.elts):
                self.fail(
                    stmt, "Expected same number of elements on lhs and rhs of assignments."
                )
            for p, r in zip(lhs.elts, rhs.elts):
                assign(p, r)
        else:
            assign(lhs, rhs)

    def translate_return_stmt(self, stmt: ast.Return):
        def check_num_outputs(n):
            if self.returntype is not None:
                if n != len(self.returntype):
                    raise SyntaxError(
                        self.message(
                            stmt,
                            f"Mismatch in number of return values and types. Keyword "
                            f"'return' cannot be used in a subgraph (test, loop).  "
                            f"returntype is {self.returntype!r}, num_outputs={n!r}.",
                        )
                    )

        def ret(exp, i, suffix):
            preferred_name = f"return_val{suffix}"
            return_var = self.translate_expr(exp, preferred_name).name
            val = self.lookup(return_var, self.source_of(exp), False)
            if val and val.kind == values.DynamicKind.Input:
                # In ONNX, a graph-input cannot be an output of the graph.
                # We need to insert a copy.
                return_var = self.emit_copy(return_var, preferred_name)
            if self.returntype is None:
                t = None
            else:
                t = self.returntype[i]
            self.ir_builder.add_output(self.current_fn, return_var, t, self.source_of(stmt))
            return return_var

        val = stmt.value
        assert val is not None, "Return statement without return-value not supported."
        if isinstance(val, ast.Tuple):
            check_num_outputs(len(val.elts))
            return [ret(exp, i, str(i)) for i, exp in enumerate(val.elts)]
        check_num_outputs(1)
        return ret(val, 0, "")

    def translate_if_stmt(self, stmt: ast.If):
        if hasattr(stmt, "live_out"):
            live_defs = list(stmt.live_out.intersection(analysis.defs(stmt)))
        else:
            live_defs = list(analysis.defs(stmt))
        test = self.translate_expr(stmt.test, "cond").name
        lineno = self.source_of(stmt).lineno
        thenGraph, sub_fct_then = self.translate_block(
            stmt.body, f"thenGraph_{lineno}", live_defs, parent_stmt=stmt
        )
        thenAttr = self.ir_builder.make_attr("then_branch", thenGraph)
        elseGraph, sub_fct_else = self.translate_block(
            stmt.orelse, f"elseGraph_{lineno}", live_defs, parent_stmt=stmt
        )
        elseAttr = self.ir_builder.make_attr("else_branch", elseGraph)

        def rename(x):
            r = self.generate_unique_name(x)
            self.bind(
                x,
                values.Dynamic(r, values.DynamicKind.Intermediate, self.source_of(stmt)),
            )
            return r

        # no break condition
        renamed = [rename(x) for x in live_defs]
        if not renamed:
            self.fail(stmt, "A subgraph for a test do not have any output variable.")

        sub_functions = {}
        sub_functions.update(sub_fct_then)
        sub_functions.update(sub_fct_else)
        if renamed == [test]:
            self.fail(stmt, f"Input and output cannot be the same {renamed!r}.")
        self.emit(
            renamed,
            values.Op(self.default_opset, "If"),
            [test],
            [thenAttr, elseAttr],
            sub_functions=sub_functions,
        )

    def translate_loop_stmt(self, loop_stmt: Union[ast.For, ast.While]):
        # loop-variable
        if isinstance(loop_stmt, ast.For):
            if not isinstance(loop_stmt.target, ast.Name):
                self.fail(loop_stmt, "For loop target must be a single variable.")
            p_loop_var = loop_stmt.target.id
            # iter
            iter = loop_stmt.iter
            assert isinstance(iter, ast.Call), "Loop bound not a call."
            if not isinstance(iter.func, ast.Name):
                self.fail(loop_stmt, f"Unsupported loop bound {iter.func!r}.")
            if iter.func.id != "range":
                self.fail(
                    loop_stmt, "Unsupported loop bound, only function 'range' is allowed."
                )
            if not iter.args or len(iter.args) != 1:
                self.fail(loop_stmt, "Unsupported loop bound, it should be 'range(?)'.")
            assert not iter.keywords, "Unsupported loop bound."
            o_loop_bound = self.translate_expr(iter.args[0], "loop_bound").name
            o_cond_var = self.generate_unique_name("cond_in")
            i_cond_var = o_cond_var
            cond_while = None
        elif isinstance(loop_stmt, ast.While):
            test = loop_stmt.test
            if not isinstance(test, ast.Name):
                self.fail(
                    loop_stmt,
                    "Unexpected condition type {type(loop_stmt)!r} for a while loop, "
                    "it should be 'while <condition_name>:'.",
                )
            p_loop_var = "infinite_loop"
            o_loop_bound = ""
            i_cond_var = test.id
            cond_while = test.id
            o_cond_var = None
            # we need to go through all the instructions to see
            # which instruction defines the condition test.id
        else:
            self.fail(loop_stmt, f"Unexpected loop type {type(loop_stmt)!r}.")
        # analyze loop body
        exposed_uses = analysis.exposed_uses(loop_stmt.body, self.message)
        vars_def_in_loop = analysis.defs(loop_stmt.body)
        loop_state_vars = vars_def_in_loop.intersection(exposed_uses | loop_stmt.live_out)
        scan_outputs = set()  # TODO
        outputs = list(loop_state_vars | scan_outputs)

        # loop-condition:
        o_true = self.emit_const(True, "true", self.source_of(loop_stmt))
        # o_loop_bound = self.emit_const(3, "loop_bound")

        # build loop_body
        self.enter_scope("loop_body", loop_stmt)
        o_loop_var = self.generate_unique_name(p_loop_var)
        self.ir_builder.add_input(
            self.current_fn,
            o_loop_var,
            onnx_types.INT64,
            self.source_of(loop_stmt),
        )
        self.bind(
            p_loop_var,
            values.Dynamic(o_loop_var, values.DynamicKind.Loop, self.source_of(loop_stmt)),
        )

        self.ir_builder.add_input(
            self.current_fn,
            i_cond_var,
            onnx_types.BOOL,
            self.source_of(loop_stmt),
        )

        for pv in loop_state_vars:
            ov = self.generate_unique_name(pv)
            # TODO: retrieve the annotation for variable pv is any is specified.
            # typeinfo = self.eval_constant_expr(pv.annotation)
            typeinfo = None
            self.ir_builder.add_input(self.current_fn, ov, typeinfo, self.source_of(loop_stmt))
            self.bind(
                pv,
                values.Dynamic(ov, values.DynamicKind.Loop, self.source_of(loop_stmt)),
            )

        condition_name = None
        operator_name = "Identity"
        for i, s in enumerate(loop_stmt.body):
            # We first need to intercept a break instruction in test block.
            # It must be something like `if <condition_name>: break`.
            # This instruction must be the last of the loop body.
            if isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], ast.Break):
                if not isinstance(s.test, ast.Name):
                    self.fail(
                        s,
                        f"Instruction break can be introduced with test but it must be "
                        f"if <condition>: break. However condition is of type "
                        f"{type(s.test)!r}.",
                    )
                if i != len(loop_stmt.body) - 1:
                    self.fail(s, "Instruction break must be the last one of the loop.")

                current_scope = self.current_scope()
                if s.test.id not in current_scope:
                    self.fail(
                        loop_stmt,
                        f"Unable to find condition variable {s.test.id!r} in known "
                        f"variables {list(current_scope)!r}.",
                    )
                condition_name = current_scope[s.test.id].value
                operator_name = "Not"
                continue
            self.translate_stmt(s)

        o_cond_out = self.generate_unique_name("cond_out")

        if cond_while is not None:
            # Loop while
            current_scope = self.current_scope()
            if cond_while not in current_scope:
                self.fail(
                    loop_stmt,
                    f"Unable to find condition variable {cond_while!r} in known "
                    f"variables {list(current_scope)!r}.",
                )
            o_cond_var = current_scope[cond_while].value

        self.emit(
            [o_cond_out],
            values.Op(self.default_opset, operator_name),
            [condition_name or o_cond_var],
            [],
        )

        self.ir_builder.add_output(
            self.current_fn,
            o_cond_out,
            onnx_types.BOOL,
            self.source_of(loop_stmt),
        )
        for pv in loop_state_vars:
            ov = self.py_var_to_onnx_var(pv, self.source_of(loop_stmt))
            if ov not in self.current_fn.assigned_names:
                # When converting the loop-body into a graph, we need to handle
                # identity assignments of the form "x = y" inside the loop body
                # specially if y represents a value computed outside the loop body.
                # In this case, we create a copy of y, treating the statement as
                # shorthand for "x = op.Identity(y)".
                ov = self.emit_copy(ov, pv)
            # TODO: retrieve variable type for the annotation if any.
            typeinfo = None
            self.ir_builder.add_output(
                self.current_fn, ov, typeinfo, self.source_of(loop_stmt)
            )
        body = self.exit_scope()
        inputs = [o_loop_bound, o_true] + [
            self.py_var_to_onnx_var(pv, self.source_of(loop_stmt)) for pv in loop_state_vars
        ]
        graph, sub_functions = body.to_graph_and_functions()
        attrs = [self.ir_builder.make_attr("body", graph)]
        return self.emit_loop(
            outputs,
            "Loop",
            inputs,
            attrs,
            sub_functions=sub_functions,
            info=self.source_of(loop_stmt),
        )

    def translate_block(self, stmts, name, live_defs, parent_stmt=None):
        """Translation of a statement-block to GraphProto attribute."""
        info_stmt = stmts[0] if len(stmts) > 0 else parent_stmt
        self.enter_scope(name, None)
        for s in stmts:
            self.translate_stmt(s)
        for pvar in live_defs:
            if pvar in self.current_scope():
                pv_val = self.current_scope()[pvar]
                output = self.to_onnx_var(pv_val, pvar)
                if output not in self.current_fn.assigned_names:
                    # To return an outer-scope variable, an ONNX Graph has to
                    # use an explicit copy via Identity.
                    output = self.emit_copy(output, pvar)
                self.ir_builder.add_output(
                    self.current_fn,
                    output,
                    pv_val.typeinfo,
                    self.source_of(info_stmt),
                )
            else:
                pv_val = None
                for scope in self.locals:  # TODO: skip current_scope
                    if pvar in scope:
                        pv_val = scope[pvar]
                        break
                if pv_val is None:
                    self.fail(
                        stmts[0],
                        f"Variable {pvar} is not assigned a value along a conditional "
                        f"branch, known variables: {list(self.locals)}.",
                    )
                # introduce a copy
                ovar = self.generate_unique_name(pvar)
                self.emit(
                    [ovar],
                    values.Op(self.default_opset, "Identity"),
                    [self.to_onnx_var(pv_val, pvar)],
                    [],
                )
                # TODO: retrieve the annotation if any.
                typeinfo = None
                self.ir_builder.add_output(
                    self.current_fn, ovar, typeinfo, self.source_of(info_stmt)
                )
        graph = self.exit_scope()
        return graph.to_graph_and_functions()

    def translate_nested_function_def(self, fn: ast.FunctionDef):
        """Translate a nested function definition."""
        self.enter_scope(fn.name, fn)
        self.translate_function_def(fn)
        function_ir = self.exit_scope()
        outer_scope_vars = analysis.outer_scope_variables(fn, self.message)
        function_ir.outer_scope_variables = [
            (var, self.lookup(var, self.source_of(fn))) for var in outer_scope_vars
        ]
        self.bind(fn.name, function_ir)
        # TODO: Does not yet handle nested functions within nested functions.
        self.current_fn.add_nested_function(function_ir)

    def translate_function_def(self, fn: ast.FunctionDef):
        logger.debug("Converter:translate_function_def:%s", fn.name)
        args = fn.args
        if args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg:
            warn(f"{fn.name}: Unsupported feature in function signature.")
        domain = self.this_module.domain
        self.current_fn = self.ir_builder.new_function(fn.name, domain, True)
        for i, x in enumerate(args.args):
            arg_with_default_start_index = len(args.args) - len(args.defaults)
            if args.defaults and i >= arg_with_default_start_index:
                default_value = self.eval_constant_expr(
                    args.defaults[i - arg_with_default_start_index]
                )
            else:
                default_value = None
            if x.annotation:
                typeinfo = self.eval_constant_expr(x.annotation)
                if not ta.is_valid_type(typeinfo):
                    self.warn(
                        x.annotation,
                        f"Unsupported type annotation for argument {x.arg}.",
                    )
                    typeinfo = None
            else:
                # The code can only be exported as a function.
                typeinfo = None
            if typeinfo and ta.is_attr_type(typeinfo):
                self.ir_builder.add_attr_parameter(
                    self.current_fn,
                    x.arg,
                    default_value,
                )
                self.bind(x.arg, values.AttrRef(x.arg, typeinfo, self.source_of(x)))
            else:
                self.ir_builder.add_input(self.current_fn, x.arg, typeinfo, self.source_of(x))
                self.used_vars.add(x.arg)
                self.bind(
                    x.arg,
                    values.Dynamic(x.arg, values.DynamicKind.Input, self.source_of(x)),
                )
        if fn.returns:
            type_annotation = self.eval_constant_expr(fn.returns)
            self.returntype = ta.get_return_types(type_annotation)
            invalid = False
            for t in self.returntype:
                if not ta.is_valid_type(t):
                    self.warn(
                        fn.returns,
                        f"Unsupported type annotation for return value {t}.",
                    )
                    invalid = True
            if invalid:
                self.returntype = None
        else:
            self.returntype = None
        for i, s in enumerate(fn.body):
            self.translate_stmt(s, index_of_stmt=i)
        return self.current_fn

    def top_level_stmt(self, stmt: ast.FunctionDef):
        if isinstance(stmt, ast.FunctionDef):
            self.init_function_translation()
            if self.default_opset_ is None:
                opset = self.find_onnx_opset(stmt)
                if opset:
                    self.set_default_opset(opset, stmt)
            analysis.do_liveness_analysis(stmt, self.message)
            fn_ir = self.translate_function_def(stmt)
            fn_ir.debug_print()
            self.this_module.add_function_def(fn_ir)
            return fn_ir
        raise ValueError(f"Unsupported top-level statement type {type(stmt)!r}.")
