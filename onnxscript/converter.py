# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import ast
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import onnx

import onnxscript
from onnxscript import irbuilder, onnx_types, sourceinfo, values
from onnxscript import type_annotation as ta
from onnxscript._internal import analysis, ast_utils, autocast, param_manipulation

PY_VERSION_GE_39 = ast_utils.PY_VERSION_GE_39


logger = logging.getLogger("onnxscript")


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


class Variable:
    """Represents an ONNX variable.

    TODO(rama): Consider merging this with IRVar. However, "castable" is specific to this
    converter.
    """

    def __init__(self, name: str, castable: bool = False):
        """Initialize the instance.

        Args:
           name: Name of the ONNX variable
           castable: Whether this variable is castable to a desired target type.
              Used for ONNX variables representing constants created from python values
              like 0 or 1 or 0.5 which are treated as polymorphic values castable to other
              types as needed.
        """
        self.name = name
        self.is_castable = castable

    def __str__(self) -> str:
        return self.name


if TYPE_CHECKING:
    # The type-alias LocalSymValue represents the types of values that local names in a
    # script-function may be bound to during translation, (ONNX IR values).
    # TODO(rama): Rationalize this and values.SymbolValue

    LocalSymValue = Union[values.SymbolValue, irbuilder.IRFunction]

    # The type-alias PyValue is used to represent the types of python values that may be used
    # in an ONNX Script function.
    # TODO(rama): Flesh out the set of valid types here. These include values such as
    # 1 (int), 1.0 (float), [2, 4], [1.0], etc. which will be converted to ONNX, for
    # use as value-parameters or attribute-parameters in an ONNX call (Node).

    PyValue = Any

    # The type-alias SymValue denotes values that an identifier may be bound to during
    # translation. A local name will be bound to a LocalSymValue, while a global name
    # will be bound to a PyValue.

    SymValue = Union[LocalSymValue, PyValue]

    # PreferredName is a type-alias used to represent the preferred name used in the generated
    # ONNX for a value returned by an expression. There is no guarantee that the specified
    # name will be used exactly. The converter will modify the name (with a suffix),
    # if necesssary, to ensure that it is unique (to ensure ONNX's SSA requirement).

    PreferredName = str

    # The type-alias OnnxVar indicates variable names used in the generated ONNX.
    OnnxVarName = str


class Converter:
    """Main class to translate python code into ONNX operators.

    Args:
        ir_builder: convert AST node into ONNX structures, if None,
            class :class:`onnxscript.irbuilder.IRBuilder` is used

    The class uses logger `onnxscript`. Logging can be enabled with the following code:

    ::

        import logging
        logging.basicConfig(level=logging.DEBUG)

    Or if you need to enable only the logger used by this module:

    ::

        import logging
        logger = logging.getLogger('onnxscript')
        logger.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        logger.addHandler(console)
    """

    def __init__(
        self,
        ir_builder: Optional[irbuilder.IRBuilder] = None,
        opset: Optional[values.Opset] = None,
        global_names: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
        default_opset: Optional[values.Opset] = None,
    ):
        self.ir_builder = ir_builder or irbuilder.IRBuilder()
        self.source = source
        if global_names is not None:
            # We make a copy in case function eval modifies it.
            self.globals = global_names.copy()
        self.this_module = opset
        self.default_opset_ = default_opset

        # States initialized by `_init_function_translation`
        self._outer: List[irbuilder.IRFunction] = []
        self._current_fn: irbuilder.IRFunction = None
        self._nextvar: int = 0
        self._used_vars: set[str] = set()
        self._locals: List[Dict[str, LocalSymValue]] = [{}]

    @property
    def default_opset(self) -> values.Opset:
        if self.default_opset_ is None:
            raise RuntimeError(
                "default_opset must be specified in script for functions "
                "that do not contain any use of an ONNX opset."
            )
        return self.default_opset_

    def _set_default_opset(self, opset: values.Opset, node: ast.AST) -> None:
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

    def _find_onnx_opset(self, node: ast.AST) -> Optional[values.Opset]:
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
            res = self._find_onnx_opset(child)
            if res is not None:
                return res
        return None

    def _init_function_translation(self) -> None:
        """Initialize self for translating a new (top-level) function."""
        self._outer = []
        self._current_fn: Optional[irbuilder.IRFunction] = None
        self._nextvar = 0
        self._used_vars = set()
        self._locals: List[Dict[str, LocalSymValue]] = [{}]

    def _source_of(self, node: ast.AST) -> sourceinfo.SourceInfo:
        return sourceinfo.SourceInfo(node, self.source, self._current_fn.name)

    def _message(self, node: ast.AST, error_msg: str) -> str:
        """Constructs an error _message containing source information about an ast node."""
        return self._source_of(node).msg(error_msg)

    def warn(self, node: ast.AST, error_msg: str) -> None:
        warn(self._message(node, error_msg))

    def fail(self, node: ast.AST, error_msg: str) -> NoReturn:
        fail(self._message(node, error_msg))

    # Name resolution and namescopes: This component handles the following aspects:
    # * Name-scopes are different in Python and the generated ONNX:
    #   - Control-flow blocks (a loop body or the then-or-else block of an if-stmt)
    #     form part of the same name-scope in Python, but will be mapped to a nested
    #     name-scope (as a sub-graph) in ONNX.
    # * Script-time name-value tracking: Name _lookup during script-time returns
    #   statically-known information about the value the name will have at runtime.
    def _enter_scope(self, name: str, parent_node: ast.AST):
        """Enter a control-flow block (a loop body or if-then-else branch).
        The block is translated into a nested-scope in ONNX.
        """
        self._outer.insert(0, self._current_fn)
        self._current_fn = self.ir_builder.new_function(name)
        self._locals.insert(0, {})
        logger.debug("Converter:_enter_scope:%d:node:%s", len(self._locals), type(parent_node))

    def _exit_scope(self) -> irbuilder.IRFunction:
        """Exit from a control-flow block (a loop body or if-then-else branch)."""
        logger.debug("Converter:_exit_scope:%d", len(self._locals))
        graph = self._current_fn
        self._current_fn = self._outer.pop(0)
        self._locals.pop(0)
        return graph

    def _current_scope(self) -> Dict[str, LocalSymValue]:
        return self._locals[0]

    def _bind(self, name: str, val: LocalSymValue) -> None:
        logger.debug("Converter:_bind:%s", name)
        self._locals[0][name] = val

    def _lookup(
        self, name: str, info: sourceinfo.SourceInfo, raise_exception: bool = True
    ) -> SymValue:
        for scope in self._locals:
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
        while r in self._used_vars:
            r = f"{candidate}_{self._nextvar}"
            self._nextvar = self._nextvar + 1
        self._used_vars.add(r)
        return r

    def _make_onnx_attr(
        self, attrname: str, attrval: Any, attrtype: Optional[int] = None
    ) -> irbuilder.IRAttributeValue:
        def tensor_name_generator() -> str:
            """Return name to be used for tensor, if we need to create one."""
            return self.generate_unique_name(f"attr_{attrname}")

        proto = autocast.pyvalue_to_onnx_attribute(
            attrname, attrval, tensor_name_generator, attrtype
        )
        return self.ir_builder.make_attr(proto)

    def _to_onnx_attr_ref(
        self, val: values.AttrRef, info: Optional[sourceinfo.SourceInfo]
    ) -> irbuilder.IRAttributeValue:
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

    def _to_onnx_var(
        self,
        val: values.SymbolValue | PyValue,
        target: Optional[PreferredName] = None,
        info: Optional[sourceinfo.SourceInfo] = None,
    ) -> Variable:
        if isinstance(val, values.AttrRef):
            # promote attribute to value
            result = self.generate_unique_name(target or "tmp")
            attr = self._to_onnx_attr_ref(val, info)
            self.emit([result], values.Op(self.default_opset, "Constant"), [], [attr])
            if ta.base_type_is_bool(val.typeinfo):
                # ONNX attributes use an int-encoding for bools, but ONNX tensor types
                # distinguish between int and bool. So we cast the int tensor to a bool tensor,
                # to promote a (python) bool attribute to a ONNX bool tensor.
                result_as_bool = self.generate_unique_name(result + "_as_bool")
                cast_attr = self._make_onnx_attr("to", onnx_types.BOOL.dtype)
                self.emit(
                    [result_as_bool],
                    values.Op(self.default_opset, "Cast"),
                    [result],
                    [cast_attr],
                )
                return Variable(result_as_bool, True)
            return Variable(result, True)
        if isinstance(val, values.Dynamic):
            return Variable(val.value)
        # Assume value is a python-value convertible to a tensor
        # TODO: check if value is convertible to a TensorProto, so that we can
        # produce a better error _message otherwise
        return self._emit_const(val, target or "tmp", info)

    def _py_var_to_onnx_var(self, py_var: str, info: sourceinfo.SourceInfo) -> Variable:
        return self._to_onnx_var(self._lookup(py_var, info), target=py_var, info=info)

    def emit(
        self,
        outputs: Sequence[str],
        callee: values.Op | str,
        inputs: Sequence[Optional[str]],
        attrs: Optional[Sequence[irbuilder.IRAttributeValue]] = None,
        sub_functions: Optional[dict[str, onnx.FunctionProto]] = None,
    ):
        if not isinstance(callee, values.Op):
            callee = values.Op(self.default_opset, callee)
        if attrs is None:
            attrs = []
        if sub_functions is None:
            sub_functions = {}
        self.ir_builder.add_stmt(
            self._current_fn,
            outputs,
            callee,
            inputs,
            attrs,
            sub_functions,
        )

    def _emit_const(
        self,
        pyvalue: PyValue,
        suggested_name: Optional[PreferredName],
        info: sourceinfo.SourceInfo,
    ) -> Variable:
        if suggested_name is None:
            if isinstance(pyvalue, int):
                if pyvalue >= 0:
                    suggested_name = f"int64_{pyvalue}"
                else:
                    suggested_name = f"int64_m{abs(pyvalue)}"
            elif (
                isinstance(pyvalue, list) and len(pyvalue) == 1 and isinstance(pyvalue[0], int)
            ):
                if pyvalue[0] >= 0:
                    suggested_name = f"int64_{pyvalue[0]}_1d"
                else:
                    suggested_name = f"int64_m{abs(pyvalue[0])}_1d"
            else:
                suggested_name = "const"
        ovar = self.generate_unique_name(suggested_name)
        try:
            tensor = autocast.pyvalue_to_onnx_tensor(ovar, pyvalue)
        except ValueError as e:
            fail(info.msg(str(e)))
        attr = self._make_onnx_attr("value", tensor)
        self.emit([ovar], values.Op(self.default_opset, "Constant"), [], [attr])
        return Variable(ovar, True)

    def _emit_copy(self, original_var: str, suggested_name: str) -> str:
        """Emits a copy statement, using the ONNX Identity operator."""
        new_var = self.generate_unique_name(suggested_name)
        self.emit([new_var], "Identity", [original_var])
        return new_var

    def _is_constant_expr(self, node: ast.AST) -> None:
        if isinstance(node, ast.UnaryOp):
            return self._is_constant_expr(node.operand)
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
            return all(self._is_constant_expr(c) for c in ast.iter_child_nodes(node))
        return False

    def _eval_constant_expr(self, expr: ast.AST) -> PyValue:
        """Evaluates a sub-expression that is assumed to represent a constant value.
        The expression can refer only to global names (inherited from the scope
        where the script is evaluated) and cannot refer to local names defined
        within the script.) Further, these expressions are assumed to be constants.
        Thus, any subsequent mutation of any state/variables (used in computing
        this constant value) will potentially lead to unexpected behavior (such
        as divergence between eager-mode execution and evaluation of the ONNX
        function.)
        """
        # TODO: assert (self._is_constant_expr(expr))
        # TODO: Refine types
        locals: dict[Any, Any] = {}
        expr = ast.Expression(expr, lineno=expr.lineno, col_offset=expr.col_offset)
        cpl = compile(expr, filename="<ast>", mode="eval")
        try:
            return eval(cpl, self.globals, locals)  # pylint: disable=eval-used
        except NameError as e:
            raise NameError(
                self._message(
                    expr,
                    f"Missing names, globals contains {list(self.globals)!r}, "
                    f"locals {list(locals)!r}.",
                )
            ) from e

    def _translate_attr(
        self,
        attr_name: str,
        expr: ast.AST,
        attr_meta: Optional[onnx.defs.OpSchema.Attribute] = None,
    ) -> Optional[irbuilder.IRAttributeValue]:
        """Translate an attribute-value specification of the form `attr_name=<expr>`
        in a call to an op. expr is an AST. The following cases are supported:
        * Expr evaluates to a script-time constant (a python-value) that can be mapped
        into an ONNX attribute value, or
        * Expr evaluates to None, in which case None is returned, or
        * Expr must be an attribute-reference, that is a name representing an
        attribute-parameter of a containing function.
        """

        if isinstance(expr, ast.Name):
            val = self._lookup(expr.id, self._source_of(expr))
            if isinstance(val, values.AttrRef):
                attr_ref = self.ir_builder.make_attr_ref(attr_name, val.value, val.typeinfo)
                if attr_meta is not None and (attr_ref.type != attr_meta.type):
                    self.fail(
                        expr,
                        f"Attribute type '{attr_ref.type}' does not match expected type '{attr_meta.type}'",
                    )
                return attr_ref
            if isinstance(val, irbuilder.IRFunction):
                # Check that outer-scope variables referenced by function have same value
                # at function-definition site and use-as-attribute site, to avoid errors.
                for pyvar, previous in val.outer_scope_variables:
                    current = self._lookup(pyvar, self._source_of(expr))
                    if current.value != previous.value:
                        self.fail(
                            expr,
                            f"Outer scope variable '{pyvar}' referenced by function "
                            f"'{expr.id!r}' modified.",
                        )

                # Create GraphProto attribute
                val = val.to_graph_proto()
        else:
            val = self._eval_constant_expr(expr)

        # In ONNX, there is no way to explicitly specify a None value for an attribute.
        # Instead, the attribute must be omitted from the attribute list.
        # Hence, we do not create an attribute-proto if the value is None.
        # The caller is responsible for omitting such attribute-values from the list of attributes
        # in a NodeProto.
        if val is None:
            if attr_meta and attr_meta.required:
                self.fail(expr, f"Attribute '{attr_name}' is required.")
            return None
        attr_type = attr_meta.type if attr_meta else None
        attr = self._make_onnx_attr(attr_name, val, attr_type)
        if attr_meta and (attr.type != attr_meta.type):
            self.fail(
                expr,
                f"Attribute type '{attr.type}' does not match expected type '{attr_meta.type}'",
            )
        return attr

    def _translate_docstring(self, node: ast.Expr) -> None:
        if hasattr(node.value, "value"):
            # python 3.8+
            return self.ir_builder.add_docstring(self._current_fn, node.value.value)
        raise TypeError(
            f"Unexpected type {type(node)!r} for node. Unsupoorted version of python."
        )

    def _translate_expr(
        self, node: ast.AST, target: Optional[PreferredName] = None
    ) -> Variable:
        """Expression-translation generates "IR statements/nodes" that compute the value of
        the expression into a target-variable, and returns the variable that is
        assigned this value.
        """
        if isinstance(node, ast.Call):
            r = self._translate_call_expr(node)
        elif isinstance(node, (ast.BinOp, ast.BitAnd, ast.BitOr)):
            r = self._translate_binary_op_expr(node)
        elif isinstance(node, ast.UnaryOp):
            r = self._translate_unary_op_expr(node)
        elif isinstance(node, ast.Compare):
            r = self._translate_compare_expr(node)
        elif isinstance(node, ast.Name):
            r = self._translate_name_expr(node)
        elif isinstance(node, ast.Subscript):
            r = self._translate_subscript_expr(node, target)
        elif self._is_constant_expr(node):
            r = self._emit_const(self._eval_constant_expr(node), target, self._source_of(node))
        else:
            raise ValueError(
                self._message(node, f"Unsupported expression type {type(node)!r}.")
            )
        if isinstance(r, Variable):
            return r
        callee, args, attrs = r
        target = "tmp" if target is None else target
        assert isinstance(target, str)
        result = self.generate_unique_name(target)
        self.emit([result], callee, args, attrs)
        return Variable(result)

    def _translate_opt_expr(self, node: ast.expr) -> Optional[Variable]:
        """Translation of an expression where "None" is permitted (eg., for an optional argument).
        None is represented as a NameConstant in Python 3.7 and Constant in Python 3.9.
        """
        if isinstance(node, (ast.NameConstant, ast.Constant)) and (node.value is None):
            return None
        return self._translate_expr(node)

    def _translate_subscript_expr(
        self, node: ast.Subscript, target: Optional[PreferredName]
    ) -> Variable:
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
        var = self._translate_expr(node.value)
        var_name = var.name
        if target is None:
            target = f"{var_name}_subscripted"
        target = self.generate_unique_name(target)
        indices = ast_utils.normalize_subscript_expr(node)
        info = self._source_of(node.slice if PY_VERSION_GE_39 else node)

        # Create cached int constants:
        # TODO: Do this at a graph-scope level.
        cached_int_consts = {}

        def const_1d(value, name: Optional[str] = None):
            nonlocal cached_int_consts
            if value not in cached_int_consts:
                cached_int_consts[value] = self._emit_const([value], name, info)
            return cached_int_consts[value]

        def one_1d():
            return const_1d(1)

        # Max/min 64-bit int values are used to represent default values for start/stop in Slice.
        maxint = (1 << 63) - 1
        minint = -(1 << 63)

        def translate_slice_component(
            node_arg, default_value: Optional[int] = None
        ) -> tuple[str, Optional[int]]:
            """Translate optional start/stop/step component of a Slice expression."""
            if node_arg is None:
                if default_value is None:
                    # TODO: Emit "Where(step > 0, pos_default, neg_default)"
                    raise RuntimeError(
                        "Default start/stop not supported when step direction is unknown."
                    )
                return const_1d(default_value), default_value

            if self._is_constant_expr(node_arg):
                cst = self._eval_constant_expr(node_arg)
                if isinstance(cst, int):
                    return const_1d(cst), cst
                else:
                    raise RuntimeError(f"Slice component type must be int, not {type(cst)}")
            else:
                name = self._translate_expr(node_arg).name
                reshaped = self.generate_unique_name(f"{name}_reshaped")
                self.emit(
                    [reshaped],
                    values.Op(self.default_opset, "Reshape"),
                    [name, one_1d().name],
                    [],
                )
                return reshaped, None

        def translate_slice(slice_expr: ast.Slice) -> tuple[str, str, str]:
            """Translate slice-expression of the form from:to:step."""
            step_name, step = translate_slice_component(slice_expr.step, 1)
            if step is None:
                # Step direction unknown.
                # TODO: Handle default-values using runtime check on sign of step.
                lower_name, _ = translate_slice_component(slice_expr.lower, None)
                upper_name, _ = translate_slice_component(slice_expr.upper, None)
            elif step > 0:
                lower_name, _ = translate_slice_component(slice_expr.lower, 0)
                upper_name, _ = translate_slice_component(slice_expr.upper, maxint)
            else:
                lower_name, _ = translate_slice_component(slice_expr.lower, maxint)
                upper_name, _ = translate_slice_component(slice_expr.upper, minint)
            return (lower_name, upper_name, step_name)

        # An input like X[2] is translated into a Gather op.
        # An input like X[1:5:2] is translated into a Slice op.
        # An input like X[2, 3] is translated into a Slice + Squeeze (instead of two Gathers),
        #   as an optimization.
        # An input like X[I, J] is translated into two Gathers (which is correct whatever the
        #   rank of I and J)
        # To replace multiple Gathers by the Slice we need to know that the index-values
        # are scalars.

        # As the first step, we partition the index elements into four kinds: Slice (eg., 1:5:2),
        # known-to-be-scalar (eg., 2), other-tensor (eg., I), skip/no-op (that is, just ":")
        sliced_indices: List[Tuple[int, ast.expr]] = []
        scalar_indices: List[Tuple[int, ast.expr]] = []
        non_scalar_indices: List[Tuple[int, ast.expr]] = []
        for axis, elt in enumerate(indices):
            if isinstance(elt, ast.Slice):
                # Add to sliced_indices, unless it is "::", which is a no-op.
                if not (elt.lower is None and elt.upper is None and elt.step is None):
                    sliced_indices.append((axis, elt))
            elif self._is_constant_expr(elt) and isinstance(
                self._eval_constant_expr(elt), int
            ):
                scalar_indices.append((axis, elt))
            else:
                non_scalar_indices.append((axis, elt))
        if not (sliced_indices or scalar_indices or non_scalar_indices):
            # Edge case: no index specified. Eg. A[:, :]
            self.emit([target], "Identity", [var_name])
            return Variable(target)
        if sliced_indices or len(scalar_indices) > 1:
            # We emit a Slice operation if we have any indices like 1:5:2 or if the number of
            # scalar indices (like 2) is more than 1.
            starts = []
            ends = []
            axes = []
            steps = []
            squeezed_axes = []
            for axis, expr in scalar_indices:
                # Treat a scalar index i as slice "i:i+1:1", but squeeze the axis finally.
                # TODO: handle negative i
                index = self._eval_constant_expr(expr)
                squeezed_axes.append(axis)
                kwargs = dict(
                    lineno=getattr(expr, "lineno", node.lineno),
                    col_offset=getattr(expr, "col_offset", node.col_offset),
                )
                element = ast.Slice(
                    ast.Constant(index, **kwargs),
                    ast.Constant(index + 1, **kwargs),
                    ast.Constant(1, **kwargs),
                )
                sliced_indices.append((axis, element))
            scalar_indices = []
            for axis, element in sliced_indices:
                axis_var = const_1d(axis)
                inputs = translate_slice(element)
                starts.append(inputs[0])
                ends.append(inputs[1])
                axes.append(axis_var.name)
                steps.append(inputs[2])

            if len(starts) > 1:
                axis_0_attr = self._make_onnx_attr("axis", 0)
                start_name = self.generate_unique_name(f"{var_name}_start")
                self.emit([start_name], "Concat", starts, [axis_0_attr])

                end_name = self.generate_unique_name(f"{var_name}_end")
                self.emit([end_name], "Concat", ends, [axis_0_attr])

                axes_name = self.generate_unique_name(f"{var_name}_axis")
                self.emit([axes_name], "Concat", axes, [axis_0_attr])

                steps_name = self.generate_unique_name(f"{var_name}_step")
                self.emit([steps_name], "Concat", steps, [axis_0_attr])
            else:
                start_name = starts[0]
                end_name = ends[0]
                axes_name = axes[0]
                steps_name = steps[0]

            if squeezed_axes:
                sliced_name = self.generate_unique_name(f"{var_name}_sliced")
                self.emit(
                    [sliced_name],
                    "Slice",
                    [var_name, start_name, end_name, axes_name, steps_name],
                )
                squeezed_axes = self._emit_const(squeezed_axes, "squeezed_axes", info)

                if non_scalar_indices:  # use temporary to store result of squeeze
                    result = self.generate_unique_name(f"{var_name}_squeezed")
                else:  # store squeezed result in final target
                    result = target

                self.emit([result], "Squeeze", [sliced_name, squeezed_axes])
            else:
                if non_scalar_indices:  # use temporary to store result of Slice
                    result = self.generate_unique_name(f"{var_name}_sliced")
                else:  # store result of Slice in final target
                    result = target
                slice_inputs = [var_name, start_name, end_name, axes_name, steps_name]
                self.emit([result], "Slice", slice_inputs)
        else:
            result = var_name
        non_scalar_indices.extend(scalar_indices)
        if non_scalar_indices:
            last_axis, _ = non_scalar_indices[-1]
        for axis, index_expr in non_scalar_indices:
            index_value = self._translate_expr(index_expr)
            axis_attr = self._make_onnx_attr("axis", axis)
            # use Gather to perform indexing
            # Assign gathered value to either temporary or final target
            if axis != last_axis:  # use temporary to store result of Gather
                gathered = self.generate_unique_name(f"{var_name}_axis_{axis}")
            else:  # store result of Gather in final target
                gathered = target
            self.emit([gathered], "Gather", [str(result), index_value], [axis_attr])
            result = gathered

        return Variable(result)

    def _translate_call_expr(self, node: ast.Call):
        """Translates a call-expression."""
        callee = self._translate_callee_expr(node.func)
        param_schemas = callee.param_schemas()
        # If the callee's schema is available, we use it to determine the inputs and attributes.
        # Otherwise, we map named arguments to attributes and positional arguments to inputs.
        if param_schemas:
            kwargs = {x.arg: x.value for x in node.keywords}
            args, attrs = param_manipulation.separate_input_attributes_from_arguments(
                param_schemas, node.args, kwargs, fill_defaults=False
            )
            args = [self._translate_opt_expr(x) for x in args]
            attrs = [
                self._translate_attr(x, y, callee.op_schema.attributes[x])
                for x, y in attrs.items()
            ]
        else:
            args = [self._translate_opt_expr(x) for x in node.args]
            attrs = [self._translate_attr(x.arg, x.value) for x in node.keywords]
        args = autocast.static_cast_inputs(self, callee.op_schema, args)

        # In ONNX, there is no way to explicitly specify a None value for an attribute.
        # Instead, the attribute must be omitted from the attribute list.
        # Hence, we do not create an attribute-proto if the value is None.
        attrs = [attr for attr in attrs if attr is not None]
        return callee, args, attrs

    def _cast_like_binary_expression(self, op, left, right):
        schema = op.op_schema
        return autocast.static_cast_inputs(self, schema, (left, right))

    def _translate_binary_op_expr(self, node: ast.BinOp):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(self._message(node, f"Unsupported operator {op!r}."))

        attr = []
        if isinstance(node.op, ast.Mod) and self._is_constant_expr(node.right):
            # specific case X % f where f is a float.
            # attribute fmod=1 is added in that case.
            cst = self._eval_constant_expr(node.right)
            if isinstance(cst, float):
                attr = [self._make_onnx_attr("fmod", 1)]

        op = values.Op(self.default_opset, primop_map[op])
        left, right = self._cast_like_binary_expression(
            op, self._translate_expr(node.left), self._translate_expr(node.right)
        )
        return op, [left, right], attr

    def _translate_unary_op_expr(self, node):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(self._message(node, self).msg(f"Unsupported operator {op!r}."))
        if self._is_constant_expr(node.operand):
            # This function changed the constant node.operand
            # and returns it. The function calling this one
            # should intercept this call and replace node
            # by node.operand.
            # This mechanism does not handle somthing like `(-(-5))`.
            if hasattr(node.operand, "value"):
                # python 3.8+
                val = node.operand.value
            else:
                raise TypeError(
                    f"Unable to guess constant value from type {type(node.operand)!r} "
                    f"and attributes {dir(node.operand)!r}."
                )
            if op == ast.USub:
                cst = ast.Constant(-val, lineno=node.lineno, col_offset=node.col_offset)
                return self._translate_expr(cst)
            if op == ast.UAdd:
                return self._translate_expr(node.operand)
        opname = primop_map[op]
        operand = self._translate_expr(node.operand)
        return values.Op(self.default_opset, opname), [operand], []

    def _translate_compare_expr(self, node):
        # TODO: handle multiple comparisons in one expression
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        op = type(node.ops[0])
        if op not in primop_map:
            raise ValueError(self._message(node, f"Unsupported operator {op!r}."))
        opname = primop_map[op]
        left = self._translate_expr(node.left)
        right = self._translate_expr(node.comparators[0])

        # NotEqual is not a standard ONNX op, and needs to be translated into
        # an Equal op/node followed by a Not op/node.
        op = values.Op(self.default_opset, opname if opname != "NotEqual" else "Equal")
        left, right = self._cast_like_binary_expression(op, left, right)
        if opname == "NotEqual":
            tmp = self.generate_unique_name()
            self.emit([tmp], op, [left, right])
            not_op = values.Op(self.default_opset, "Not")
            return not_op, [tmp], []

        return op, [left, right], []

    def _translate_name_expr(self, node: ast.Name) -> Variable:
        return self._py_var_to_onnx_var(node.id, self._source_of(node))

    # pylint: disable=inconsistent-return-statements
    def _translate_opset_expr(self, node: ast.Attribute) -> values.Opset:
        """Return an Opset"""
        if isinstance(node, ast.Name):
            val = self._lookup(node.id, self._source_of(node), raise_exception=False)
            if isinstance(val, values.Opset):
                return val
            self.fail(node, f"'{node.id}' is not an instance of type Opset but {type(val)}.")
        elif isinstance(node, ast.Attribute):
            self.fail(node, "Nested module unimplemented.")  # TODO
        else:
            self.fail(node, "Invalid opset expression.")

    # pylint: enable=inconsistent-return-statements
    def _translate_callee_expr(self, node: ast.AST) -> values.Op:  # pylint: disable=R1710
        """Return an Op"""
        if isinstance(node, ast.Attribute):
            module = self._translate_opset_expr(node.value)
            self._set_default_opset(module, node)
            opname = node.attr
            if opname in module:
                return values.Op(module, node.attr)
            warn(f"'{opname}' is not a known op in '{module}'")
            return values.Op(module, node.attr)
        if isinstance(node, ast.Name):
            function_name = node.id
            found = self._lookup(function_name, self._source_of(node), raise_exception=False)
            if isinstance(found, onnxscript.OnnxFunction):
                self._current_fn.add_called_function(found)
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

    def _translate_stmt(self, node: ast.stmt, index_of_stmt=None) -> None:
        """Statement translation: A single Python statement is mapped into a
        sequence of IR statements.
        """
        if isinstance(node, ast.Assign):
            return self._translate_assign_stmt(node)
        if isinstance(node, ast.AnnAssign):
            return self._translate_assign_stmt(node)
        if isinstance(node, ast.Return):
            if index_of_stmt is not None:
                return self._translate_return_stmt(node)
            raise ValueError(
                self._message(
                    node, "Return statements are not permitted inside control-flow statements."
                )
            )
        if isinstance(node, ast.If):
            return self._translate_if_stmt(node)
        if isinstance(node, (ast.For, ast.While)):
            return self._translate_loop_stmt(node)
        if ast_utils.is_doc_string(node):
            if index_of_stmt == 0:
                return self._translate_docstring(node)
            return None
        if isinstance(node, ast.FunctionDef):
            return self._translate_nested_function_def(node)
        if ast_utils.is_print_call(node):
            return None
        raise ValueError(self._message(node, f"Unsupported statement type '{type(node)!r}'."))

    def _translate_assign_stmt(self, stmt: Union[ast.Assign, ast.AnnAssign]) -> None:
        def assign(lhs: ast.AST, rhs: ast.AST) -> None:
            if isinstance(lhs, ast.Name):
                # Assignments of the form "x = SomeExpression"
                info = self._source_of(lhs)
                lhs = lhs.id
                t = self._translate_expr(rhs, lhs).name
                if isinstance(stmt, ast.AnnAssign):
                    typeinfo = self._eval_constant_expr(stmt.annotation)
                else:
                    typeinfo = None
                var = values.Dynamic(t, values.DynamicKind.Intermediate, info, typeinfo)
                self._bind(lhs, var)
            elif isinstance(lhs, ast.Tuple):
                # Assignments of the form "x, y, z = op.SomeOp(...)"
                if not isinstance(rhs, ast.Call):
                    self.fail(
                        rhs,
                        f"RHS must be a Call expression for unpacking, found: '{type(rhs)!r}'",
                    )
                callee, inputs, attrs = self._translate_call_expr(rhs)

                def generate_onnx_name(x: ast.AST):
                    if not isinstance(x, ast.Name):
                        self.fail(x, f"LHS must be a Name for unpacking, found: '{type(x)!r}'")
                    onnx_name = self.generate_unique_name(x.id)
                    self._bind(
                        x.id,
                        values.Dynamic(
                            onnx_name, values.DynamicKind.Intermediate, self._source_of(x)
                        ),
                    )
                    return onnx_name

                outputs = [generate_onnx_name(x) for x in lhs.elts]
                self.emit(outputs, callee, inputs, attrs)
            else:
                self.fail(lhs, f"Unsupported construct in LHS of assignment: '{type(lhs)!r}'")

        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        else:
            targets = [stmt.target]
        if len(targets) != 1:
            # Assignments of the form "x = y = SomeExpression"
            self.fail(stmt, "Multi-assignment not supported.")
        lhs = targets[0]
        rhs = stmt.value
        if isinstance(rhs, ast.Tuple):
            # Assignments of the form "... = Expression1, Expression2"
            if not isinstance(lhs, ast.Tuple):
                # Assignments of the form "single_var = Expression1, Expression2".
                # We do not support tuple-typed variables.
                self.fail(lhs, f"Left term must be a tuple not '{type(lhs)!r}'.")
            # Parallel assignments of the form "x, y = Expression1, Expression2"
            if len(lhs.elts) != len(rhs.elts):
                self.fail(
                    stmt, "Expected same number of elements on lhs and rhs of assignments."
                )
            for p, r in zip(lhs.elts, rhs.elts):
                assign(p, r)
        else:
            assign(lhs, rhs)

    def _translate_return_stmt(self, stmt: ast.Return) -> None:
        def check_num_outputs(n):
            if self.returntype is not None:
                if n != len(self.returntype):
                    raise SyntaxError(
                        self._message(
                            stmt,
                            f"Mismatch in number of return values and types. Keyword "
                            f"'return' cannot be used in a subgraph (test, loop).  "
                            f"returntype is {self.returntype!r}, num_outputs={n!r}.",
                        )
                    )

        def ret(exp, i, suffix):
            preferred_name = f"return_val{suffix}"
            return_var = self._translate_expr(exp, preferred_name).name
            val = self._lookup(return_var, self._source_of(exp), False)
            if val and val.kind == values.DynamicKind.Input:
                # In ONNX, a graph-input cannot be an output of the graph.
                # We need to insert a copy.
                return_var = self._emit_copy(return_var, preferred_name)
            for prev_output in self._current_fn.outputs:
                if prev_output.name == return_var:
                    # ONNX does not allow duplicate output names.
                    return_var = self._emit_copy(return_var, f"{return_var}_copy")
                    break
            if self.returntype is None:
                t = None
            else:
                t = self.returntype[i]
            self.ir_builder.add_output(self._current_fn, return_var, t, self._source_of(stmt))
            return return_var

        val = stmt.value
        assert val is not None, "Return statement without return-value not supported."
        if isinstance(val, ast.Tuple):
            check_num_outputs(len(val.elts))
            return [ret(exp, i, str(i)) for i, exp in enumerate(val.elts)]
        check_num_outputs(1)
        return ret(val, 0, "")

    def _translate_if_stmt(self, stmt: ast.If) -> None:
        if hasattr(stmt, "live_out"):
            live_defs = list(
                stmt.live_out.intersection(analysis.assigned_vars(stmt, self._message))
            )
        else:
            live_defs = list(analysis.assigned_vars(stmt, self._message))
        test = self._translate_expr(stmt.test, "cond").name
        lineno = self._source_of(stmt).lineno
        thenGraph, sub_fct_then = self._translate_block(
            stmt.body, f"thenGraph_{lineno}", live_defs, parent_stmt=stmt
        )
        thenAttr = self._make_onnx_attr("then_branch", thenGraph)
        elseGraph, sub_fct_else = self._translate_block(
            stmt.orelse, f"elseGraph_{lineno}", live_defs, parent_stmt=stmt
        )
        elseAttr = self._make_onnx_attr("else_branch", elseGraph)

        def rename(x):
            r = self.generate_unique_name(x)
            self._bind(
                x,
                values.Dynamic(r, values.DynamicKind.Intermediate, self._source_of(stmt)),
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

    def _translate_loop_stmt(self, loop_stmt: Union[ast.For, ast.While]) -> None:
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
            o_loop_bound = self._translate_expr(iter.args[0], "loop_bound").name
            o_cond_var = self.generate_unique_name("cond_in")
            i_cond_var = o_cond_var
            cond_while = None
            o_loop_condition = ""  # No condition for a for loop.
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
            o_loop_condition = self._translate_name_expr(test)
            # we need to go through all the instructions to see
            # which instruction defines the condition test.id
        else:
            self.fail(loop_stmt, f"Unexpected loop type {type(loop_stmt)!r}.")
        # analyze loop body
        exposed_uses = analysis.exposed_uses(loop_stmt.body, self._message)
        vars_def_in_loop = analysis.assigned_vars(loop_stmt.body, self._message)
        loop_state_vars = vars_def_in_loop.intersection(exposed_uses | loop_stmt.live_out)
        scan_outputs = set()  # TODO
        outputs = list(loop_state_vars | scan_outputs)

        # loop-condition:
        # o_loop_condition = self._emit_const(True, "true", self._source_of(loop_stmt))

        # build loop_body
        self._enter_scope("loop_body", loop_stmt)
        o_loop_var = self.generate_unique_name(p_loop_var)
        self.ir_builder.add_input(
            self._current_fn,
            o_loop_var,
            onnx_types.INT64,
            self._source_of(loop_stmt),
        )
        self._bind(
            p_loop_var,
            values.Dynamic(o_loop_var, values.DynamicKind.Loop, self._source_of(loop_stmt)),
        )

        self.ir_builder.add_input(
            self._current_fn,
            i_cond_var,
            onnx_types.BOOL,
            self._source_of(loop_stmt),
        )

        for pv in loop_state_vars:
            ov = self.generate_unique_name(pv)
            # TODO: retrieve the annotation for variable pv is any is specified.
            # typeinfo = self._eval_constant_expr(pv.annotation)
            typeinfo = None
            self.ir_builder.add_input(
                self._current_fn, ov, typeinfo, self._source_of(loop_stmt)
            )
            self._bind(
                pv,
                values.Dynamic(ov, values.DynamicKind.Loop, self._source_of(loop_stmt)),
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

                _current_scope = self._current_scope()
                if s.test.id not in _current_scope:
                    self.fail(
                        loop_stmt,
                        f"Unable to find condition variable {s.test.id!r} in known "
                        f"variables {list(_current_scope)!r}.",
                    )
                condition_name = _current_scope[s.test.id].value
                operator_name = "Not"
                continue
            self._translate_stmt(s)

        o_cond_out = self.generate_unique_name("cond_out")

        if cond_while is not None:
            # Loop while
            _current_scope = self._current_scope()
            if cond_while not in _current_scope:
                self.fail(
                    loop_stmt,
                    f"Unable to find condition variable {cond_while!r} in known "
                    f"variables {list(_current_scope)!r}.",
                )
            o_cond_var = _current_scope[cond_while].value

        self.emit(
            [o_cond_out],
            values.Op(self.default_opset, operator_name),
            [condition_name or o_cond_var],
            [],
        )

        self.ir_builder.add_output(
            self._current_fn,
            o_cond_out,
            onnx_types.BOOL,
            self._source_of(loop_stmt),
        )
        for pv in loop_state_vars:
            ov = self._py_var_to_onnx_var(pv, self._source_of(loop_stmt)).name
            if ov not in self._current_fn.assigned_names:
                # When converting the loop-body into a graph, we need to handle
                # identity assignments of the form "x = y" inside the loop body
                # specially if y represents a value computed outside the loop body.
                # In this case, we create a copy of y, treating the statement as
                # shorthand for "x = op.Identity(y)".
                ov = self._emit_copy(ov, pv)
            # TODO: retrieve variable type for the annotation if any.
            typeinfo = None
            self.ir_builder.add_output(
                self._current_fn, ov, typeinfo, self._source_of(loop_stmt)
            )
        body = self._exit_scope()
        inputs = [o_loop_bound, o_loop_condition] + [
            self._py_var_to_onnx_var(pv, self._source_of(loop_stmt)).name
            for pv in loop_state_vars
        ]
        graph, sub_functions = body.to_graph_and_functions()
        attrs = [self._make_onnx_attr("body", graph)]
        info = self._source_of(loop_stmt)

        def rename(x):
            r = self.generate_unique_name(x)
            self._bind(x, values.Dynamic(r, values.DynamicKind.Output, info))
            return r

        onnx_outputs = [rename(x) for x in outputs]
        self.emit(
            onnx_outputs,
            "Loop",
            inputs,
            attrs,
            sub_functions=sub_functions,
        )

    def _translate_block(
        self,
        stmts: Sequence[ast.stmt],
        name: str,
        live_defs: Sequence[str],
        parent_stmt: ast.stmt,
    ):
        """Translation of a statement-block to GraphProto attribute."""
        info_stmt = stmts[0] if len(stmts) > 0 else parent_stmt
        source = self._source_of(info_stmt)
        self._enter_scope(name, None)
        for s in stmts:
            self._translate_stmt(s)
        for pvar in live_defs:
            if pvar in self._current_scope():
                pv_val = self._current_scope()[pvar]
                output = self._to_onnx_var(pv_val, pvar).name
                if output not in self._current_fn.assigned_names:
                    # To return an outer-scope variable, an ONNX Graph has to
                    # use an explicit copy via Identity.
                    output = self._emit_copy(output, pvar)
                self.ir_builder.add_output(
                    self._current_fn,
                    output,
                    pv_val.typeinfo,
                    source,
                )
            else:
                pv_val = None
                for scope in self._locals:  # TODO: skip _current_scope
                    if pvar in scope:
                        pv_val = scope[pvar]
                        break
                if pv_val is None:
                    self.fail(
                        stmts[0],
                        f"Variable {pvar} is not assigned a value along a conditional "
                        f"branch, known variables: {list(self._locals)}.",
                    )
                # introduce a copy
                ovar = self._emit_copy(self._to_onnx_var(pv_val, pvar).name, pvar)

                # TODO: retrieve the annotation if any.
                typeinfo = None
                self.ir_builder.add_output(self._current_fn, ovar, typeinfo, source)
        graph = self._exit_scope()
        return graph.to_graph_and_functions()

    def _translate_nested_function_def(self, fn: ast.FunctionDef) -> None:
        """Translate a nested function definition."""
        self._enter_scope(fn.name, fn)
        self._translate_function_def_common(fn)
        function_ir = self._exit_scope()
        outer_scope_vars = analysis.outer_scope_variables(fn, self._message)
        function_ir.outer_scope_variables = [
            (var, self._lookup(var, self._source_of(fn))) for var in outer_scope_vars
        ]
        self._bind(fn.name, function_ir)
        # TODO: Does not yet handle nested functions within nested functions.
        self._current_fn.add_nested_function(function_ir)

    def _translate_function_signature_common(
        self, fn: ast.FunctionDef
    ) -> irbuilder.IRFunction:
        """Translate a function signature (top-level or nested)."""
        args = fn.args
        if args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg:
            warn(f"{fn.name}: Unsupported feature in function signature.")
        for i, x in enumerate(args.args):
            arg_with_default_start_index = len(args.args) - len(args.defaults)
            if args.defaults and i >= arg_with_default_start_index:
                default_value = self._eval_constant_expr(
                    args.defaults[i - arg_with_default_start_index]
                )
            else:
                default_value = None
            if x.annotation:
                typeinfo = self._eval_constant_expr(x.annotation)
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
                    self._current_fn,
                    x.arg,
                    ta.pytype_to_attrtype(typeinfo),
                    default_value,
                )
                self._bind(x.arg, values.AttrRef(x.arg, typeinfo, self._source_of(x)))
            else:
                self.ir_builder.add_input(
                    self._current_fn, x.arg, typeinfo, self._source_of(x)
                )
                self._used_vars.add(x.arg)
                self._bind(
                    x.arg,
                    values.Dynamic(x.arg, values.DynamicKind.Input, self._source_of(x)),
                )
        if fn.returns:
            type_annotation = self._eval_constant_expr(fn.returns)
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

        return self._current_fn

    def _translate_function_def_common(self, fn: ast.FunctionDef) -> irbuilder.IRFunction:
        """Translate a function definition, including the signature and its body."""
        logger.debug("Converter:_translate_function_def_common:%s", fn.name)
        _ = self._translate_function_signature_common(fn)
        for i, s in enumerate(fn.body):
            self._translate_stmt(s, index_of_stmt=i)
        return self._current_fn

    def translate_function_def(self, stmt: ast.FunctionDef) -> irbuilder.IRFunction:
        if isinstance(stmt, ast.FunctionDef):
            self._init_function_translation()
            if self.default_opset_ is None:
                opset = self._find_onnx_opset(stmt)
                if opset:
                    self._set_default_opset(opset, stmt)
            domain = self.this_module.domain
            self._current_fn = self.ir_builder.new_function(stmt.name, domain, True)
            analysis.do_liveness_analysis(stmt, self._message)
            fn_ir = self._translate_function_def_common(stmt)
            fn_ir.debug_print()
            self.this_module.add_function_def(fn_ir)
            return fn_ir
        raise ValueError(f"Unsupported top-level statement type {type(stmt)!r}.")

    def translate_function_signature(self, fn: ast.FunctionDef) -> irbuilder.IRFunction:
        """Translate a (top-level) function signature."""
        domain = self.this_module.domain
        self._current_fn = self.ir_builder.new_function(fn.name, domain, True)
        return self._translate_function_signature_common(fn)
