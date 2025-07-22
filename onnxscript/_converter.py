# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python-to-IR converter"""

from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import onnx
import onnx_ir as ir
from onnxscript.ir import _schemas

import onnxscript
from onnxscript import irbuilder, onnx_types, sourceinfo, values
from onnxscript import type_annotation as ta
from onnxscript._internal import _analysis, ast_utils, autocast, param_manipulation

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


logger = logging.getLogger(__name__)


def not_allowed(construct):
    return f"{construct}not supported."


class TranslationError(RuntimeError):
    pass


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
_PRIMOP_MAP = {
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


@dataclasses.dataclass
class ASTMeta:
    """Metadata for an AST node.

    This class is used to store metadata about an AST node.
    """

    # For liveness analysis,
    live_out: set[str] | None = None
    live_in: set[str] | None = None


class Converter:
    """Main class to translate python code into ONNX operators.

    The converter translates a Python function into an ONNX function by
    traversing the Python AST of the function and generating ONNX nodes
    that represent the operations in the Python code.

    ..tip::

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
        root: ast.FunctionDef,
        *,
        opset: Optional[values.Opset] = None,
        global_names: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
        default_opset: Optional[values.Opset] = None,
    ):
        """Initialize the converter.

        Args:
            root: The root AST node of the function to be converted.
            opset: The ONNX opset to use for the conversion. If None, the default opset is used.
            global_names: A dictionary of global names available in the script.
            source: Optional source code string for error reporting.
            default_opset: The default ONNX opset to use if no ONNX opset is specified in the script.
        """
        if not isinstance(root, ast.FunctionDef):
            raise TypeError(f"Converter expects an AST FunctionDef node, got {type(root)}.")
        self._ast_root = root
        self._opset = opset

        if global_names is not None:
            # We make a copy in case function eval modifies it.
            self._globals = global_names.copy()
        else:
            self._globals = {}

        self._source = source
        self._default_opset = default_opset or _find_onnx_opset(root, self._globals)
        if self._default_opset is None:
            raise ValueError(
                "default_opset must be specified in script for functions "
                "that do not contain any use of an ONNX opset."
            )

        # TODO(justinchuby): Update ir version to be user defined
        # TODO(justinchuby): Maybe just store a list of functions
        self._model = ir.Model(ir.Graph((), (), nodes=()), ir_version=10)

        # A stack of functions in the outer scope
        self._outer: list[ir.Function] = []
        self._current_fn: ir.Function = ir.Function(
            domain=self._opset.domain,
            name="",
            graph=ir.Graph((), (), nodes=[]),
            attributes={},
        )
        self._nextvar: int = 0
        self._used_vars: set[str] = set()
        self._locals: list[dict[str, LocalSymValue]] = [{}]
        self._finalized = False
        self.meta: defaultdict[ast.AST, ASTMeta] = defaultdict(ASTMeta)

    # def _init_function_translation(self) -> None:
    #     """Initialize self for translating a new (top-level) function."""
    #     self._outer = []
    #     # TODO(justinchuby): Update this
    #     self._current_fn = ir.Function(
    #         domain=self._opset.domain,
    #         name="",
    #         graph=ir.Graph((), (), nodes=[]),
    #         attributes={},
    #     )
    #     self._nextvar = 0
    #     self._used_vars = set()
    #     self._locals: List[Dict[str, LocalSymValue]] = [{}]

    def _source_of(self, node: ast.AST) -> sourceinfo.SourceInfo:
        return sourceinfo.SourceInfo(node, self._source, self._current_fn.name)

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
        self._outer.append(self._current_fn)
        assert self._opset is not None
        self._current_fn = ir.Function(
            domain=self._opset.domain,
            name=name,
            graph=ir.Graph((), (), nodes=[]),
            attributes={},
        )
        self._locals.append({})
        logger.debug("Converter:_enter_scope:%d:node:%s", len(self._locals), type(parent_node))

    def _exit_scope(self) -> ir.Function:
        """Exit from a control-flow block (a loop body or if-then-else branch)."""
        logger.debug("Converter:_exit_scope:%d", len(self._locals))
        graph = self._current_fn
        self._current_fn = self._outer.pop()
        self._locals.pop()
        assert graph is not None
        return graph

    def _current_scope(self) -> Dict[str, LocalSymValue]:
        return self._locals[-1]

    def _bind(self, name: str, val: LocalSymValue) -> None:
        logger.debug("Converter:_bind:%s", name)
        self._locals[-1][name] = val

    def _lookup(
        self, name: str, info: sourceinfo.SourceInfo, raise_exception: bool = True
    ) -> SymValue:
        for scope in reversed(self._locals):
            if name in scope:
                return scope[name]
        if name in self._globals:
            return self._globals[name]
        if raise_exception:
            raise ValueError(info.msg(f"Unbound name: {name}."))
        return None

    def _generate_unique_name(self, candidate: str = "tmp") -> str:
        # TODO(justinchuby): Can we reduce the O complexity of this function?
        r = candidate
        while r in self._used_vars:
            r = f"{candidate}_{self._nextvar}"
            self._nextvar = self._nextvar + 1
        self._used_vars.add(r)
        return r

    def _to_onnx_var(
        self,
        val: values.SymbolValue | PyValue,
        target: PreferredName = "tmp",
        *,
        info: sourceinfo.SourceInfo,
    ) -> Variable:
        """Convert a value to an ONNX variable."""
        if isinstance(val, values.AttrRef):
            # promote attribute to value
            result = self._generate_unique_name(target)
            attr = _to_onnx_ref_attr(val, info)
            self.emit([], "Constant", [result], [attr])
            if ta.base_type_is_bool(val.typeinfo):
                # ONNX attributes use an int-encoding for bools, but ONNX tensor types
                # distinguish between int and bool. So we cast the int tensor to a bool tensor,
                # to promote a (python) bool attribute to a ONNX bool tensor.
                result_as_bool = self._generate_unique_name(result + "_as_bool")
                self.emit(
                    [result], "Cast", [result_as_bool], [ir.AttrInt64("to", ir.DataType.BOOL)]
                )
                return Variable(result_as_bool, castable=True)
            return Variable(result, castable=True)

        if isinstance(val, values.Dynamic):
            return Variable(val.value)

        # Assume value is a python-value convertible to a tensor
        return self._emit_const(val, target, info)

    def _py_var_to_onnx_var(self, py_var: str, info: sourceinfo.SourceInfo) -> Variable:
        """Convert a python variable to an ONNX variable."""
        return self._to_onnx_var(self._lookup(py_var, info), target=py_var, info=info)

    def emit(
        self,
        outputs: Sequence[str],
        op_type: str,
        inputs: Sequence[str],
        attrs: Sequence[ir.Attr] = (),
        domain: str = "",
    ):
        """Emit an ONNX operator with the given inputs, outputs, and attributes."""
        node = ir.Node(
            domain=domain,
            op_type=op_type,
            inputs=[self._lookup(inp, self._source_of(inputs[0])) for inp in inputs],
            attributes=attrs,
            outputs=[self._lookup(out, self._source_of(outputs[0])) for out in outputs],
        )
        self._current_fn.append(node)

    def _emit_const(
        self,
        pyvalue: PyValue,
        suggested_name: PreferredName | None,
        info: sourceinfo.SourceInfo,
    ) -> Variable:
        """Emit a constant value as an ONNX Constant node."""
        # Obtain a name for the constant
        if suggested_name is None:
            if isinstance(pyvalue, int):
                suggested_name = f"int64_{pyvalue}"
            elif (
                isinstance(pyvalue, list) and len(pyvalue) == 1 and isinstance(pyvalue[0], int)
            ):
                suggested_name = f"int64_{pyvalue[0]}_1d"
            else:
                suggested_name = "const"
        var_name = self._generate_unique_name(suggested_name)

        # Create a tensor from the python value
        try:
            tensor = ir.tensor(pyvalue, name=var_name)
        except Exception as e:
            fail(info.msg(str(e)))

        self.emit([], "Constant", [var_name], [ir.AttrTensor("value", tensor)])
        return Variable(var_name, True)

    def _emit_copy(self, original_var: str, suggested_name: str) -> str:
        """Emits a copy statement, using the ONNX Identity operator."""
        new_var = self._generate_unique_name(suggested_name)
        self.emit([original_var], "Identity", [new_var])
        return new_var

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
        # TODO: assert (_is_constant_expr(expr))
        # TODO(justinchuby): Expand locals?
        locals: dict[Any, Any] = {}
        # TODO(justinchuby): Find a better way to pass lineno and col_offset
        expr = ast.Expression(expr, lineno=expr.lineno, col_offset=expr.col_offset)
        cpl = compile(expr, filename="<ast>", mode="eval")
        try:
            return eval(cpl, self._globals, locals)  # pylint: disable=eval-used
        except NameError as e:
            raise NameError(
                self._message(
                    expr,
                    f"Missing names, globals contains {list(self._globals)!r}, "
                    f"locals {list(locals)!r}.",
                )
            ) from e

    def _translate_attr(
        self,
        attr_name: str,
        expr: ast.AST,
        # TODO(justinchuby): Is attr_meta needed?
        attr_meta: ir.Attr | None = None,
    ) -> ir.Attr | None:
        """Translate an attribute-value specification of the form `attr_name=<expr>` in a call to an op. expr is an AST.

        The following cases are supported:
        * Expr evaluates to a script-time constant (a python-value) that can be mapped
        into an ONNX attribute value, or
        * Expr evaluates to None, in which case None is returned, or
        * Expr must be an attribute-reference, that is a name representing an
        attribute-parameter of a containing function.
        """
        if isinstance(expr, ast.Name):
            val = self._lookup(expr.id, self._source_of(expr))
            if isinstance(val, values.AttrRef):
                attr_ref = _to_onnx_ref_attr(val, val.typeinfo)
                if attr_meta is not None and (attr_ref.type != attr_meta.type):
                    self.fail(
                        expr,
                        f"Attribute type '{attr_ref.type}' does not match expected type '{attr_meta.type}'",
                    )
                return attr_ref
            if isinstance(val, ir.Graph):
                # if isinstance(val, irbuilder.IRFunction):
                # Check that outer-scope variables referenced by function have same value
                # at function-definition site and use-as-attribute site, to avoid errors.

                # TODO(justinchuby): Capture outer_scope_variables?
                # And implement the following
                # for pyvar, previous in val.outer_scope_variables:
                #     current = self._lookup(pyvar, self._source_of(expr))
                #     if current.value != previous.value:
                #         self.fail(
                #             expr,
                #             f"Outer scope variable '{pyvar}' referenced by function "
                #             f"'{expr.id!r}' modified.",
                #         )

                # Create Graph attribute
                pass
        else:
            val = self._eval_constant_expr(expr)

        # In ONNX, there is no way to explicitly specify a None value for an attribute.
        # Instead, the attribute must be omitted from the attribute list.
        # Hence, we do not create an attribute-proto if the value is None.
        # The caller is responsible for omitting such attribute-values from the list of attributes
        # in a NodeProto.
        if val is None:
            return None
        attr = ir.convenience.convert_attribute(
            attr_name, val, attr_type=attr_meta.type if attr_meta else None
        )
        return attr

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
        elif _is_constant_expr(node):
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
        result = self._generate_unique_name(target)
        self.emit([result], callee, args, attrs)
        return Variable(result)

    def _translate_opt_expr(self, node: ast.expr) -> Optional[Variable]:
        """Translation of an expression where "None" is permitted (eg., for an optional argument).
        None is represented as a Constant in Python 3.9+.
        """
        if isinstance(node, ast.Constant) and (node.value is None):
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
        target = self._generate_unique_name(target)
        indices = ast_utils.normalize_subscript_expr(node)
        info = self._source_of(node.slice)

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

            if _is_constant_expr(node_arg):
                cst = self._eval_constant_expr(node_arg)
                if isinstance(cst, int):
                    return const_1d(cst), cst
                else:
                    raise RuntimeError(f"Slice component type must be int, not {type(cst)}")
            else:
                name = self._translate_expr(node_arg).name
                reshaped = self._generate_unique_name(f"{name}_reshaped")
                self.emit(
                    [reshaped],
                    values.Op(self._default_opset, "Reshape"),
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
            elif _is_constant_expr(elt) and isinstance(self._eval_constant_expr(elt), int):
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
                start_name = self._generate_unique_name(f"{var_name}_start")
                self.emit([start_name], "Concat", starts, [axis_0_attr])

                end_name = self._generate_unique_name(f"{var_name}_end")
                self.emit([end_name], "Concat", ends, [axis_0_attr])

                axes_name = self._generate_unique_name(f"{var_name}_axis")
                self.emit([axes_name], "Concat", axes, [axis_0_attr])

                steps_name = self._generate_unique_name(f"{var_name}_step")
                self.emit([steps_name], "Concat", steps, [axis_0_attr])
            else:
                start_name = starts[0]
                end_name = ends[0]
                axes_name = axes[0]
                steps_name = steps[0]

            if squeezed_axes:
                sliced_name = self._generate_unique_name(f"{var_name}_sliced")
                self.emit(
                    [sliced_name],
                    "Slice",
                    [var_name, start_name, end_name, axes_name, steps_name],
                )
                squeezed_axes = self._emit_const(squeezed_axes, "squeezed_axes", info)

                if non_scalar_indices:  # use temporary to store result of squeeze
                    result = self._generate_unique_name(f"{var_name}_squeezed")
                else:  # store squeezed result in final target
                    result = target

                self.emit([result], "Squeeze", [sliced_name, squeezed_axes])
            else:
                if non_scalar_indices:  # use temporary to store result of Slice
                    result = self._generate_unique_name(f"{var_name}_sliced")
                else:  # store result of Slice in final target
                    result = target
                slice_inputs = [var_name, start_name, end_name, axes_name, steps_name]
                self.emit([result], "Slice", slice_inputs)
        else:
            result = var_name
        non_scalar_indices.extend(scalar_indices)
        if non_scalar_indices:
            last_axis, _ = non_scalar_indices[-1]
        else:
            # TODO(justinchuby): Clarify what last_axis should be when non_scalar_indices is False
            last_axis = None
        for axis, index_expr in non_scalar_indices:
            index_value = self._translate_expr(index_expr)
            axis_attr = self._make_onnx_attr("axis", axis)
            # use Gather to perform indexing
            # Assign gathered value to either temporary or final target
            if axis != last_axis:  # use temporary to store result of Gather
                gathered = self._generate_unique_name(f"{var_name}_axis_{axis}")
            else:  # store result of Gather in final target
                gathered = target
            self.emit([gathered], "Gather", [str(result), index_value], [axis_attr])
            result = gathered

        return Variable(result)

    def _translate_call_expr(self, node: ast.Call):
        """Translates a call-expression."""
        callee = self._translate_callee_expr(node.func)
        op_signature = callee.op_signature
        # If the callee's schema is available, we use it to determine the inputs and attributes.
        # Otherwise, we map named arguments to attributes and positional arguments to inputs.
        if op_signature is not None:
            args = node.args
            kwargs: dict[str, ast.expr] = {x.arg: x.value for x in node.keywords}
            # First separate inputs from attributes. This is needed because in Python
            # it is possible to pass onnx inputs as kwargs
            inputs, attrs = _separate_inputs_and_attrs(op_signature, args, kwargs)
            onnx_inputs = [self._translate_opt_expr(x) for x in inputs]
            attrs = [
                self._translate_attr(x, y, op_signature.params_map[x])
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
        if op not in _PRIMOP_MAP:
            raise ValueError(self._message(node, f"Unsupported operator {op!r}."))

        attr = []
        if isinstance(node.op, ast.Mod) and _is_constant_expr(node.right):
            # specific case X % f where f is a float.
            # attribute fmod=1 is added in that case.
            cst = self._eval_constant_expr(node.right)
            if isinstance(cst, float):
                attr = [self._make_onnx_attr("fmod", 1)]

        op = values.Op(self._default_opset, _PRIMOP_MAP[op])
        left, right = self._cast_like_binary_expression(
            op, self._translate_expr(node.left), self._translate_expr(node.right)
        )
        return op, [left, right], attr

    def _translate_unary_op_expr(self, node):
        op = type(node.op)
        if op not in _PRIMOP_MAP:
            raise ValueError(self._message(node, self).msg(f"Unsupported operator {op!r}."))
        if _is_constant_expr(node.operand):
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
        opname = _PRIMOP_MAP[op]
        operand = self._translate_expr(node.operand)
        return values.Op(self._default_opset, opname), [operand], []

    def _translate_compare_expr(self, node):
        # TODO: handle multiple comparisons in one expression
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        op = type(node.ops[0])
        if op not in _PRIMOP_MAP:
            raise ValueError(self._message(node, f"Unsupported operator {op!r}."))
        opname = _PRIMOP_MAP[op]
        left = self._translate_expr(node.left)
        right = self._translate_expr(node.comparators[0])

        # NotEqual is not a standard ONNX op, and needs to be translated into
        # an Equal op/node followed by a Not op/node.
        op = values.Op(self._default_opset, opname if opname != "NotEqual" else "Equal")
        left, right = self._cast_like_binary_expression(op, left, right)
        if opname == "NotEqual":
            tmp = self._generate_unique_name()
            self.emit([tmp], op, [left, right])
            not_op = values.Op(self._default_opset, "Not")
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
                if function_name not in self._default_opset:
                    warn(
                        f"Unknown function name {function_name!r}. "
                        f"The ONNX graph may not work."
                    )
                return values.Op(self._default_opset, function_name)
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
                    onnx_name = self._generate_unique_name(x.id)
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
        if (live_out := self.meta[stmt].live_out) is not None:
            live_defs = list(
                live_out.intersection(_analysis.assigned_vars(stmt, self._message))
            )
        else:
            live_defs = list(_analysis.assigned_vars(stmt, self._message))
        test = self._translate_expr(stmt.test, "cond").name
        lineno = self._source_of(stmt).lineno

        # TODO(justinchuby): Ensure the values are obtained from the live_defs
        then_graph, sub_fct_then = self._translate_block(
            stmt.body, f"then_graph_{lineno}", live_defs, parent_stmt=stmt
        )
        then_attr = ir.AttrGraph("then_branch", then_graph)
        else_graph, sub_fct_else = self._translate_block(
            stmt.orelse, f"else_graph_{lineno}", live_defs, parent_stmt=stmt
        )
        else_attr = ir.AttrGraph("else_branch", else_graph)

        def rename(x):
            r = self._generate_unique_name(x)
            self._bind(
                x,
                values.Dynamic(r, values.DynamicKind.Intermediate, self._source_of(stmt)),
            )
            return r

        # no break condition
        renamed = [rename(x) for x in live_defs]
        if not renamed:
            # TODO(justinchuby): This needs comments. What is it doing?
            self.fail(stmt, "A subgraph for an if condition has no outputs.")

        # TODO(justinchuby): Collect the subfunctions to self
        sub_functions = {}
        sub_functions.update(sub_fct_then)
        sub_functions.update(sub_fct_else)
        if renamed == [test]:
            self.fail(stmt, f"Input and output cannot be the same {renamed!r}.")
        self.emit(
            [test],
            "If",
            renamed,
            [then_attr, else_attr],
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
            o_cond_var = self._generate_unique_name("cond_in")
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
        exposed_uses = _analysis.exposed_uses(loop_stmt.body, self._message)
        vars_def_in_loop = _analysis.assigned_vars(loop_stmt.body, self._message)
        live_out = self.meta[loop_stmt].live_out or set()
        loop_state_vars = vars_def_in_loop.intersection(exposed_uses | live_out)
        scan_outputs = set()  # TODO
        outputs = list(loop_state_vars | scan_outputs)

        # loop-condition:
        # o_loop_condition = self._emit_const(True, "true", self._source_of(loop_stmt))

        # build loop_body
        self._enter_scope("loop_body", loop_stmt)
        o_loop_var = self._generate_unique_name(p_loop_var)
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
            ov = self._generate_unique_name(pv)
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

                current_scope = self._current_scope()
                if s.test.id not in current_scope:
                    self.fail(
                        loop_stmt,
                        f"Unable to find condition variable {s.test.id!r} in known "
                        f"variables {list(current_scope)!r}.",
                    )
                condition_name = current_scope[s.test.id].value
                operator_name = "Not"
                continue
            self._translate_stmt(s)

        o_cond_out = self._generate_unique_name("cond_out")

        if cond_while is not None:
            # Loop while
            current_scope = self._current_scope()
            if cond_while not in current_scope:
                self.fail(
                    loop_stmt,
                    f"Unable to find condition variable {cond_while!r} in known "
                    f"variables {list(current_scope)!r}.",
                )
            o_cond_var = current_scope[cond_while].value

        self.emit(
            [o_cond_out],
            values.Op(self._default_opset, operator_name),
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
            r = self._generate_unique_name(x)
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
                for scope in reversed(self._locals):  # TODO: skip _current_scope
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
        self._translate_function_def(fn)
        function_ir = self._exit_scope()
        outer_scope_vars = _analysis.outer_scope_variables(fn, self._message)
        function_ir.outer_scope_variables = [
            (var, self._lookup(var, self._source_of(fn))) for var in outer_scope_vars
        ]
        self._bind(fn.name, function_ir)
        # TODO: Does not yet handle nested functions within nested functions.
        self._current_fn.add_nested_function(function_ir)

    def _translate_function_signature_common(self, fn: ast.FunctionDef) -> ir.Function:
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

    def _translate_function_def(self, node: ast.FunctionDef) -> ir.Function:
        """Translate a function definition, including the signature and its body."""
        logger.debug("Converter:_translate_function_def:%s", node.name)
        _ = self._translate_function_signature_common(node)
        for i, s in enumerate(node.body):
            self._translate_stmt(s, index_of_stmt=i)

        # Update docstring if available
        if docstring := ast.get_docstring(node):
            self._current_fn.doc_string = docstring
        return self._current_fn

    def _finalize(self) -> None:
        self._finalized = True

    def convert(self) -> ir.Function:
        """Convert the Python AST to an ONNX IR function."""
        if self._finalized:
            return self._current_fn

        func_def = self._ast_root
        _analysis.do_liveness_analysis(func_def, self._message, self.meta)
        return self._translate_function_def(func_def)
        # TODO(justinchuby): Handle function registration to the opset
        # self._opset.add_function_def(fn_ir)


def _is_constant_expr(node: ast.AST) -> bool:
    """Check if the AST node is a constant expression."""
    if isinstance(node, ast.UnaryOp):
        return _is_constant_expr(node.operand)
    if isinstance(
        node,
        (
            ast.Call,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Attribute,
            ast.List,
            ast.Load,
            ast.Constant,
        ),
    ):
        return all(_is_constant_expr(c) for c in ast.iter_child_nodes(node))
    return False


def _separate_inputs_and_attrs(
    signature: _schemas.OpSignature,
    args: Sequence[ast.expr],
    kwargs: Mapping[str, ast.expr],
) -> tuple[Sequence[ast.expr], dict[str, ast.expr]]:
    """Construct two mappings: name to inputs and named to attributes based on the signature and args/kwargs.

    This function uses the OpSignature to determine which argument in args and kwargs corresponds to
    which parameter in the signature. ONNX node inputs are stored in named_inputs, and attributes are
    stored in named_attrs. If an _optional input_ is not provided, it is filled with None.

    Args:
        signature: The OpSignature for the node.
        args: The positional arguments for the node.
        kwargs: The keyword arguments for the node.

    Returns:
        A tuple of two mappings: named_inputs and named_attrs.

    Raises:
        ValueError: If a required parameter is not provided.
    """
    # 1. Construct inputs, attrs based on (args, kwargs) and the signature.
    #   a. Loop over all parameters in the signature and args together
    #   b. Depending on param.is_input, Record inputs or named_attrs[param.name] = arg
    #   c. Handle kwargs as well
    inputs_reversed: Sequence[Any] = []
    named_attrs: dict[str, Any] = {}
    reversed_args_stack = list(reversed(args))
    for param in signature.params:
        if isinstance(param, _schemas.Parameter):
            # Handle inputs
            if reversed_args_stack:
                # First exhaust the positional arguments
                if param.variadic:
                    # Handle variadic arguments
                    inputs_reversed = [*reversed(args)]
                    reversed_args_stack.clear()
                else:
                    inputs_reversed.append(reversed_args_stack.pop())
            elif param.name in kwargs:
                inputs_reversed.append(kwargs[param.name])
            elif param.required:
                raise ValueError(
                    f"Required parameter '{param.name}' is not provided. "
                    f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                )
            else:
                logger.debug(
                    "Optional parameter '%s' is not provided. Added as None. Signature: %s",
                    param.name,
                    signature,
                )
                inputs_reversed.append(None)
        else:
            # Handle attributes
            attribute: ir.Attr | None
            assert isinstance(param, _schemas.AttributeParameter), (
                f"Expected AttributeParameter, got {type(param)}"
            )
            if reversed_args_stack:
                # First exhaust the positional arguments
                attribute = reversed_args_stack.pop()  # type: ignore[assignment]
            elif kwargs.get(param.name) is not None:
                attribute = kwargs[param.name]  # type: ignore[assignment]
            else:
                if param.required:
                    raise ValueError(
                        f"Required attribute '{param.name}' is not provided. "
                        f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                    )
                else:
                    logger.debug(
                        "Optional attribute '%s' is None. Dropped. Signature: %s",
                        param.name,
                        signature,
                    )
                    continue
            named_attrs[param.name] = attribute
    return tuple(reversed(inputs_reversed)), named_attrs


def _to_onnx_ref_attr(val: values.AttrRef, info: sourceinfo.SourceInfo | None) -> ir.Attr:
    """Convert an attribute reference to an ONNX ref attribute."""

    # TODO(justinchuby): Consider using a convenience function
    pytype = val.typeinfo
    attrtype = _schemas.get_attr_type(pytype)
    attrname = None
    if attrtype is ir.AttributeType.FLOAT:
        attrname = "value_float"
    elif attrtype is ir.AttributeType.INT:
        attrname = "value_int"
    elif attrtype is ir.AttributeType.STRING:
        attrname = "value_string"
    elif attrtype is ir.AttributeType.INTS:
        attrname = "value_ints"
    else:
        msg = f"Unsupported attribute type {pytype!r}."
        fail(info.msg(msg) if info else msg)
    # TODO(justinchuby): What is the ref attr name?
    return ir.RefAttr(attrname, val.value, attrtype)


def _find_onnx_opset(node: ast.AST, globals: dict[str, Any]) -> values.Opset | None:
    """Find the (first) ONNX opset used in the function, if any."""
    # Search for a Call expression of form "op.OpName(...)"
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            opset_expr = node.func.value
            if isinstance(opset_expr, ast.Name):
                if opset_expr.id in globals:
                    opset = globals[opset_expr.id]
                    if isinstance(opset, values.Opset) and opset.domain == "":
                        return opset
    for child in ast.iter_child_nodes(node):
        res = _find_onnx_opset(child, globals)
        if res is not None:
            return res
    return None
