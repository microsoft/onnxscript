# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import ast
import logging
import onnx
import onnx.helper as helper
from . import onnx_types as types
from .irbuilder import IRBuilder
from . import analysis as analysis
from . import type_annotation as ta
from . import values as values
from .onnx import opset15 as default_opset
from .values import (
    ConstValue, AttrRef, Dynamic, OnnxFunction, Op, DynamicKind,
    DebugInfo, CustomOpset)


logger = logging.getLogger("onnx-script")


# Python-to-IR converter:


def not_allowed(construct):
    return construct + "not supported."


class TranslationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def warn(msg):
    logger.warning(msg)


def fail(msg):
    raise TranslationError(msg)


def fail_if(cond, msg):
    if cond:
        raise TranslationError(msg)


def ignore(cond, msg):
    if cond:
        warn(msg)

# Utility to convert a python value to TensorProto:


def pyvalue_to_tensor(tensor_name: str, pyvalue):
    if isinstance(pyvalue, bool):
        return helper.make_tensor(tensor_name, onnx.TensorProto.BOOL, [], [int(pyvalue)])
    if isinstance(pyvalue, int):
        return helper.make_tensor(tensor_name, onnx.TensorProto.INT64, [], [pyvalue])
    if isinstance(pyvalue, float):
        return helper.make_tensor(tensor_name, onnx.TensorProto.FLOAT, [], [pyvalue])
    # TODO: str, sequences of values
    fail("Unimplemented")


# map from python operators to ONNX ops
primop_map = {
    ast.Add: "Add",
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
    ast.NotEq: 'NotEqual',
    ast.Pow: "Pow",
    ast.Sub: "Sub",
    ast.USub: "Neg",
}


def _known_modules():
    import onnxscript
    import onnxscript.onnx_types
    import onnxscript.onnx
    return {
        'onnx': onnx,
        'onnx.helper': onnx.helper,
        'onnxscript': onnxscript,
        'onnxscript.onnx': onnxscript.onnx,
        'onnxscript.onnx_types': onnxscript.onnx_types,
        'onnxscript.onnx.opset15': onnxscript.onnx.opset15
    }


class Converter:
    """
    Main class to translate python code into ONNX operators.

    :param ir_builder: convert AST node into ONNX structures,
        if None, class :class:`onnxscript.irbuilder.IRBuilder` is used

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

    def __init__(self, ir_builder=None, opset=None, global_names=None):
        self.ir_builder = ir_builder or IRBuilder()
        self.known_modules = _known_modules()
        if (global_names is None):
            # TODO: Cleanup: This should be eventually removed.
            self.globals = {"int": int, "float": float,
                            "str": str, "oxs": default_opset,
                            "msdomain": values.msdomain1}
        else:
            self.globals = global_names
        self.pure_modules = ["onnxscript"]
        self.default_type = types.FLOAT[...]
        self.this_module = opset or CustomOpset('this', 1)

    def init_function_translation(self):
        """Initialize self for translating a new function."""
        self.outer = []
        self.current_fn = None
        self.nextvar = 0
        self.used_vars = set()
        self.locals = [{}]

    def enter_scope(self, name, parent_node):
        self.outer.insert(0, self.current_fn)
        self.current_fn = self.ir_builder.new_function(name)
        self.locals.insert(0, {})
        logger.debug("Converter:enter_scope:%d:node:%s", len(self.locals), type(parent_node))

    def exit_scope(self):
        logger.debug("Converter:exit_scope:%d", len(self.locals))
        graph = self.current_fn
        self.current_fn = self.outer[0]
        self.outer.pop(0)
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

    def generate_unique_name(self, candidate="tmp"):
        r = candidate
        while r in self.used_vars:
            r = candidate + "_" + str(self.nextvar)
            self.nextvar = self.nextvar + 1
        self.used_vars.add(r)
        return r

    def to_onnx_attr_ref(self, val: AttrRef):
        pytype = val.typeinfo
        attrname = "value_float" if (pytype is float) else (
            "value_int" if (pytype is int) else "value_string")
        return self.ir_builder.attr_ref(attrname, val.value, pytype)

    def to_onnx_var(self, val, target=None):
        if isinstance(val, AttrRef):
            # promote attribute to value
            result = self.generate_unique_name(target if target else "tmp")
            attr = self.to_onnx_attr_ref(val)
            self.emit([result], Op(default_opset, "Constant"), [], [attr])
            return result
        if isinstance(val, ConstValue) and isinstance(val.value, float):  # TODO
            result = self.generate_unique_name(target if target else "tmp")
            return self.emit_const(val.value, result)
        if isinstance(val, Dynamic):
            return val.value
        fail("Cannot convert to onnx variable")

    def py_var_to_onnx_var(self, py_var, info):
        return self.to_onnx_var(self.lookup(py_var, info))

    def emit_docstring(self, docstring):
        self.ir_builder.add_docstring(self.current_fn, docstring)

    def emit(self, outputs, callee, inputs, attrs):
        if callee.opname == 'NotEqual':
            if len(attrs) != 0:
                raise RuntimeError(
                    "Operator %r does not support attributes." % callee.opname)
            tmp = self.generate_unique_name()
            self.ir_builder.add_stmt(
                self.current_fn, [tmp], callee.opset, "Equal", inputs, attrs)
            self.ir_builder.add_stmt(
                self.current_fn, outputs, callee.opset, "Not", [tmp], attrs)
        else:
            self.ir_builder.add_stmt(
                self.current_fn, outputs, callee.opset,
                callee.opname, inputs, attrs)

    def emit_loop(self, outputs, callee, inputs, attrs, info):
        def rename(x):
            r = self.generate_unique_name(x)
            self.bind(x, Dynamic(r, DynamicKind.Output, info))
            return r

        # [ self.to_onnx_var(self.lookup(pvar)) for pvar in inputs ]
        onnx_inputs = inputs
        onnx_outputs = [rename(x) for x in outputs]
        self.emit(onnx_outputs, Op(default_opset, callee), onnx_inputs, attrs)

    def emit_const(self, pyvalue, suggested_name):
        ovar = self.generate_unique_name(suggested_name)
        tensor = pyvalue_to_tensor(ovar, pyvalue)
        attr = self.ir_builder.attr("value", tensor)
        self.emit([ovar], Op(default_opset, "Constant"), [], [attr])
        return ovar

    def is_pure_module(self, m):
        return (m in self.pure_modules)

    def is_constant_expr(self, node):
        if isinstance(node, ast.Name):
            val = self.lookup(node.id, DebugInfo(node), raise_exception=False)
            if val is None:
                # A function...
                return False
            return isinstance(val, ConstValue) and self.is_pure_module(val.value)
        if isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Compare,
                             ast.Num, ast.Str, ast.Attribute)):
            return all([self.is_constant_expr(c) for c in ast.iter_child_nodes(node)])
        return False

    def eval_constant_expr(self, node):
        # TODO: assert (self.is_constant_expr(node))
        locals = {}  # TODO
        return (eval(compile(ast.Expression(node), filename="<ast>", mode="eval"),
                self.globals, locals))

    def eval_attr(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.NameConstant):
            if not isinstance(node.value, bool):
                raise ValueError(f"Unsupported NameConstant attribute: {node.value}.")
            return 1 if node.value else 0
        if isinstance(node, ast.List):
            return [self.eval_attr(x) for x in node.elts]
        if isinstance(node, (ast.Call, ast.Attribute, ast.UnaryOp)):
            return self.eval_constant_expr(node)
        raise ValueError(f"Unsupported attribute type: {type(node).__name__}.")

    def translate_attr(self, attr_name, node):
        if isinstance(node, ast.Name):
            val = self.lookup(node.id, DebugInfo(node))
            if (isinstance(val, AttrRef)):
                return self.to_onnx_attr_ref(val)
            else:
                # TODO: lookup value; if func.def., compile it to Graph; if a
                # constant; etc.
                fail("Unimplemented attribute construct")
        return self.ir_builder.attr(attr_name, self.eval_attr(node))

    def translate_docstring(self, node):
        if hasattr(node.value, 'value'):
            # python 3.8+
            return self.emit_docstring(node.value.value)
        if hasattr(node.value, 's'):
            # python 3.7
            return self.emit_docstring(node.value.s)
        raise TypeError("Unexpected type %r for node. "
                        "Unsupoorted version of python." % type(node))

    # Expression-translation generates "IR statements/nodes" that compute the value of
    # the expression into a target-variable, and returns the variable that is
    # assigned this value.
    def translate_expr(self, node, target="tmp"):
        if isinstance(node, ast.Call):
            r = self.translate_call_expr(node)
        elif isinstance(node, ast.BinOp):
            r = self.translate_bin_op_expr(node)
        elif isinstance(node, ast.UnaryOp):
            r = self.translate_unary_op_expr(node)
        elif isinstance(node, ast.Compare):
            r = self.translate_compare_expr(node)
        elif isinstance(node, ast.Name):
            r = self.translate_name_expr(node)
        elif isinstance(node, ast.Num):
            r = self.emit_const(node.n, target)
        elif isinstance(node, ast.NameConstant):
            r = self.emit_const(node.value, target)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}.")
        if isinstance(r, tuple):
            if isinstance(target, str):
                result = self.generate_unique_name(target)
                callee, args, attrs = r
                self.emit([result], callee, args, attrs)
                return result
            assert isinstance(target, list)
            results = [self.generate_unique_name(x) for x in target]
            callee, args, attrs = r
            self.emit(results, callee, args, attrs)
            return results
        return r

    def translate_call_expr(self, node):
        # TODO: for now, we map named arguments to attributes, and positional
        # arguments to inputs.
        callee = self.translate_callee_expr(node.func)
        args = [self.translate_expr(x) for x in node.args]
        attrs = [self.translate_attr(x.arg, x.value) for x in node.keywords]
        return callee, args, attrs

    def translate_bin_op_expr(self, node):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(DebugInfo(node).msg("Unsupported operator %r." % op))
        opname = primop_map[op]
        left = self.translate_expr(node.left)
        right = self.translate_expr(node.right)
        return Op(default_opset, opname), [left, right], []

    def translate_unary_op_expr(self, node):
        op = type(node.op)
        if op not in primop_map:
            raise ValueError(DebugInfo(node).msg("Unsupported operator %r." % op))
        opname = primop_map[op]
        operand = self.translate_expr(node.operand)
        return Op(default_opset, opname), [operand], []

    def translate_compare_expr(self, node):
        # TODO: handle multiple comparisons in one expression
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        op = type(node.ops[0])
        if op not in primop_map:
            raise ValueError(DebugInfo(node).msg("Unsupported operator %r." % op))
        opname = primop_map[op]
        left = self.translate_expr(node.left)
        right = self.translate_expr(node.comparators[0])
        return Op(default_opset, opname), [left, right], []

    def translate_name_expr(self, node):
        return self.py_var_to_onnx_var(node.id, DebugInfo(node))

    def translate_opset_expr(self, node) -> values.Opset:
        """Return an Opset"""
        if isinstance(node, ast.Name):
            try:
                val = self.lookup(node.id, DebugInfo(node))
                if isinstance(val, ConstValue):  # TODO
                    val = val.value
                if isinstance(val, values.Opset):
                    return val
                fail(f"{node.id} has value of type {type(node.id)} and used as opset.")
            except BaseException:
                warn(f"Unknown opset name {node.id}.")
                return values.Opset(node.id, 1)
        elif isinstance(node, ast.Attribute):
            fail("Nested module unimplemented")  # TODO
        else:
            fail("Invalid opset expression.")

    def translate_callee_expr(self, node) -> values.Op:
        """Return an Op"""
        if isinstance(node, ast.Attribute):
            module = self.translate_opset_expr(node.value)
            opname = node.attr
            if opname in module:
                return Op(module, node.attr)
            warn(f"'{opname}' is not a known op in '{str(module)}'")
            return Op(module, node.attr)
        if isinstance(node, ast.Name):
            function_name = node.id
            found = self.lookup(function_name, DebugInfo(node), raise_exception=False)
            if isinstance(found, OnnxFunction):
                self.current_fn.append_function(found)
                return found
            if isinstance(found, Op):
                return found
            if not found:
                default_opset = values.opset15
                if function_name not in default_opset:
                    warn(f"Unknown function name {node.id}. The ONNX graph may not work.")
                return Op(default_opset, node.id)
        fail("Invalid callee")

    # Statement translation: A single Python statement is mapped into a
    # sequence of IR statements.

    def translate_stmt(self, node, index_of_stmt=None):
        if isinstance(node, ast.Assign):
            return self.translate_assign_stmt(node)
        if isinstance(node, ast.Return):
            return self.translate_return_stmt(node)
        if isinstance(node, ast.If):
            return self.translate_if_stmt(node)
        if isinstance(node, ast.For):
            return self.translate_for_stmt(node)
        if isinstance(node, ast.Expr):
            if index_of_stmt == 0 and hasattr(node, 'value'):
                if hasattr(node.value, 'value') and isinstance(node.value.value, str):
                    # python 3.8+
                    return self.translate_docstring(node)
                if hasattr(node.value, 's') and isinstance(node.value.s, str):
                    # python 3.7
                    return self.translate_docstring(node)
        raise ValueError(DebugInfo(node).msg(
            f"Unsupported statement type: {type(node).__name__}."))

    def translate_assign_stmt(self, stmt: ast.Assign):
        def assign(lhs, rhs):
            info = DebugInfo(lhs)
            if isinstance(lhs, ast.Name):
                lhs = lhs.id
                if self.is_constant_expr(rhs):
                    self.bind(lhs, ConstValue(self.eval_constant_expr(rhs), info))
                else:
                    t = self.translate_expr(rhs, lhs)
                    self.bind(lhs, Dynamic(t, DynamicKind.Intermediate, info))
            elif isinstance(lhs, ast.Tuple):
                def id(x):
                    assert isinstance(x, ast.Name)
                    return x.id
                ids = [id(x) for x in lhs.elts]
                onnxids = self.translate_expr(rhs, ids)
                for x, y in zip(ids, onnxids):
                    self.bind(x, Dynamic(y, DynamicKind.Intermediate))
            else:
                fail("Unsupported construct in LHS of assignment.")

        assert len(stmt.targets) == 1, "Multi-assignment not supported."
        lhs = stmt.targets[0]
        rhs = stmt.value
        if isinstance(rhs, ast.Tuple):
            assert isinstance(lhs, ast.Tuple)
            assert len(lhs.elts) == len(rhs.elts), \
                "Expected same number of elements on lhs and rhs of assignments."
            for p, r in zip(lhs.elts, rhs.elts):
                assign(p, r)
        else:
            assign(lhs, rhs)

    def translate_return_stmt(self, stmt: ast.Return):
        def ret(exp, suffix=""):
            ovar = self.translate_expr(exp, "return_val" + suffix)
            # if hasattr(self, returntype) and self.num_outputs <
            # len(self.returntype):
            try:
                t = self.returntype[self.num_outputs]
            except Exception:
                t = self.default_type
            self.ir_builder.add_output(self.current_fn, ovar, t)
            self.num_outputs += 1
            return ovar

        val = stmt.value
        assert val is not None, "Return statement without return-value not supported."
        if (isinstance(val, ast.Tuple)):
            return [ret(exp, str(i)) for i, exp in enumerate(val.elts)]
        else:
            return ret(val)

    def translate_if_stmt(self, stmt: ast.If):
        live_defs = list(stmt.live_out.intersection(analysis.defs(stmt)))
        test = self.translate_expr(stmt.test, "cond")
        thenGraph = self.translate_block(stmt.body, "thenGraph", live_defs)
        thenAttr = self.ir_builder.attr("then_branch", thenGraph)
        elseGraph = self.translate_block(stmt.orelse, "elseGraph", live_defs)
        elseAttr = self.ir_builder.attr("else_branch", elseGraph)

        def rename(x):
            r = self.generate_unique_name(x)
            self.bind(x, Dynamic(r, DynamicKind.Intermediate, DebugInfo(stmt)))
            return r

        renamed = [rename(x) for x in live_defs]
        self.emit(renamed, Op(default_opset, "If"), [test], [thenAttr, elseAttr])

    def translate_for_stmt(self, for_stmt: ast.For):
        # loop-variable
        assert isinstance(for_stmt.target, ast.Name), \
            "For loop target must be a single variable."
        p_loop_var = for_stmt.target.id
        # iter
        iter = for_stmt.iter
        assert isinstance(iter, ast.Call), "Loop bound not a call."
        assert isinstance(iter.func, ast.Name), "Unsupported loop bound."
        assert iter.func.id == "range", "Unsupported loop bound."
        assert iter.args and len(iter.args) == 1, "Unsupported loop bound."
        assert not iter.keywords, "Unsupported loop bound."
        o_loop_bound = self.translate_expr(iter.args[0], "loop_bound")
        # analyze loop body
        exposed_uses = analysis.exposed_uses(for_stmt.body)
        vars_def_in_loop = analysis.defs(for_stmt.body)
        loop_state_vars = vars_def_in_loop.intersection(
            exposed_uses | for_stmt.live_out)
        scan_outputs = set()  # TODO
        outputs = list(loop_state_vars | scan_outputs)

        # loop-condition:
        o_true = self.emit_const(True, "true")
        # o_loop_bound = self.emit_const(3, "loop_bound")

        # build loop_body
        self.enter_scope("loop_body", for_stmt)
        o_loop_var = self.generate_unique_name(p_loop_var)
        self.ir_builder.add_input(self.current_fn, o_loop_var, types.INT64)
        self.bind(p_loop_var, Dynamic(o_loop_var, DynamicKind.Loop, DebugInfo(for_stmt)))
        o_cond_var = self.generate_unique_name("cond_in")
        self.ir_builder.add_input(self.current_fn, o_cond_var, types.BOOL)
        for pv in loop_state_vars:
            ov = self.generate_unique_name(pv)
            self.ir_builder.add_input(self.current_fn, ov, self.default_type)
            self.bind(pv, Dynamic(ov, DynamicKind.Loop, DebugInfo(for_stmt)))
        for s in for_stmt.body:
            self.translate_stmt(s)
        o_cond_out = self.generate_unique_name("cond_out")
        self.emit([o_cond_out], Op(default_opset, "Identity"), [o_cond_var], [])
        self.ir_builder.add_output(self.current_fn, o_cond_out, types.BOOL)
        for pv in loop_state_vars:
            ov = self.py_var_to_onnx_var(pv, DebugInfo(for_stmt))
            self.ir_builder.add_output(
                self.current_fn, ov, self.default_type)  # TODO: type
        body = self.exit_scope()

        inputs = [o_loop_bound, o_true] + \
                 [self.py_var_to_onnx_var(pv, DebugInfo(for_stmt)) for pv in loop_state_vars]
        attrs = [self.ir_builder.attr("body", body.to_graph_proto())]
        return self.emit_loop(outputs, "Loop", inputs, attrs, DebugInfo(for_stmt))

    # Translation of a statement-block to GraphProto attribute
    def translate_block(self, stmts, name, live_defs):
        self.enter_scope(name, None)
        for s in stmts:
            self.translate_stmt(s)
        for pvar in live_defs:
            if pvar in self.current_scope():
                pv_val = self.current_scope()[pvar]
                output = self.to_onnx_var(pv_val, pvar)
                self.ir_builder.add_output(
                    self.current_fn, output, self.default_type)  # TODO: need type!
            else:
                pv_val = None
                for scope in self.locals:  # TODO: skip current_scope
                    if pvar in scope:
                        pv_val = scope[pvar]
                        break
                if pv_val is None:
                    fail(f"Variable {pvar} is not assigned a value along a conditional "
                         f"branch, known variables: {list(sorted(self.locals))}.")
                # introduce a copy
                ovar = self.generate_unique_name(pvar)
                self.emit([ovar], Op(default_opset, "Identity"),
                          [self.to_onnx_var(pv_val, pvar)], [])
                # TODO: need type!
                self.ir_builder.add_output(self.current_fn, ovar, self.default_type)
        graph = self.exit_scope()
        return graph.to_graph_proto()

    def translate_function_def(self, fn: ast.FunctionDef):
        logger.debug("Converter:translate_function_def:%s", fn.name)
        if fn.name in self.this_module:
            warn(f"{fn.name}: Already defined.")
        args = fn.args
        if args.defaults:
            warn(f"{fn.name}: Default values not yet implemented.")
        if args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg:
            warn(f"{fn.name}: Unsupported feature in function signature.")
        domain = self.this_module.domain
        self.current_fn = self.ir_builder.new_function(fn.name, domain, True)
        for x in args.args:
            if x.annotation:
                typeinfo = self.eval_constant_expr(x.annotation)
            else:
                typeinfo = self.default_type
            assert ta.is_valid(typeinfo)
            if ta.is_attr(typeinfo):
                self.ir_builder.add_attr(self.current_fn, x.arg, typeinfo)
                self.bind(x.arg, AttrRef(x.arg, typeinfo, DebugInfo(x)))
            else:
                self.ir_builder.add_input(self.current_fn, x.arg, typeinfo)
                self.bind(x.arg, Dynamic(x.arg, DynamicKind.Input, DebugInfo(x)))
        if fn.returns:
            returntype = self.eval_constant_expr(fn.returns)
            if isinstance(returntype, tuple):
                assert all([ta.is_valid(t) for t in returntype])
                self.returntype = returntype
            else:
                assert ta.is_valid(returntype)
                self.returntype = (returntype,)
        else:
            self.returntype = None
        self.num_outputs = 0
        for i, s in enumerate(fn.body):
            self.translate_stmt(s, index_of_stmt=i)
        if self.returntype is not None:
            assert self.num_outputs == len(self.returntype), \
                "Mismatch in number of return values and types"
        return self.current_fn

    def do_import(self, alias):
        logger.debug("Importing %r as %r.", alias.name, alias.asname)
        fail_if(alias.name not in self.known_modules,
                f"Import: unsupported module {alias.name}")
        asname = alias.asname if alias.asname else alias.name
        self.globals[asname] = self.known_modules[alias.name]

    def top_level_stmt(self, stmt):
        if isinstance(stmt, ast.FunctionDef):
            self.init_function_translation()
            analysis.do_liveness_analysis(stmt)
            fn_ir = self.translate_function_def(stmt)
            fn_ir.debug_print()
            self.this_module[stmt.name] = fn_ir
            return fn_ir

        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                self.do_import(alias)
        elif isinstance(stmt, ast.ImportFrom):
            fail_if(stmt.module is None, "Import: module unspecified.")
            fail_if(stmt.module not in self.known_modules,
                    f"Import: unsupported module '{stmt.module}' in "
                    f"{list(sorted(self.known_modules))}")
            module = self.known_modules[stmt.module]
            for alias in stmt.names:
                asname = alias.asname if alias.asname else alias.name
                self.globals[asname] = getattr(module, alias.name)
        else:
            raise ValueError(f"Unsupported top-level statement type: {type(stmt).__name__}.")


def convert(script):
    converter = Converter()
    return converter.convert(script)
