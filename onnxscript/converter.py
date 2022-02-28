from numpy import isin
import onnxscript
import onnxscript.types as types

import os
import inspect
import ast
from ast import *

import irbuilder
from irbuilder import IRBuilder
import analysis
import type_annotation as ta
import values
from values import Value, ConstValue, AttrRef, Dynamic, Op

import onnx
import onnx.helper as helper

print_flag = True

# Python-to-IR converter:

def not_allowed(construct):
    return construct + "not supported."

class TranslationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def warn(msg):
    print ("Warning: " + msg)

def fail(msg):
    raise TranslationError(msg)

def fail_if(cond, msg):
    if cond:
        raise TranslationError(msg)

def ignore (cond, msg):
    if cond:
        warn (msg)

def pyvalue_to_tensor(tensor_name : str, pyvalue):
    if isinstance(pyvalue, bool):
        return helper.make_tensor(tensor_name, onnx.TensorProto.BOOL, [], [int(pyvalue)])
    elif isinstance(pyvalue, int):
        return helper.make_tensor(tensor_name, onnx.TensorProto.INT64, [], [pyvalue])
    elif isinstance(pyvalue, float):
        return helper.make_tensor(tensor_name, onnx.TensorProto.FLOAT, [], [pyvalue])
    else:
        fail("Unimplemented")

# map from python operators to ONNX ops
primop_map = {
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mul",
    ast.Div: "Div",
    ast.USub: "Neg",
    ast.Lt: "Less",
    ast.Gt: "Greater",
    ast.LtE: "LessOrEqual",
    ast.GtE : "GreaterOrEqual",
    ast.MatMult : "MatMul",
    ast.Mod : "Mod",
    ast.Pow : "Pow"
}

class Converter:
    def __init__(self, ir_builder = IRBuilder()):
        self.ir_builder = ir_builder
        self.known_modules = { 'onnxscript' : onnxscript, 'onnxscript.types' : types, 'onnx.opset15' : values.opset15 }
        self.globals = { "int" : int, "float" : float, "str" : str, "onnx" : values.opset15, "Onnx" : values.opset15 } # 'os' : onnxscript
        self.pure_modules = ["onnxscript"]
        self.default_type = types.FLOAT[...]

    def initFunctionTranslation(self):
        self.outer = []
        self.current_fn = None
        self.nextvar = 0
        self.used_vars = set()
        self.locals = [{}]

    def enterScope(self, name):
        self.outer.insert(0,self.current_fn)
        self.current_fn = self.ir_builder.newFunction(name)
        self.locals.insert(0, {})
    
    def exitScope(self):
        graph = self.current_fn
        self.current_fn = self.outer[0]
        self.outer.pop(0)
        self.locals.pop(0)
        return graph
    
    def currentScope(self):
        return self.locals[0]

    def bind(self, name, val):
        self.locals[0][name] = val
    
    def lookup(self, name):
        for scope in self.locals:
            if (name in scope): return scope[name]
        if (name in self.globals): return self.globals[name]
        raise ValueError("Unbound name: " + name)

    def generateUniqueName(self, candidate = "tmp"):
        r = candidate
        while (r in self.used_vars):
            r = candidate + "_" + str(self.nextvar)
            self.nextvar = self.nextvar + 1
        self.used_vars.add(r)
        return r

    def to_onnx_attr_ref(self, val : AttrRef):
        pytype = val.type
        attrname = "value_float" if (pytype is float) else ("value_int" if (pytype is int) else "value_string")
        return self.ir_builder.attr_ref(attrname, val.value, pytype)

    def to_onnx_var(self, val, target = None):
        if (isinstance(val, AttrRef)):
            # promote attribute to value
            result = self.generateUniqueName(target if target else "tmp")
            attr = self.to_onnx_attr_ref(val)
            self.emit ([result], Op("", "Constant"), [], [attr])
            return result
        elif (isinstance(val, ConstValue) and isinstance(val.value, float)): # TODO
            result = self.generateUniqueName(target if target else "tmp")
            return self.emitConst(val.value, result)
        elif isinstance(val, Dynamic):
            return val.value
        else:
            fail(f"Cannot convert to onnx variable")

    def py_var_to_onnx_var(self, py_var): return self.to_onnx_var(self.lookup(py_var))
    
    def emit(self, outputs, callee, inputs, attrs):
        self.ir_builder.addStmt(self.current_fn, outputs, callee.opset, callee.opname, inputs, attrs)

    def emit2(self, outputs, callee, inputs, attrs):
        def rename(x):
            r = self.generateUniqueName(x)
            self.bind(x, Dynamic(r))
            return r
        onnx_inputs = inputs # [ self.to_onnx_var(self.lookup(pvar)) for pvar in inputs ]
        onnx_outputs = [ rename(x) for x in outputs ]
        self.emit(onnx_outputs, Op("", callee), onnx_inputs, attrs)

    def emitConst (self, pyvalue, suggested_name):
        ovar = self.generateUniqueName(suggested_name)
        tensor = pyvalue_to_tensor (ovar, pyvalue)
        attr = self.ir_builder.attr("value", tensor)
        self.emit([ovar], Op("", "Constant"), [], [attr])
        return ovar

    def isPureModule(self, m):
        return (m in self.pure_modules)

    def isConstantExpr (self, node):
        if (isinstance(node, ast.Name)):
            val = self.lookup(node.id)
            return isinstance(val, ConstValue) and self.isPureModule(val.value)
        if isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Num, ast.Str, ast.Attribute)):
            return all([self.isConstantExpr(c) for c in ast.iter_child_nodes(node)])
        return False

    def evalConstantExpr(self, node):
        # TODO: assert (self.isConstantExpr(node))
        locals = {} # TODO
        return eval(compile(ast.Expression(node), filename="<ast>", mode="eval"), self.globals, locals)

    def eval_attr(self, node):
        if (isinstance(node, ast.Num)):
            return node.n
        elif (isinstance(node, ast.Str)):
            return node.s
        elif (isinstance(node, ast.NameConstant)):
            if (node.value == True):
                return 1
            elif (node.value == False):
                return 0
            else:
                raise ValueError("Unsupported NameConstant attribute : " + str(node.value))
        elif (isinstance(node, ast.List)):
            return [self.eval_attr(x) for x in node.elts]
        elif (isinstance(node, (ast.Call, ast.Attribute))):
            return self.evalConstantExpr(node)
        else:
            raise ValueError("Unsupported attribute type: " + type(node).__name__)

    def translateAttr(self, attr_name, node):
        if (isinstance(node, ast.Name)):
            val = self.lookup(node.id)
            if (isinstance(val, AttrRef)):
                return self.to_onnx_attr_ref(val)
            else:
                # TODO: lookup value; if func.def., compile it to Graph; if a constant; etc.
                fail("Unimplemented attribute construct")
        return self.ir_builder.attr(attr_name, self.eval_attr(node))

    # Expression-translation generates "IR statements/nodes" that compute the value of
    # the expression into a target-variable, and returns the variable that is assigned this value.
    def translateExpr(self, node, target="tmp"):
        if (isinstance(node, ast.Call)):
            r = self.translateCall(node)
        elif (isinstance(node, ast.BinOp)):
            r = self.translateBinOp (node)
        elif (isinstance(node, ast.UnaryOp)):
            r = self.translateUnaryOp (node)
        elif (isinstance(node, ast.Compare)):
            r = self.translateCompare (node)
        elif (isinstance(node, ast.Name)):
            r = self.translateName (node)
        elif (isinstance(node, ast.Num)):
            r = self.emitConst(node.n, target)
        elif isinstance(node, ast.NameConstant):
            r = self.emitConst(node.value, target)
        # elif (isinstance(node, ast.Attribute)):
        #     r = self.translateAttribute (node)
        else:
            raise ValueError("Unsupported expression type: " + type(node).__name__)
        if (isinstance(r, tuple)):
            if isinstance(target, str):
                result = self.generateUniqueName(target)
                callee, args, attrs = r
                self.emit([result], callee, args, attrs)
                return result
            else:
                assert isinstance(target, list)
                results = [self.generateUniqueName(x) for x in target]
                callee, args, attrs = r
                self.emit(results, callee, args, attrs)
                return results              
        return r

    def translateCall(self, node):
        # TODO: for now, we map named arguments to attributes, and positional arguments to inputs.
        callee = self.translateCallee(node.func)
        args = [self.translateExpr(x) for x in node.args]
        attrs = [self.translateAttr(x.arg, x.value) for x in node.keywords]
        return (callee, args, attrs)

    def translateBinOp(self, node):
        op = type(node.op)
        assert (op in primop_map)
        opname = primop_map[op]
        left = self.translateExpr(node.left)
        right = self.translateExpr(node.right)
        return (Op("", opname), [left, right], [])

    def translateUnaryOp(self, node):
        op = type(node.op)
        assert (op in primop_map)
        opname = primop_map[op]
        operand = self.translateExpr(node.operand)
        return (Op("", opname), [operand], [])
    
    def translateCompare(self, node):
        assert (len(node.ops) == 1) # TODO: handle multiple comparisons in one expression
        assert (len(node.comparators) == 1)
        op = type(node.ops[0])
        assert (op in primop_map)
        opname = primop_map[op]
        left = self.translateExpr(node.left)
        right = self.translateExpr(node.comparators[0])        
        return (Op("", opname), [left, right], [])

    # TODO: returns???
    def translateName(self, node):
        return self.py_var_to_onnx_var(node.id)

    def translateNum(self, node):
        return self.emitConst(node.n, "Const")

    def translateModule(self, node):
        if isinstance(node, ast.Name):
            try:
                val = self.lookup(node.id)
                if isinstance(val, ConstValue): #TODO
                    val = val.value
                if isinstance(val, values.Opset):
                    return val
                fail(f"{node.id} has value of type {type(node.id)} and used as module")
            except:
                warn(f"Unknown module name {node.id}.")
                return values.Opset(node.id, 1)
        elif isinstance (node, ast.Attribute):
            fail("Nested module unimplemented")
        else:
            fail("Invalid module.")

    def translateCallee(self, node):
        if isinstance(node, ast.Attribute):
            module = self.translateModule(node.value)
            opname = node.attr
            if (opname not in module): warn (f"{opname} is not a known op in {str(module)}")
            return Op(module, node.attr)
        elif isinstance(node, ast.Name):
            try:
                val = self.lookup(node.id)
            except:
                default_opset = values.opset15
                if (node.id not in default_opset):
                    warn(f"Unknown function name {node.id}.")
                return Op(default_opset, node.id)
        else:
            fail ("Invalid callee")

    # Statement translation: A single Python statement is mapped into a sequence of IR statements.

    def translateStmt(self, node):
        if (isinstance(node, ast.Assign)):
            self.translateAssign(node)
        elif (isinstance(node, ast.Return)):
            self.translateReturn(node)
        elif (isinstance(node, ast.If)):
            self.translateIf(node)
        elif isinstance(node, ast.For):
            self.translateFor(node)
        else:
            raise ValueError("Unsupported statement type: " + type(node).__name__)

    def translateAssign(self, node: Assign):
        def assign(lhs, rhs):
            if (isinstance(lhs, ast.Name)):
                lhs = lhs.id 
                if (self.isConstantExpr(rhs)):
                    self.bind(lhs, ConstValue(self.evalConstantExpr(rhs)))
                else:
                    t = self.translateExpr(rhs, lhs)
                    self.bind(lhs, Dynamic(t))
            elif isinstance(lhs, ast.Tuple):
                def id(x):
                    assert isinstance(x, ast.Name)
                    return x.id
                ids = [id(x) for x in lhs.elts]
                onnxids = self.translateExpr(rhs, ids)
                for x, y in zip(ids, onnxids):
                    self.bind(x, Dynamic(y))
            else:
                fail("Unsupported construct in LHS of assignment.")
        assert (len(node.targets) == 1), "Multi-assignment not supported."
        lhs = node.targets[0]
        rhs = node.value
        if (isinstance(rhs, ast.Tuple)):
            assert isinstance(lhs, ast.Tuple)
            assert len(lhs.elts) == len(rhs.elts), "Expected same number of elements on lhs and rhs of assignments."
            for l,r in zip(lhs.elts, rhs.elts):
                assign(l, r)
        else:
            assign(lhs, rhs)

    def translateReturn(self, node):
        def ret(exp, suffix=""):
            ovar = self.translateExpr(exp, "return_val" + suffix)
            # if hasattr(self, returntype) and self.num_outputs < len(self.returntype):
            try:
                t = self.returntype[self.num_outputs]
            except Exception as e:
                t = self.default_type
            self.ir_builder.addOutput(self.current_fn, ovar, t)
            self.num_outputs += 1
            return ovar

        val = node.value
        assert (val != None), "Return statement without return-value not supported."
        if (isinstance(val, ast.Tuple)):
            return [ret(exp,str(i)) for i,exp in enumerate(val.elts)]
        else:
            return ret(val)

    def translateIf(self, node):
        live_defs = list(node.live_out.intersection(analysis.defs(node)))
        # print(live_defs)
        test = self.translateExpr(node.test, "cond")
        thenGraph = self.translateBlock(node.body, "thenGraph", live_defs)
        thenAttr = self.ir_builder.attr("then_branch", thenGraph)
        elseGraph = self.translateBlock(node.orelse, "elseGraph", live_defs)
        elseAttr = self.ir_builder.attr("else_branch", elseGraph)
        def rename(x):
            r = self.generateUniqueName(x)
            self.bind(x, Dynamic(r))
            return r
        renamed = [ rename(x) for x in live_defs ]
        self.emit(renamed, Op("", "If"), [test], [thenAttr, elseAttr])

    def translateFor(self, for_stmt: ast.For):
        # loop-variable
        assert isinstance(for_stmt.target, ast.Name), "For loop target must be a single variable."
        p_loop_var = for_stmt.target.id
        # iter
        iter = for_stmt.iter
        assert isinstance(iter, ast.Call), "Loop bound not a call."
        assert isinstance(iter.func, ast.Name), "Unsupported loop bound."
        assert iter.func.id == "range", "Unsupported loop bound."
        assert iter.args and len(iter.args) == 1, "Unsupported loop bound."
        assert not iter.keywords, "Unsupported loop bound."
        o_loop_bound = self.translateExpr(iter.args[0], "loop_bound")
        # analyze loop body
        exposed_uses = analysis.exposed_uses(for_stmt.body)
        vars_def_in_loop = analysis.defs(for_stmt.body)
        loop_state_vars = vars_def_in_loop.intersection(exposed_uses | for_stmt.live_out)
        scan_outputs = set() # TODO
        outputs = list(loop_state_vars | scan_outputs)

        # loop-condition:
        o_true = self.emitConst(True, "true")
        # o_loop_bound = self.emitConst(3, "loop_bound")

        # build loop_body
        self.enterScope("loop_body")
        o_loop_var = self.generateUniqueName(p_loop_var)
        self.ir_builder.addInput(self.current_fn, o_loop_var, types.INT64)
        self.bind(p_loop_var, Dynamic(o_loop_var))
        o_cond_var = self.generateUniqueName("cond_in")
        self.ir_builder.addInput(self.current_fn, o_cond_var, types.BOOL)
        for pv in loop_state_vars:
            ov = self.generateUniqueName(pv)
            self.ir_builder.addInput(self.current_fn, ov, self.default_type) 
            self.bind(pv, Dynamic(ov))            
        for s in for_stmt.body:
            self.translateStmt(s)
        o_cond_out = self.generateUniqueName("cond_out")
        self.emit([o_cond_out], Op("", "Identity"), [o_cond_var], [])
        self.ir_builder.addOutput(self.current_fn, o_cond_out, types.BOOL)
        for pv in loop_state_vars:
            ov = self.py_var_to_onnx_var(pv)
            self.ir_builder.addOutput(self.current_fn, ov, self.default_type) # TODO: type
        body = self.exitScope()
        # if (print_flag):
        #     print("Generated loop body:")
        #     body.print()

        inputs = [o_loop_bound, o_true] + [self.py_var_to_onnx_var(pv) for pv in loop_state_vars]
        attrs = [self.ir_builder.attr("body", body.toGraph())]
        self.emit2(outputs, "Loop", inputs, attrs)

    # Translation of a statement-block to GraphProto attribute
    def translateBlock(self, stmts, name, live_defs):
        self.enterScope(name)
        for s in stmts:
            self.translateStmt(s)
        for pvar in live_defs:
            if (pvar in self.currentScope()):
                pv_val = self.currentScope()[pvar]
                output = self.to_onnx_var(pv_val, pvar)
                self.ir_builder.addOutput(self.current_fn, output, self.default_type) # TODO: need type!
            else:
                pv_val = None
                for scope in self.locals: # TODO: skip currentScope
                    if (pvar in scope):
                        pv_val = scope[pvar]
                        break
                if (pv_val is None):
                    fail (f"Variable {pvar} is not assigned a value along a conditional branch.")               
                # introduce a copy
                ovar = self.generateUniqueName(pvar)
                self.emit([ovar], Op("", "Identity"), [self.to_onnx_var(pv_val, pvar)], [])
                self.ir_builder.addOutput(self.current_fn, ovar, self.default_type) # TODO: need type!
        graph = self.exitScope()
        # if print_flag:
        #     print ("Generated block")
        #     graph.print()
        return graph.toGraph()

    def convert_FunctionDef(self, node):
        args = node.args
        if (args.defaults):
            warn (f"{node.name}: Default values not yet implemented.")
        if (args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg):
            warn (f"{node.name}: Unsupported feature in function signature.")
        self.current_fn = self.ir_builder.newFunction(node.name)
        for x in args.args:
            if x.annotation:
                typeinfo = self.evalConstantExpr(x.annotation)
            else:
                typeinfo = self.default_type
            assert ta.is_valid(typeinfo)
            if (ta.is_attr(typeinfo)):
                self.ir_builder.addAttr(self.current_fn, x.arg, typeinfo)
                self.bind(x.arg, AttrRef(x.arg, typeinfo)) 
            else:
                self.ir_builder.addInput(self.current_fn, x.arg, typeinfo)
                self.bind(x.arg, Dynamic(x.arg))
        if node.returns:
            returntype = self.evalConstantExpr(node.returns)
            if isinstance(returntype, tuple):
                assert all([ta.is_valid(t) for t in returntype])
                self.returntype = returntype
            else:
                assert ta.is_valid(returntype)
                self.returntype = (returntype,)
        else:
            self.returntype = None
        self.num_outputs = 0
        for s in node.body:
            self.translateStmt(s)
        if self.returntype is not None:
            assert (self.num_outputs == len(self.returntype)), "Mismatch in number of return values and types"
        return self.current_fn

    def do_import(self, alias):
        if print_flag:
            print (f"Importing {alias.name} as {alias.asname}")
        fail_if (alias.name not in self.known_modules, f"Import: unsupported module {alias.name}")
        asname = alias.asname if alias.asname else alias.name
        self.globals[asname] = self.known_modules[alias.name]

    def top_level_stmt (self, stmt):
        if isinstance(stmt, ast.FunctionDef):
            self.initFunctionTranslation()
            analysis.do_liveness_analysis(stmt)
            fn_ir = self.convert_FunctionDef(stmt)
            if print_flag:
                print("============== OUTPUT =============")
                fn_ir.print()
                print()
            return fn_ir
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                self.do_import(alias)
        elif isinstance(stmt, ast.ImportFrom):
            fail_if(stmt.module is None, "Import: module unspecified.")
            fail_if (stmt.module not in self.known_modules, f"Import: unsupported module {stmt.module}")
            module = self.known_modules[stmt.module]
            for alias in stmt.names:
                asname = alias.asname if alias.asname else alias.name
                self.globals[asname] = getattr(module, alias.name)
        else:
            raise ValueError("Unsupported top-level statement type: " + type(stmt).__name__)

    def convert_source(self, src):
        if print_flag:
            print("============== INPUT =============")
            print(src)
            print()
        module = ast.parse(src)
        assert type(module) == ast.Module
        converted = [self.top_level_stmt(d) for d in module.body]
        return [x for x in converted if x is not None]

    def convert_file(self, filename):
        src = open(filename).read()
        return self.convert_source(src)  

    def convert(self, f):
        if (isinstance(f, str)):
            if (os.path.exists(f)):
                return self.convert_file(f)
            else:
                return self.convert_source(f)
        elif (inspect.isfunction(f)):
            src = inspect.getsource(f)
            return self.convert_source(src) 
        else:
            fail("Unknown type of input to converter.") 

def convert (script):
    converter = Converter()
    converter.convert(script)
