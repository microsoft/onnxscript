from types import ModuleType
import ast
import inspect
from .converter import Converter
import onnx.helper
from . import values

def script_check(f: ast.FunctionDef, globalvars):
    '''
    Check that a function falls into the ONNXScript subset of Python.
    '''
    # See if conversion succeeds.
    # TODO: cleanup Converter interface/API, separating checker from
    # converter
    converter = Converter(global_names=globalvars)
    return converter.top_level_stmt(f)


def script_decorator(is_model=False):
    def transform(f):
        if inspect.isfunction(f):
            src = inspect.getsource(f)
            module = inspect.getmodule(f)
            top_level_ast = ast.parse(src)
            assert type(top_level_ast) == ast.Module
            assert len(top_level_ast.body) == 1
            f_ast = top_level_ast.body[0]
            assert type(f_ast) == ast.FunctionDef
            result = script_check(f_ast, module.__dict__.copy())
            # For now, we simply store the result of conversion as an attribute.
            # TODO: we should produce a new type of function-like object instead.
            f.function_ir = result
            # TODO: add transformations.
            return f
        raise TypeError("The ONNXScript decorator should be applied to functions only.")
    return transform


def is_converted_fun(f):
    '''
    Return True if f is a function converted by onnx-script decorator.
    A simple temporary check for now. Ideally, we should use our own type
    for such functions, and this will become 'isinstance(f, ScriptFunction)'
    '''
    return inspect.isfunction(f) and hasattr(f, "function_ir")


func = script_decorator(is_model=False)

model = script_decorator(is_model=True)


def export_onnx_lib(module: ModuleType, filename: str) -> None:
    funs = set([v for k, v in module.__dict__.items() if is_converted_fun(v)])

    # Since we don't yet have LibProto defined, we use a ModelProto as a temporary
    # container for the list of functions exported as a library, with an empty graph
    # and dummy opset_imports.
    model = onnx.helper.make_model(
        onnx.GraphProto(),
        functions=[f.function_ir.to_function_proto(values.Opset(f.function_ir.domain, 1)) for f in funs],
        producer_name='p2o',
        opset_imports=[onnx.helper.make_opsetid("", 15)])

    onnx.save(model, filename)
