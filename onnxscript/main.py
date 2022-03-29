import ast
import inspect
from .converter import Converter


def script_check(f: ast.FunctionDef, globalvars):
    '''
    Check that a function falls into the ONNXScript subset of Python.
    '''
    # See if conversion succeeds.
    # TODO: cleanup Converter interface/API, separating checker from
    # converter
    converter = Converter(global_names=globalvars)
    converter.top_level_stmt(f)
    return True


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
            script_check(f_ast, module.__dict__)
            # TODO: add transformations.
            return f
        raise RuntimeError("Expecting a function.")
    return transform


func = script_decorator(is_model=False)

model = script_decorator(is_model=True)
