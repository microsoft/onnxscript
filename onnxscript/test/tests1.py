import onnx
from onnxscript.converter import convert

import onnxscript.onnx.opset15 as op

def isomorphic(f1: onnx.FunctionProto, f2: onnx.FunctionProto):
    ...


def check_same_translated_function(fun1, fun2):
    result1 = convert(fun1)
    result2 = convert(fun2)
    # In normal use, expect one translated function from each.
    # But support a more general case to compare multiple functions from each
    assert len(result1) == len(result2)
    for (f1, f2) in zip(result1, result2):
        assert isomorphic(f1.toFunctionProto(), f2.toFunctionProto())


def test_nested_expression():
    def fun1(A, B, C):
        return (A+B)*C

    def fun2(A, B, C):
        t1 = op.Add(A, B)
        t2 = op.Mul(t1, C)
        return t2
    check_same_translated_function(fun1, fun2)
