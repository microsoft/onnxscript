
def float_attr_ref_test (X, alpha: float) :
    return onnx.Foo(X, alpha)

def int_attr_ref_test (X, alpha: int) :
    return onnx.Foo(X, alpha)

def str_attr_ref_test (X, alpha: str) :
    return onnx.Foo(X, alpha)