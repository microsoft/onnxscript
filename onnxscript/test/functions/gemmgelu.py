# flake8: noqa
from onnxscript.onnx_types import FLOAT


def gemmgelu(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:

    a = oxs.Constant(value_float=0.5)
    b = oxs.Constant(value_float=0.797885)
    c = oxs.Constant(value_float=0.035677)
    one = oxs.Constant(value_float=1.0)
    P1 = oxs.MatMul(A, W)
    X = oxs.Add(P1, Bias)
    T1 = oxs.Mul(X, X)
    T2 = oxs.Mul(c, T1)
    T3 = oxs.Add(b, T2)
    T4 = oxs.Mul(X, T3)
    T5 = oxs.Tanh(T4)
    T6 = oxs.Add(one, T5)
    T7 = oxs.Mul(X, T6)
    Y = oxs.Mul(a, T7)
    return Y
