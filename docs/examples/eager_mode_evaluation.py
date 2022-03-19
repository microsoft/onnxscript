import numpy as np
from onnxscript import eager_mode_evaluator as oxs
from onnxscript.onnx_types import FLOAT
from onnx import ModelProto, TensorProto

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

m = 2048
k = 16
n = 4096
a = np.random.rand(k, m).astype('float32').T
w = np.random.rand(n, k).astype('float32').T
b = np.random.rand(n,).astype('float32').T

print(gemmgelu(a, w, b))


def square_loss(X: FLOAT["N", 1], Y: FLOAT["N", 1]) -> FLOAT[1, 1]:
    diff = X - X
    for i in range(3):
        diff = diff + X - Y
    return oxs.ReduceSum(diff * diff, keepdims=1)

x = np.random.rand(n,).astype('float32').T
y = np.random.rand(n,).astype('float32').T
print(square_loss(x, y))

def sub_sequence_process(X: FLOAT["SubLength"]):
    a = oxs.Constant(value_float=0.5)
    Y = oxs.Add(X, a)
    return Y

# def sequance_map(X: FLOAT["Length"], sub_process):
#     model_input_length = 32
#     input_length = len(X)
#     # sequence_output = oxs.ConstantOfShape(oxs.Shape(X), value=np.zeros([1,]))
#     # sequence_output = oxs.ConstantOfShape(oxs.Shape(X), value=[1,])
#     # sequence_output = oxs.ConstantOfShape(oxs.Shape(X), value=0)
#     sequence_output = np.array([])
#     for sub_sequence_count in range((int)(input_length / model_input_length)):
#         start = oxs.Cast(np.array([sub_sequence_count * model_input_length]), to=TensorProto.INT32)
#         end = oxs.Cast(np.array([sub_sequence_count * model_input_length + model_input_length - 1]), to=TensorProto.INT32)
#         axis = 0
#         step = 1
#         sub_input = oxs.Slice(X, start, end, np.array([axis]), np.array([step]))
#         sequence_output = oxs. = sub_process(sub_input)
#     return sequence_output

# print(sequance_map(x, sub_sequence_process))