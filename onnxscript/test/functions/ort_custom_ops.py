from onnxscript import script
from onnxscript.onnx import opset16 as op
from onnxscript.onnx_types import FLOAT, INT64
import math

M_SQRT1_2 = math.sqrt(0.5)


@script()
def Gelu(X):
    phiX = 0.5 * (op.Erf(M_SQRT1_2 * X) + 1.0)
    return X * phiX


M_2_SQRTPI = 2.0 / math.sqrt(math.pi)
alpha = M_2_SQRTPI * M_SQRT1_2 * 0.5


@script()
def GeluGrad(dY, X):
    phiX = 0.5 * (op.Erf(M_SQRT1_2 * X) + 1.0)
    XGradPhiX = alpha * X * op.Exp(-0.5 * X * X)
    grad = phiX + XGradPhiX
    dX = dY * grad
    return dX


kAlpha = M_2_SQRTPI * M_SQRT1_2
kGamma = 0.044715
kBeta = kGamma * kAlpha * 3.0


@script()
def FastGeluGrad(dY, X):
    XCube = X * X * X
    tanh = op.Tanh(kAlpha * (X + kGamma * XCube))
    sech_square = 1.0 - tanh * tanh
    sum = alpha * X + kBeta * XCube
    grad = 0.5 * (tanh + sech_square * sum + 1.0)
    return dY * grad


@script()
def SigmoidGrad(dY, Y):
    dX = dY * Y * (1.0 - Y)
    return dX


@script()
def TanhGrad(dY, Y):
    return dY * (1.0 - Y*Y)
