# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Same functions as in onnxfns1.py, using autocast and default-attribute-values

from onnxscript import script
from onnxscript.onnx_opset import opset18 as op


@script()
def Relu(X):
    zero = op.CastLike(0, X)
    return op.Max(X, zero)


@script()
def Selu(
    X,
    alpha: float = 1.67326319217681884765625,
    gamma: float = 1.05070102214813232421875,
):
    neg = gamma * (alpha * op.Exp(X) - alpha)
    pos = gamma * X
    return op.Where(X <= 0.0, neg, pos)


@script()
def Elu(X, alpha: float = 1.0):
    return op.Where(X < 0.0, alpha * (op.Exp(X) - 1.0), X)


@script()
def Elu05(X):
    return op.Where(X < 0.0, 0.5 * (op.Exp(X) - 1.0), X)


@script()
def ThresholdedRelu(X, alpha: float = 1.0):
    zero = op.CastLike(0, X)
    return op.Where(X > alpha, X, zero)


@script()
def LeakyRelu(X, alpha: float = 0.01):
    return op.Where(X < 0.0, alpha * X, X)


@script()
def PRelu(X, slope):
    # future-work: capturing extra requirements such as:
    # slope must be unidirectionally broadcastable to X's shape.
    return op.Where(X < 0.0, slope * X, X)


@script()
def HardSigmoid(X, alpha: float = 0.2, beta: float = 0.5):
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    return op.Max(zero, op.Min(one, alpha * X + beta))


@script()
def Shrink(x, lambd: float = 0.5, bias: float = 0.0):
    zero = op.CastLike(0, x)
    return op.Where(x < -lambd, x + bias, op.Where(x > lambd, x - bias, zero))


@script()
def Softplus(X):
    return op.Log(op.Exp(X) + 1.0)


@script()
def Softsign(X):
    return X / (1.0 + op.Abs(X))


@script()
def Clip(input, min, max):
    # TODO: default values specified for min/max
    return op.Where(input < min, min, op.Where(input > max, max, input))
