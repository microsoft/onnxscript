# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Features included:
# Overloaded operators such as <=, +, /
# Nested expressions
# Type annotation for attributes (to distinguish inputs and attributes)
# Importing (predefined) opsets

# Features not yet covered:
# Auto-cast for constants/attributes. (Must include explicit CastLike ops)
# No default-values for attributes yet, which are currently present in the opschema definition (in ONNX).
# No default-values for inputs yet.
# Element-type annotation for tensors

from onnxscript import script
from onnxscript.onnx_opset import opset18 as op


@script()
def Relu(X):
    zero = op.CastLike(0, X)
    return op.Max(X, zero)


@script()
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)


@script()
def Elu(X, alpha: float):
    alphaX = op.CastLike(alpha, X)
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    return op.Where(X < zero, alphaX * (op.Exp(X) - one), X)


@script()
def ThresholdedRelu(X, alpha: float):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X > alphaX, X, zero)


@script()
def LeakyRelu(X, alpha: float):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X < zero, alphaX * X, X)


@script()
def PRelu(X, slope):
    zero = op.CastLike(0, X)
    return op.Where(X < zero, slope * X, X)


@script()
def HardSigmoid(X, alpha: float, beta: float):
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    alphaX = op.CastLike(alpha, X)
    betaX = op.CastLike(beta, X)
    return op.Max(zero, op.Min(one, alphaX * X + betaX))


@script()
def Shrink(x, bias: float, lambd: float):
    zero = op.CastLike(0, x)
    return op.Where(x < -lambd, x + bias, op.Where(x > lambd, x - bias, zero))


@script()
def Softplus(X):
    one = op.CastLike(1, X)
    return op.Log(op.Exp(X) + one)


@script()
def Softsign(X):
    one = op.CastLike(1, X)
    return X / (one + op.Abs(X))


from onnxscript.onnx_types import BOOL, FLOAT


@script()
def Clip(input: FLOAT[...], min: FLOAT = None, max: FLOAT = None) -> FLOAT[...]:
    result = input
    if op.OptionalHasElement(min):
        result = op.Where(result < min, min, result)
    if op.OptionalHasElement(max):
        result = op.Where(result > max, max, result)

    return result


# @script()
# def Clip(input, min, max):
#     return op.Where(input < min, min, op.Where(input > max, max, input))


@script()
def OptionalHasElement(input: FLOAT[...]) -> BOOL:
    return op.OptionalHasElement(input)
