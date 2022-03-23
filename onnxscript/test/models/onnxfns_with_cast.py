# SPDX-License-Identifier: Apache-2.0

import onnxscript.onnx.opset15 as op

# These function definitions include explicit CastLike ops for constants/attributes.

def Relu(X):
    zero = op.CastLike(0, X)
    return op.Max(X, zero)

# TODO: default-values of attributes


def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)


def Elu(X, alpha: float = 1.0):
    alphaX = op.CastLike(alpha, X)
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    return op.Where(X < zero, alphaX * (op.Exp(X) - one), X)


def ThresholdedRelu(X, alpha: float = 1.0):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X > alphaX, X, zero)


def LeakyRelu(X, alpha: float = 0.01):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X < zero, alphaX * X, X)


def PRelu(X, slope):
    # future-work: capturing extra requirements such as:
    # slope must be unidirectionally broadcastable to X's shape.
    zero = op.CastLike(0, X)
    return op.Where(X < zero, slope * X, X)


def HardSigmoid(X, alpha: float = 0.2, beta: float = 0.5):
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    alphaX = op.CastLike(alpha, X)
    betaX = op.CastLike(beta, X)
    return op.Max(zero, op.Min(one, alphaX * X +  betaX))


def Shrink(x, bias: float = 0.0, lambd: float = 0.5):
    zero = op.CastLike(0, x)
    return op.Where(x < -lambd, x + bias, op.Where(x > lambd, x - bias, zero))


def Softplus(X):
    one = op.CastLike(1, X)
    return op.Log(op.Exp(X) + one)


def Softsign(X):
    one = op.CastLike(1, X)
    return X / (one + op.Abs(X))


def Clip(input, min, max):
    # TODO: default values specified for min/max
    return op.Where(input < min, min, op.Where(input > max, max, input))
