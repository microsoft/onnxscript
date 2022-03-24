# SPDX-License-Identifier: Apache-2.0

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

import onnxscript.onnx.opset15 as op

def Relu(X):
    zero = op.CastLike(0, X)
    return op.Max(X, zero)

def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

def Elu(X, alpha: float):
    alphaX = op.CastLike(alpha, X)
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    return op.Where(X < zero, alphaX * (op.Exp(X) - one), X)

def ThresholdedRelu(X, alpha: float):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X > alphaX, X, zero)

def LeakyRelu(X, alpha: float):
    zero = op.CastLike(0, X)
    alphaX = op.CastLike(alpha, X)
    return op.Where(X < zero, alphaX * X, X)

def PRelu(X, slope):
    zero = op.CastLike(0, X)
    return op.Where(X < zero, slope * X, X)

def HardSigmoid(X, alpha: float, beta: float):
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    alphaX = op.CastLike(alpha, X)
    betaX = op.CastLike(beta, X)
    return op.Max(zero, op.Min(one, alphaX * X +  betaX))

def Shrink(x, bias: float, lambd: float):
    zero = op.CastLike(0, x)
    return op.Where(x < -lambd, x + bias, op.Where(x > lambd, x - bias, zero))

def Softplus(X):
    one = op.CastLike(1, X)
    return op.Log(op.Exp(X) + one)

def Softsign(X):
    one = op.CastLike(1, X)
    return X / (one + op.Abs(X))

def Clip(input, min, max):
    return op.Where(input < min, min, op.Where(input > max, max, input))
