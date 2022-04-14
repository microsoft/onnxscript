# SPDX-License-Identifier: Apache-2.0

# Same functions as in onnxfns1.py, using autocast and default-attribute-values

import onnxscript.onnx.opset15 as op

def Relu(X):
    return op.Max(X, 0)

def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    neg = gamma * (alpha * op.Exp(X) - alpha)
    pos = gamma * X
    return op.Where(X <= 0, neg, pos)

def Elu(X, alpha: float = 1.0):
    return op.Where(X < 0, alpha * (op.Exp(X) - 1.0), X)

def ThresholdedRelu(X, alpha: float = 1.0):
    return op.Where(X > alpha, X, 0)

def LeakyRelu(X, alpha: float = 0.01):
    return op.Where(X < 0, alpha * X, X)

def PRelu(X, slope):
    # future-work: capturing extra requirements such as:
    # slope must be unidirectionally broadcastable to X's shape.
    return op.Where(X < 0, slope * X, X)

def HardSigmoid(X, alpha: float = 0.2, beta: float = 0.5):
    return op.Max(0, op.Min(1, alpha * X + beta))

def Shrink(x, bias: float = 0.0, lambd: float = 0.5):
    return op.Where(x < -lambd, x + bias, op.Where(x > lambd, x - bias, 0))

def Softplus(X):
    return op.Log(op.Exp(X) + 1)

def Softsign(X):
    return X / (1 + op.Abs(X))

def Clip(input, min, max):
    # TODO: default values specified for min/max
    return op.Where(input < min, min, op.Where(input > max, max, input))
