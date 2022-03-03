# SPDX-License-Identifier: Apache-2.0


def Relu(X):
    return onnx.Max(X, 0)

# TODO: default-values of attributes


def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    neg = gamma * (alpha * onnx.Exp(X) - alpha)
    pos = gamma * X
    return onnx.Where(X <= 0, neg, pos)


def Elu(X, alpha: float = 1.0):
    return onnx.Where(X < 0, alpha * (onnx.Exp(X) - 1.0), X)


def ThresholdedRelu(X, alpha: float = 1.0):
    return onnx.Where(X > alpha, X, 0)


def LeakyRelu(X, alpha: float = 0.01):
    return onnx.Where(X < 0, alpha * X, X)


def PRelu(X, slope):
    # future-work: capturing extra requirements such as:
    # slope must be unidirectionally broadcastable to X's shape.
    return onnx.Where(X < 0, slope * X, X)


def HardSigmoid(X, alpha: float = 0.2, beta: float = 0.5):
    return onnx.Max(0, onnx.Min(1, alpha * X + beta))


def Shrink(x, bias: float = 0.0, lambd: float = 0.5):
    return onnx.Where(x < -lambd, x + bias, Onnx.Where(x > lambd, x - bias, 0))


def Softplus(X):
    return onnx.Log(onnx.Exp(X) + 1)


def Softsign(X):
    return X / (1 + onnx.Abs(X))


def Clip(input, min, max):
    # TODO: default values specified for min/max
    return onnx.Where(input < min, min, onnx.Where(input > max, max, input))
