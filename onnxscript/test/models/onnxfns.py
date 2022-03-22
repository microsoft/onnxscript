# SPDX-License-Identifier: Apache-2.0


def Relu(X):
    return oxs.Max(X, 0.0)

# TODO: default-values of attributes


def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    neg = gamma * (alpha * oxs.Exp(X) - alpha)
    pos = gamma * X
    return oxs.Where(X <= oxs.CastLike(0, X), neg, pos)


def Elu(X, alpha: float = 1.0):
    return oxs.Where(X < 0, alpha * (oxs.Exp(X) - 1.0), X)


def ThresholdedRelu(X, alpha: float = 1.0):
    return oxs.Where(X > alpha, X, 0)


def LeakyRelu(X, alpha: float = 0.01):
    return oxs.Where(X < 0, alpha * X, X)


def PRelu(X, slope):
    # future-work: capturing extra requirements such as:
    # slope must be unidirectionally broadcastable to X's shape.
    return oxs.Where(X < 0, slope * X, X)


def HardSigmoid(X, alpha: float = 0.2, beta: float = 0.5):
    return oxs.Max(0, oxs.Min(1, alpha * X + beta))


def Shrink(x, bias: float = 0.0, lambd: float = 0.5):
    return oxs.Where(x < -lambd, x + bias, oxs.Where(x > lambd, x - bias, 0))


def Softplus(X):
    return oxs.Log(oxs.Exp(X) + 1)


def Softsign(X):
    return X / (1 + oxs.Abs(X))


def Clip(input, min, max):
    # TODO: default values specified for min/max
    return oxs.Where(input < min, min, oxs.Where(input > max, max, input))
