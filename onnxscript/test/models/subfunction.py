# SPDX-License-Identifier: Apache-2.0
from onnxscript.onnx_types import FLOAT


def MySelu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    neg = gamma * (alpha * oxs.Exp(X) - alpha)
    pos = gamma * X
    return oxs.Where(X <= 0, neg, pos)


def MyElu(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = oxs.Constant(value_float=1.)
    return MySelu(X, alpha, beta)
