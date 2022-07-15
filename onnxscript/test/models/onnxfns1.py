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
from onnxscript.onnx_opset import opset15 as op

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
    return op.Max(zero, op.Min(one, alphaX * X +  betaX))

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

# TODO: remove type info for optional inputs
from onnxscript.onnx_types import OptionalFLOAT, FLOAT
@script()
def Clip(input, min: OptionalFLOAT[...] = None, max: OptionalFLOAT[...] = None):
    result = input
    if min != None:
        result = op.Where(result < min, min, result)
    if max != None:
        result = op.Where(result > max, max, result)

    return result

@script()
def CallClipScriptFunctionMinMax(input: FLOAT['N'], min: OptionalFLOAT[...], max: OptionalFLOAT[...]) -> FLOAT['N']:
    return Clip(input, min, max)

@script()
def CallClipScriptFunctionMin(input: FLOAT['N'], min: OptionalFLOAT[...]) -> FLOAT['N']:
    # TODO: ort fails with following 2 statements
    # return Clip(input, min)
    # return Clip(input, min, None)
    min_tensor = op.OptionalGetElement(min)
    return op.Clip(input, min_tensor)

@script()
def CallClipScriptFunctionMax(input: FLOAT['N'], max: OptionalFLOAT[...]) -> FLOAT['N']:
    # TODO: ort fails with following statement
    # return Clip(input, None, max)
    max_tensor = op.OptionalGetElement(max)
    return op.Clip(input, None, max_tensor)

@script()
def CallClipScriptFunction(input: FLOAT['N']) -> FLOAT['N']:
    # TODO: ort fails with following 2 statements
    # return Clip(input)
    # return Clip(input, None, None)
    return op.Clip(input)
