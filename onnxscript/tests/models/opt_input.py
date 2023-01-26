from typing import Optional

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

# Design choices for optional input:
# Consider the layer-normalization function, which has an optional Bias input.
# As a toy-version of this, assume we want to compute "Log(X) + Bias", where Bias
# is optional.

# The function-implementation option1 below has the following advantages:
# (1) Easier for backend to optimize away an unnecessary addition (when no bias is present)
# (2) It is more general (than option2) since it can handle cases where a simple default
# value like zero cannot be used.
# However, this requires us to add a new (pseudo) primitive-op to ONNX that does the
# check "Bias != None"


def option1(X, Bias: FLOAT[...] = None):
    Y = op.Log(X)
    if Bias != None:
        Y = Y + Bias
    return Y


# The pros/cons of the option2 implementation below are just the dual of option1.
# (1) This leads to an unnecessary tensor creation and add operation.
# (2) It could be difficult to do in the general-case. E.g., down-below, we need
# to introduce a `CastLike` op to handle the typing aspect.
# This requires introducing default-values for parameters in FunctionProto,
# similar to initializers in GraphProto


def option2(X, Bias=op.Constant(value=make_tensor("zero", TensorProto.FLOAT, [1], [0]))):
    Y = op.Log(X)
    Bias = CastLike(Bias, Y)
    Y = Y + Bias
    return Y


# The implementation option3 is similar to option1, but differs in one aspect.
# It changes the type-signature of the op/function, namely Bias is declared to
# be an optional-type. This means that we don't need to introduce a new primitive
# op, since we already have an op to check if "Bias != None" in this case.
# (See the op OptionalHasElement.)
# While this is a cleaner version of option1, it is challenging to go back and
# change the type signature of all optional parameters in ONNX ops. That would be
# quite disruptive. In short, ONNX originally introduced a limited form of
# optional inputs (before the full-fledged optional-type was introduced) and that
# is the source of our problem.

# does not work yet.
# TypeError: typing.Optional requires a single type. Got FLOAT.
# def option3(X, Bias: Optional[FLOAT[...]] = None):
#     Y = op.Log(X)
#     if (Bias != None):
#         Y = Y + Bias
#     return Y

# Proposal: The proposal is use option 1, and define a variant of the OptionalHasElement
# op in ONNX to enable this.
