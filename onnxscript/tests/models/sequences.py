# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def make_sequence_tensor(A):
    # If replaced by [], attribute dtype can not be easily changed.
    seq = op.SequenceEmpty()
    B = A
    for i in range(5):
        seq = op.SequenceInsert(seq, B)
        B = B * 2
    return op.ConcatFromSequence(seq, axis=0)


@script()
def make_sequence_tensor_accumulated(A):
    seq = op.SequenceEmpty()
    B = A
    C = A * 0
    for i in range(5):
        seq = op.SequenceInsert(seq, B)
        B = B * 2
        C = C + B + 1
    return op.ConcatFromSequence(seq, axis=0) - C
