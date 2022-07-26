# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT



@script()
def make_sequence_tensor(A):
    seq = op.SequenceEmpty()
    for i in range(10):
        seq = op.SequenceInsert(seq, A * 2)
    return op.ConcatFromSequence(seq)


@script()
def make_sequence(A):
    seq = op.SequenceEmpty()
    for i in range(10):
        seq = op.SequenceInsert(seq, A * 2)
    return seq


@script()
def make_sequence_python(A):
    seq = []
    for i in range(10):
        B = A * 2
        seq.append(B)
    ten = op.ConcatFromSequence(seq)
    return ten 

