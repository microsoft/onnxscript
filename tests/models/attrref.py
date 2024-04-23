# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx_opset import opset15 as op


def float_attr_ref_test(X, alpha: float):
    return op.Add(X, alpha)


def int_attr_ref_test(X, alpha: int):
    return op.Add(X, alpha)


def str_attr_ref_test(X, alpha: str):
    return op.Concat(X, alpha, axis=0)
