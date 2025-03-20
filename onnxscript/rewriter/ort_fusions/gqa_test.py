# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Testing GQA fusion."""

import numpy

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64

@script()
def _gqa_prompt_script(query, key, value):
    pass