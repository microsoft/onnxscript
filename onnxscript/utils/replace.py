# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""A utility function to replace custom operations in a model with their expansions"""

from typing import Sequence

import onnx
import onnx_ir as ir
import onnx_ir.passes.common as common_passes


def replace_functions_inplace(irmodel: ir.Model, irfunctions: Sequence[ir.Function]) -> None:
    """A utility function to replace custom operations in a model with their expansions:

    The model is updated in-place.

    Args:
        irmodel: An ONNX model possibly containing calls to custom operations.
        irfunctions: A sequence of functions defining the expansions for the custom operations.


    """
    model_functions = irmodel.functions
    if len(model_functions) != 0:
        # Since we use inlining, check that there are no model-local functions.
        raise ValueError("Input model cannot have model-local functions.")
    for func in irfunctions:
        model_functions[func.identifier()] = func

    # TODO (rama): Ideally, we should provide users more control over renaming strategy for inlined values.
    common_passes.InlinePass()(irmodel)
    common_passes.RemoveUnusedOpsetsPass()(irmodel)


def replace_functions(
    model: onnx.ModelProto, functions: Sequence[onnx.FunctionProto]
) -> onnx.ModelProto:
    """A utility function to replace custom operations in a model with their expansions:
    Args:
        model: An ONNX ModelProto possibly containing calls to custom operations.
        functions: A sequence of FunctionProto defining the expansions for the custom operations.

    Returns:
        An updated ModelProto with custom operations replaced by their expansions.
    """
    irmodel = ir.from_proto(model)
    irfunctions = [ir.from_proto(func) for func in functions]
    replace_functions_inplace(irmodel, irfunctions)
    return ir.to_proto(irmodel)
