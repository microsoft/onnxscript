# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "AddInitializersToInputsPass",
    "CheckerPass",
    "ClearMetadataAndDocStringPass",
    "CommonSubexpressionEliminationPass",
    "InlinePass",
    "LiftConstantsToInitializersPass",
    "LiftSubgraphInitializersToMainGraphPass",
    "RemoveInitializersFromInputsPass",
    "RemoveUnusedFunctionsPass",
    "RemoveUnusedNodesPass",
    "RemoveUnusedOpsetsPass",
    "ShapeInferencePass",
    "TopologicalSortPass",
]

from onnx_ir.passes.common import (
    AddInitializersToInputsPass,
    CheckerPass,
    ClearMetadataAndDocStringPass,
    InlinePass,
)
from onnxscript.ir.passes.common.common_subexpression_elimination import (
    CommonSubexpressionEliminationPass,
)
from onnxscript.ir.passes.common.constant_manipulation import (
    AddInitializersToInputsPass,
    LiftConstantsToInitializersPass,
    LiftSubgraphInitializersToMainGraphPass,
    RemoveInitializersFromInputsPass,
    RemoveUnusedFunctionsPass,
    RemoveUnusedNodesPass,
    RemoveUnusedOpsetsPass,
    ShapeInferencePass,
    TopologicalSortPass,
)
