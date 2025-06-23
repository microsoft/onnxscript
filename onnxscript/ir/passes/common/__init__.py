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
    CommonSubexpressionEliminationPass,
    InlinePass,
    LiftConstantsToInitializersPass,
    LiftSubgraphInitializersToMainGraphPass,
    RemoveInitializersFromInputsPass,
    RemoveUnusedFunctionsPass,
    RemoveUnusedNodesPass,
    RemoveUnusedOpsetsPass,
    ShapeInferencePass,
    TopologicalSortPass,
)
