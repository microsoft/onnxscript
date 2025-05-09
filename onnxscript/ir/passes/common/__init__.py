# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "AddInitializersToInputsPass",
    "CheckerPass",
    "ClearMetadataAndDocStringPass",
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

from onnxscript.ir.passes.common.clear_metadata_and_docstring import (
    ClearMetadataAndDocStringPass,
)
from onnxscript.ir.passes.common.constant_manipulation import (
    AddInitializersToInputsPass,
    LiftConstantsToInitializersPass,
    LiftSubgraphInitializersToMainGraphPass,
    RemoveInitializersFromInputsPass,
)
from onnxscript.ir.passes.common.inliner import InlinePass
from onnxscript.ir.passes.common.onnx_checker import CheckerPass
from onnxscript.ir.passes.common.shape_inference import ShapeInferencePass
from onnxscript.ir.passes.common.topological_sort import TopologicalSortPass
from onnxscript.ir.passes.common.unused_removal import (
    RemoveUnusedFunctionsPass,
    RemoveUnusedNodesPass,
    RemoveUnusedOpsetsPass,
)
