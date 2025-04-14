"""Passes for debugging purposes."""

from __future__ import annotations

__all__ = [
    "LiftConstantsToInitializersPass",
]

import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


class CheckerPass(ir.passes.PassBase):
    """Run onnx checker on the model."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return False

    def __init__(self, lift_all_constants: bool = False, size_limit: int = 16):
        super().__init__()
        self.lift_all_constants = lift_all_constants
        self.size_limit = size_limit
