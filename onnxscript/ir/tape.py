# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Taping module to facilitate building IR graphs."""

# NOTE: Be *selective* about what this module exports because it is part of the public API.

from __future__ import annotations

__all__ = [
    "Tape",
]

from onnxscript.ir._tape import Tape

Tape.__module__ = __name__
