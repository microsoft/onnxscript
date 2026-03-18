# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GGUF model import support for mobius.

This package provides tools to load GGUF model files and convert them
to ONNX models using the existing graph construction pipeline.

Usage::

    from mobius.integrations.gguf import build_from_gguf

    pkg = build_from_gguf("path/to/model.gguf")
"""

from __future__ import annotations

from mobius.integrations.gguf._builder import build_from_gguf

__all__ = ["build_from_gguf"]
