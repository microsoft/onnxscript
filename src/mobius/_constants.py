# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Package-wide constants."""

from __future__ import annotations

# Default ONNX opset version used for all graph construction.
# Separated into its own module to avoid circular imports between
# tasks, components, and the top-level package.
OPSET_VERSION = 24
