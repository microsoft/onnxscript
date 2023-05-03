# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Switches to determine if the corresponding feature of onnxscript is enabled or not."""

import os

# By default: Enable
CACHE_ORT_SESSIONS: bool = os.getenv("CACHE_ORT_SESSIONS", "1") != "0"
