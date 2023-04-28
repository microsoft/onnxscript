# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Switches to determine if the corresponding feature of onnxscript is enabled or not."""

import os

# By default: Enable
cache_ort_session = os.environ.get("CACHE_ORT_SESSION", "1")
