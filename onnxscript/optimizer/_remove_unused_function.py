# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import TypeVar

import onnx

from onnxscript import ir

logger = logging.getLogger(__name__)


TModel = TypeVar("TModel", ir.Model, onnx.ModelProto)
