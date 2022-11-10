# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import onnx

class Opt:
    """An implementation of ONNX Optional type values."""

    def __init__(self, val_or_type) -> None:
        if isinstance(val_or_type, onnx.TypeProto):
            self.val = None
            self.type = val_or_type
        else:
            self.val = val_or_type
            self.type = val_or_type.type_proto()
