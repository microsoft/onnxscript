# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import numbers
from typing import Optional, Sequence

import numpy as np
import onnx
import onnx.helper

from onnxscript import tensor


def external_tensor(
    name: str,
    data_type: int,
    dims: Sequence[int],
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> onnx.TensorProto:
    """Create a TensorProto referencing externally stored tensor-data.

    Args:
        name: name of the tensor
        data_type: data type of tensor element
        dims: shape of the tensor
        location: location of the external file (relative path)
        offset: offset in the file where the tensor-data starts
        length: number of bytes containing the data
        checksum: SHA1 digest of the file
        basepath: basepath combined with location to form the full path

    Returns:
        TensorProto

    See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md for more details.
    """
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = data_type
    tensor_proto.dims.extend(dims)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL

    def add(k, v):
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    add("location", location)
    if offset is not None:
        add("offset", int(offset))
    if length is not None:
        add("length", int(length))
    if checksum is not None:
        add("checksum", checksum)
    if basepath is not None:
        add("basepath", basepath)
    return tensor_proto


def value_to_type_proto(val):
    """Return the ONNX type of a python-value."""
    if isinstance(val, (np.ndarray, tensor.Tensor)):
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(val.dtype)
        shape = val.shape
        return onnx.helper.make_tensor_type_proto(elem_type, shape)
    if isinstance(val, int):
        return onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT32, [])
    if isinstance(val, (float, np.float32)):
        return onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
    if isinstance(val, list):
        if len(val) > 0:
            return onnx.helper.make_sequence_type_proto(value_to_type_proto(val[0]))
        # Edge-case. Cannot determine a suitable ONNX type for an empty list.
        # Should be using a typed-value instead.
        # Treated as a sequence of tensors of float-type.
        return onnx.helper.make_sequence_type_proto(
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, None)
        )
    if isinstance(val, numbers.Number):
        nparray = np.array(val)
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(nparray.dtype)
        return onnx.helper.make_tensor_type_proto(elem_type, [])
    raise ValueError(f"Value of type {type(val)} is invalid as an ONNX input/output.")


def values_to_value_infos(name_values):
    """Create a list of ValueInfoProto from a list of (name, value) pairs,
    skipping any None values.
    """
    return [
        onnx.helper.make_value_info(name, value_to_type_proto(val))
        for (name, val) in name_values
        if val is not None
    ]
