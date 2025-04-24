# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience constructors for IR objects."""

from __future__ import annotations

__all__ = [
    "tensor",
    "node",
]

import typing
from typing import Mapping, Sequence

import numpy as np
import onnx

from onnxscript.ir import _convenience, _core, _enums, _protocols, serde, tensor_adapters

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from onnxscript import ir


def tensor(
    value: npt.ArrayLike | onnx.TensorProto | ir.DLPackCompatible | ir.ArrayCompatible,
    dtype: _enums.DataType | None = None,
    name: str | None = None,
    doc_string: str | None = None,
) -> _protocols.TensorProtocol:
    """Create a tensor value from an ArrayLike object or a TensorProto.

    The dtype must match the value. Reinterpretation of the value is
    not supported, unless if the value is a plain Python object, in which case
    it is converted to a numpy array with the given dtype.

    ``value`` can be a numpy array, a plain Python object, or a TensorProto.

    Example::

        >>> from onnxscript import ir
        >>> import numpy as np
        >>> import ml_dtypes
        >>> import onnx
        >>> ir.tensor(np.array([1, 2, 3], dtype=np.int16))
        Tensor<INT16,[3]>(array([1, 2, 3], dtype=int16), name=None)
        >>> ir.tensor([1, 2, 3], dtype=ir.DataType.BFLOAT16)
        Tensor<BFLOAT16,[3]>(array([1, 2, 3], dtype=bfloat16), name=None)
        >>> tp_tensor = ir.tensor(onnx.helper.make_tensor("tensor", onnx.TensorProto.FLOAT, dims=[], vals=[0.5]))
        >>> tp_tensor.numpy()
        array(0.5, dtype=float32)
        >>> import torch
        >>> ir.tensor(torch.tensor([1.0, 2.0]), name="torch_tensor")
        TorchTensor<FLOAT,[2]>(tensor([1., 2.]), name='torch_tensor')

    Args:
        value: The numpy array to create the tensor from.
        dtype: The data type of the tensor.
        name: The name of the tensor.
        doc_string: The documentation string of the tensor.

    Returns:
        A tensor value.

    Raises:
        ValueError: If the dtype does not match the value when value is not a plain Python
            object like ``list[int]``.
    """
    if isinstance(value, _protocols.TensorProtocol):
        if dtype is not None and dtype != value.dtype:
            raise ValueError(
                f"The dtype must match the value when value is a Tensor. dtype={dtype}, value.dtype={value.dtype}. "
                "You do not have to specify the dtype when value is a Tensor."
            )
        return value
    if isinstance(value, onnx.TensorProto):
        tensor_ = serde.deserialize_tensor(value)
        if name is not None:
            tensor_.name = name
        if doc_string is not None:
            tensor_.doc_string = doc_string
        if dtype is not None and dtype != tensor_.dtype:
            raise ValueError(
                f"The dtype must match the value when value is a TensorProto. dtype={dtype}, value.data_type={tensor_.dtype}"
                "You do not have to specify the dtype when value is a TensorProto."
            )
        return tensor_
    elif str(type(value)) == "<class 'torch.Tensor'>":
        # NOTE: We use str(type(...)) and do not import torch for type checking
        # as it creates overhead during import
        return tensor_adapters.TorchTensor(value, name=name, doc_string=doc_string)  # type: ignore[arg-type]
    elif isinstance(value, (_protocols.DLPackCompatible, _protocols.ArrayCompatible)):
        return _core.Tensor(value, dtype=dtype, name=name, doc_string=doc_string)
    # Plain Python object
    if dtype is not None:
        numpy_dtype = dtype.numpy()
    else:
        numpy_dtype = None
    array = np.array(value, dtype=numpy_dtype)
    return _core.Tensor(
        array,
        dtype=dtype,
        shape=_core.Shape(array.shape),
        name=name,
        doc_string=doc_string,
    )


def node(
    op_type: str,
    inputs: Sequence[ir.Value | None],
    attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
    *,
    domain: str = "",
    overload: str = "",
    num_outputs: int | None = None,
    outputs: Sequence[ir.Value] | None = None,
    version: int | None = None,
    graph: ir.Graph | None = None,
    name: str | None = None,
    doc_string: str | None = None,
    metadata_props: dict[str, str] | None = None,
) -> ir.Node:
    """Create an :class:`ir.Node`.

    This is a convenience constructor for creating a Node that supports Python
    objects as attributes.

    Example::

        >>> from onnxscript import ir
        >>> input_a = ir.Input("A", shape=ir.Shape([1, 2]), type=ir.TensorType(ir.DataType.INT32))
        >>> input_b = ir.Input("B", shape=ir.Shape([1, 2]), type=ir.TensorType(ir.DataType.INT32))
        >>> node = ir.node(
        ...     "SomeOp",
        ...     inputs=[input_a, input_b],
        ...     attributes={"alpha": 1.0, "some_list": [1, 2, 3]},
        ...     domain="some.domain",
        ...     name="node_name"
        ... )
        >>> node.op_type
        'SomeOp'

    Args:
        op_type: The name of the operator.
        inputs: The input values. When an input is None, it is an empty input.
        attributes: The attributes. RefAttr can be used only when the node is defined in a Function.
        overload: The overload name when the node is invoking a function.
        domain: The domain of the operator. For onnx operators, this is an empty string.
        num_outputs: The number of outputs of the node. If not specified, the number is 1.
        outputs: The output values. If None, the outputs are created during initialization.
        version: The version of the operator. If None, the version is unspecified and will follow that of the graph.
        graph: The graph that the node belongs to. If None, the node is not added to any graph.
            A `Node` must belong to zero or one graph.
        name: The name of the node. If None, the node is anonymous.
        doc_string: The documentation string.
        metadata_props: The metadata properties.

    Returns:
        A node with the given op_type and inputs.
    """
    if attributes is None:
        attrs: Sequence[ir.Attr | ir.RefAttr] = ()
    else:
        attrs = _convenience.convert_attributes(attributes)
    return _core.Node(
        domain=domain,
        op_type=op_type,
        inputs=inputs,
        attributes=attrs,
        overload=overload,
        num_outputs=num_outputs,
        outputs=outputs,
        version=version,
        graph=graph,
        name=name,
        doc_string=doc_string,
        metadata_props=metadata_props,
    )
