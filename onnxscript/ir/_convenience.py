# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing and manipulating the IR.

This is an internal only module. We should choose to expose some of the methods
in convenience.py after they are proven to be useful.
"""

from __future__ import annotations

__all__ = [
    "convert_attribute",
    "convert_attributes",
    "replace_all_uses_with",
]

import typing
from typing import Mapping, Sequence, Union

import numpy as np
import onnx

from onnxscript.ir import _core, _enums, _protocols, serde

if typing.TYPE_CHECKING:
    import numpy.typing as npt

SupportedAttrTypes = Union[
    str,
    int,
    float,
    Sequence[int],
    Sequence[float],
    Sequence[str],
    _protocols.TensorProtocol,  # This includes all in-memory tensor types
    onnx.TensorProto,
    _core.Attr,
    _core.RefAttr,
    _protocols.GraphProtocol,
    Sequence[_protocols.GraphProtocol],
    _protocols.TypeProtocol,
    Sequence[_protocols.TypeProtocol],
    None,
]


def _infer_attribute_type(attr: SupportedAttrTypes) -> _enums.AttributeType:
    """Infer the attribute type based on the type of the Python object."""
    if isinstance(attr, int):
        return _enums.AttributeType.INT
    if isinstance(attr, float):
        return _enums.AttributeType.FLOAT
    if isinstance(attr, str):
        return _enums.AttributeType.STRING
    if isinstance(attr, (_core.Attr, _core.RefAttr)):
        return attr.type
    if isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
        return _enums.AttributeType.INTS
    if isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
        return _enums.AttributeType.FLOATS
    if isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
        return _enums.AttributeType.STRINGS
    if isinstance(attr, (_core.TensorBase, onnx.TensorProto, _protocols.TensorProtocol)):
        # Be sure to check TensorProtocol last because isinstance checking on Protocols can be slower
        return _enums.AttributeType.TENSOR
    if isinstance(attr, (_core.Graph, _protocols.GraphProtocol)):
        return _enums.AttributeType.GRAPH
    if isinstance(attr, Sequence) and all(
        isinstance(x, (_core.Graph, _protocols.GraphProtocol)) for x in attr
    ):
        return _enums.AttributeType.GRAPHS
    if isinstance(
        attr,
        (_core.TensorType, _core.SequenceType, _core.OptionalType, _protocols.TypeProtocol),
    ):
        return _enums.AttributeType.TYPE_PROTO
    if isinstance(attr, Sequence) and all(
        isinstance(
            x,
            (
                _core.TensorType,
                _core.SequenceType,
                _core.OptionalType,
                _protocols.TypeProtocol,
            ),
        )
        for x in attr
    ):
        return _enums.AttributeType.TYPE_PROTOS
    raise TypeError(f"Unsupported attribute type: '{type(attr)}'")


def convert_attribute(
    name: str,
    attr: SupportedAttrTypes,
    attr_type: _enums.AttributeType | None = None,
) -> _core.Attr | _core.RefAttr:
    """Convert a Python object to a _core.Attr object.

    This method is useful when constructing nodes with attributes. It infers the
    attribute type based on the type of the Python value.

    Args:
        name: The name of the attribute.
        attr: The value of the attribute.
        attr_type: The type of the attribute. This is required when attr is None.
            When provided, it overrides the inferred type.

    Returns:
        A ``Attr`` object.

    Raises:
        ValueError: If :param:`attr` is ``None`` and :param:`attr_type` is not provided.
        TypeError: If the type of the attribute is not supported.
    """
    if attr is None:
        if attr_type is None:
            raise ValueError("attr_type must be provided when attr is None")
        return _core.Attr(name, attr_type, None)

    if isinstance(attr, (_core.Attr, _core.RefAttr)):
        if attr.name != name:
            raise ValueError(
                f"Attribute name '{attr.name}' does not match provided name '{name}'"
            )
        if attr_type is not None and attr.type != attr_type:
            raise ValueError(
                f"Attribute type '{attr.type}' does not match provided type '{attr_type}'"
            )
        return attr

    if attr_type is None:
        attr_type = _infer_attribute_type(attr)

    if attr_type == _enums.AttributeType.INT:
        return _core.AttrInt64(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.FLOAT:
        return _core.AttrFloat32(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.STRING:
        return _core.AttrString(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.INTS:
        return _core.AttrInt64s(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.FLOATS:
        return _core.AttrFloat32s(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.STRINGS:
        return _core.AttrStrings(name, attr)  # type: ignore
    if attr_type == _enums.AttributeType.TENSOR:
        if isinstance(attr, (_core.TensorBase, _protocols.TensorProtocol)):
            return _core.AttrTensor(name, attr)
        if isinstance(attr, onnx.TensorProto):
            return _core.AttrTensor(name, serde.TensorProtoTensor(attr))
    if attr_type == _enums.AttributeType.GRAPH:
        return _core.AttrGraph(name, attr)  # type: ignore[arg-type]
    if attr_type == _enums.AttributeType.GRAPHS:
        return _core.AttrGraphs(name, attr)  # type: ignore[arg-type]
    if attr_type == _enums.AttributeType.TYPE_PROTO:
        return _core.AttrTypeProto(name, attr)  # type: ignore[arg-type]
    if attr_type == _enums.AttributeType.TYPE_PROTOS:
        return _core.AttrTypeProtos(name, attr)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported attribute type: '{type(attr)}'")


def convert_attributes(
    attrs: Mapping[str, SupportedAttrTypes],
) -> list[_core.Attr | _core.RefAttr]:
    """Convert a dictionary of attributes to a list of _core.Attr objects.

    It infers the attribute type based on the type of the value. The supported
    types are: int, float, str, Sequence[int], Sequence[float], Sequence[str],
    :class:`_core.Tensor`, and :class:`_core.Attr`::

        >>> from onnxscript import ir
        >>> import onnx
        >>> import numpy as np
        >>> attrs = {
        ...     "int": 1,
        ...     "float": 1.0,
        ...     "str": "hello",
        ...     "ints": [1, 2, 3],
        ...     "floats": [1.0, 2.0, 3.0],
        ...     "strings": ["hello", "world"],
        ...     "tensor": ir.Tensor(np.array([1.0, 2.0, 3.0])),
        ...     "tensor_proto":
        ...         onnx.TensorProto(
        ...             dims=[3],
        ...             data_type=onnx.TensorProto.FLOAT,
        ...             float_data=[1.0, 2.0, 3.0],
        ...             name="proto",
        ...         ),
        ...     "graph": ir.Graph([], [], nodes=[], name="graph0"),
        ...     "graphs": [ir.Graph([], [], nodes=[], name="graph1"), ir.Graph([], [], nodes=[], name="graph2")],
        ...     "type_proto": ir.TensorType(ir.DataType.FLOAT),
        ...     "type_protos": [ir.TensorType(ir.DataType.FLOAT), ir.TensorType(ir.DataType.FLOAT)],
        ... }
        >>> convert_attributes(attrs)
        [AttrInt64('int', 1), AttrFloat32('float', 1.0), AttrString('str', 'hello'), AttrInt64s('ints', [1, 2, 3]), AttrFloat32s('floats', [1.0, 2.0, 3.0]), AttrStrings('strings', ['hello', 'world']), AttrTensor('tensor', Tensor<DOUBLE,[3]>(array([1., 2., 3.]), name=None)), AttrTensor('tensor_proto', TensorProtoTensor<FLOAT,[3]>(name='proto')), AttrInt64s('graph', Graph(
            name='graph0',
            inputs=(
        <BLANKLINE>
            ),
            outputs=(
        <BLANKLINE>
            ),
            len()=0
        )), AttrGraphs('graphs', [Graph(
            name='graph1',
            inputs=(
        <BLANKLINE>
            ),
            outputs=(
        <BLANKLINE>
            ),
            len()=0
        ), Graph(
            name='graph2',
            inputs=(
        <BLANKLINE>
            ),
            outputs=(
        <BLANKLINE>
            ),
            len()=0
        )]), AttrTypeProto('type_proto', Tensor(FLOAT)), AttrTypeProtos('type_protos', [Tensor(FLOAT), Tensor(FLOAT)])]

    Args:
        attrs: A dictionary of {<attribute name>: <python objects>} to convert.

    Returns:
        A list of _core.Attr objects.
    """
    attributes: list[_core.Attr | _core.RefAttr] = []
    for name, attr in attrs.items():
        if attr is not None:
            attributes.append(convert_attribute(name, attr))
    return attributes


def replace_all_uses_with(
    values: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol],
    replacements: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol],
) -> None:
    """Replace all uses of the given values with the replacements.

    This is useful when nodes in the graph are replaced with new nodes, where
    the old users need to be updated to use the outputs of the new nodes.

    For example, suppose we have the following graph::

        A -> {B, C}

    We want to replace the node A with a new node D::

        >>> from onnxscript import ir
        >>> input = ir.Input("input")
        >>> node_a = ir.Node("", "A", [input])
        >>> node_b = ir.Node("", "B", node_a.outputs)
        >>> node_c = ir.Node("", "C", node_a.outputs)
        >>> node_d = ir.Node("", "D", [input])
        >>> replace_all_uses_with(node_a.outputs, node_d.outputs)
        >>> len(node_b.inputs)
        1
        >>> node_b.inputs[0].producer().op_type
        'D'
        >>> len(node_c.inputs)
        1
        >>> node_c.inputs[0].producer().op_type
        'D'
        >>> len(node_a.outputs[0].uses())
        0

    When values and replacements are sequences, they are zipped into pairs. All
    users of the first value is replaced with the first replacement, and so on.

    .. note::
        You still need to update the graph outputs if any of the values being
        replaced are part of the graph outputs. Be sure to remove the old nodes
        from the graph using ``graph.remove()`` if they are no longer needed.

    Args:
        values: The value or values to be replaced.
        replacements: The new value or values to use as inputs.
    """
    if not isinstance(values, Sequence):
        values = (values,)
    if not isinstance(replacements, Sequence):
        replacements = (replacements,)
    if len(values) != len(replacements):
        raise ValueError("The number of values and replacements must match.")
    for value, replacement in zip(values, replacements):
        for user_node, index in tuple(value.uses()):
            user_node.replace_input_with(index, replacement)


def tensor(
    value: npt.ArrayLike
    | onnx.TensorProto
    | _protocols.DLPackCompatible
    | _protocols.ArrayCompatible,
    dtype: _enums.DataType | None = None,
    name: str | None = None,
    doc_string: str | None = None,
) -> _protocols.TensorProtocol:
    """Create a tensor value from an ArrayLike object or a TensorProto.

    The dtype must match the value. Reinterpretation of the value is
    not supported, unless if the value is a plain Python object, in which case
    it is converted to a numpy array with the given dtype.

    :param:`value` can be a numpy array, a plain Python object, or a TensorProto.

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
    elif isinstance(value, (_protocols.DLPackCompatible, _protocols.ArrayCompatible)):
        tensor_ = _core.Tensor(value, dtype=dtype, name=name, doc_string=name)
    else:
        if dtype is not None:
            numpy_dtype = dtype.numpy()
        else:
            numpy_dtype = None
        array = np.array(value, dtype=numpy_dtype)
        tensor_ = _core.Tensor(
            array,
            dtype=dtype,
            shape=_core.Shape(array.shape),
            name=name,
            doc_string=name,
        )
    return tensor_


def create_value_mapping(graph: _core.Graph) -> dict[str, _core.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


def replace_nodes_and_values(
    graph_or_function: _core.Graph | _core.Function,
    /,
    insertion_point: _core.Node,
    old_nodes: Sequence[_core.Node],
    new_nodes: Sequence[_core.Node],
    old_values: Sequence[_core.Value],
    new_values: Sequence[_core.Value],
) -> None:
    """Replaces nodes and values in the graph or function.

    Args:
        graph_or_function: The graph or function to replace nodes and values in.
        insertion_point: The node to insert the new nodes after.
        old_nodes: The nodes to replace.
        new_nodes: The nodes to replace with.
        old_values: The values to replace.
        new_values: The values to replace with.
    """

    for old_value, new_value in zip(old_values, new_values):
        # Propagate relevant info from old value to new value
        # TODO(Rama): Perhaps this should be a separate utility function. Also, consider
        # merging old and new type/shape info.
        new_value.type = old_value.type
        new_value.shape = old_value.shape
        new_value.const_value = old_value.const_value
        new_value.name = old_value.name

    # Reconnect the users of the deleted values to use the new values
    replace_all_uses_with(old_values, new_values)
    # Update graph/function outputs if the node generates output
    replacement_mapping = dict(zip(old_values, new_values))
    for idx, graph_or_function_output in enumerate(graph_or_function.outputs):
        if graph_or_function_output in replacement_mapping:
            graph_or_function.outputs[idx] = replacement_mapping[graph_or_function_output]

    # insert new nodes after the index node
    graph_or_function.insert_after(insertion_point, new_nodes)
    graph_or_function.remove(old_nodes, safe=True)
