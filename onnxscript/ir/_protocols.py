# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Protocols for the ONNX IR.

This file defines the interfaces for tools to interact with the IR. The interfaces
are designed such that tools leveraging the IR can be decoupled from the IR
implementation. This allows for the implementation to evolve independently of the
tools.
"""

# 👀
# NOTE: Why are we using protocols, instead of abstract base classes?
#
# Protocols are more flexible than abstract base classes. Users can define their
# own classes that implement the protocols without having to inherit from a
# specific base class. For example, a user can define a custom tensor class that
# implements the TensorProtocol without explicitly inheriting, and the IR can
# work with that class without any changes.
#
# `isinstance` checks can be slower with protocols. Avoid using `isinstance`
# checks when you can. Always check for concrete classes first.
#
# NOTE: Why are we using protocols, instead of using concrete classes directly?
#
# Protocols define the interface that is typically more stable. If you find yourself
# updating the protocols, pause 🛑, and carefully make sure it is absolutely needed
# and will improve the design. If you are adding new methods, consider if the method
# should be part of the protocol or if it should be a higher level convenience function
# defined outside the protocol.

from __future__ import annotations

import typing
from typing import (
    Any,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    OrderedDict,
    Protocol,
    Sequence,
    Tuple,
)

from onnxscript.ir import _enums

if typing.TYPE_CHECKING:
    import numpy as np
    from typing_extensions import TypeAlias

# An identifier that will uniquely identify an operator. E.g (domain, op_type, overload)
OperatorIdentifier: TypeAlias = Tuple[str, str, str]


@typing.runtime_checkable
class ArrayCompatible(Protocol):
    """Protocol for array-like objects.

    An example of an array-like object is a numpy ndarray or a PyTorch Tensor.
    Read more at https://numpy.org/devdocs/user/basics.interoperability.html
    """

    def __array__(self, dtype: Any) -> np.ndarray: ...


@typing.runtime_checkable
class DLPackCompatible(Protocol):
    """Protocol for objects that can support dlpack.

    Computation backends can call __dlpack__ to obtain the underlying data in a
    tensor without copying the data. This allows use to use tensorflow tensors etc.
    without copying the data.
    """

    def __dlpack__(self, *, stream: Any = ...) -> Any:
        """Return PyCapsule."""
        ...

    def __dlpack_device__(self) -> Any:
        """Return the device."""
        ...


@typing.runtime_checkable
class TensorProtocol(ArrayCompatible, DLPackCompatible, Protocol):
    """Concrete tensor backed by data.

    The protocol does not specify how the data is stored. That data is exposed
    through the :attr:`raw` attribute for examination, but accessing :attr:`raw`
    is typically not needed.

    To use the tensor as a numpy array, call :meth:`numpy`. To convert the tensor
    to a byte string for serialization, call :meth:`tobytes`.

    It is recommended to check the size of the tensor first before accessing the
    underlying data, because accessing the data may be expensive and incur IO
    overhead.

    Attributes:
        name: The name of the tensor.
        shape: The shape of the tensor.
        dtype: The data type of the elements of the tensor. It is an :class:`ir.DataType` enum.
        doc_string: Documentation string.
        raw: The raw data behind this tensor. It can be anything.
        size: The number of elements in the tensor.
        nbytes: The number of bytes in the tensor.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    name: str | None
    shape: ShapeProtocol
    dtype: _enums.DataType
    doc_string: str | None
    raw: Any
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]

    @property
    def size(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array."""
        ...

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the tensor as a numpy array, compatible with np.array."""
        ...

    def __dlpack__(self, *, stream: Any = ...) -> Any:
        """Return PyCapsule."""
        ...

    def __dlpack_device__(self) -> Any:
        """Return the device."""
        ...

    def tobytes(self) -> bytes:
        """Return the tensor as a byte string conformed to the ONNX specification, in little endian."""
        ...


@typing.runtime_checkable
class ValueProtocol(Protocol):
    """Protocol for values.

    A value is a named entity that can be used to represent an input or output of a graph,
    a function, or a node. The information it stores generalizes over ``ValueInfoProto``
    in the ONNX specification.

    A :class:`Value` is always not owned or owned by exactly one node. When the value is not
    owned, it must be an input of a graph or a function. ``producer`` and ``index``
    are ``None``.

    When the value is owned by a node, it is an output of the node.
    The node that produces the value can be accessed with :meth:`producer`.
    The index of the output of the node that produces the value can be accessed with
    :meth:`index`.

    To find all the nodes that use this value as an input, call :meth:`uses`.

    To check if the value is an output of a graph, call :meth:`is_graph_output`.

    Attributes:
        name: The name of the value. A value is always named when it is part of a graph.
        shape: The shape of the value.
        type: The type of the value.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
        doc_string: Documentation string.
        const_value: The constant tensor is the value constant.
    """

    name: str
    shape: ShapeProtocol | None
    type: TypeProtocol | None
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]
    doc_string: str | None
    const_value: TensorProtocol | None

    def producer(self) -> NodeProtocol | None:
        """The node that produces this value."""
        ...

    def index(self) -> int | None:
        """The index of the output of the node that produces this value."""
        ...

    def uses(self) -> Collection[tuple[NodeProtocol, int]]:
        """The set of (node, input_index) with node being those that use this value as an input."""
        ...

    def is_graph_output(self) -> bool:
        """Whether this value is an output of a graph."""
        ...


@typing.runtime_checkable
class NodeProtocol(Protocol):
    """Protocol for nodes.

    A node represents an invocation of an operation on the :class:`Value` s in
    the computational graph.

    A node can be optionally named. A name should typically be assigned when the
    node is added to a graph.

    :attr:`domain`, :attr:`op_type`, and :attr:`overload` together uniquely identify
    the operator, and are always strings. For ONNX operators, :attr:`domain` and :attr:`overload`
    are both empty strings.

    :attr:`inputs` and :attr:`outputs` are the input and output values of the node.

    :attr:`attributes` are the attributes of the node. The attributes are stored in an
    ordered dictionary to preserve the order of the attributes. This is a deviation from
    the current ONNX spec where attributes are unordered, but it is helpful for tools
    that rely on the order of the attributes, e.g. those converting to and from Python
    function keyword arguments.

    :attr:`version` is unique to the IR and is not specified in the ONNX spec. This
    allows the IR to represent a graph with mixed opset versions. Deserializers
    should decide how to reconcile the different versions within the graph. A typical
    graph will have a single version, declared in the :class:`Graph` object and
    the nodes will have ``None`` as the version.

    Attributes:
        domain: The domain of the operator. E.g. ``""`` for ONNX operators.
        op_type: The operator name.
        overload: The overload name when the node is invoking a function.
        inputs: Input values.
        outputs: Output values.
        attributes: The attributes of the operator.
        version: The version of the operator.
        doc_string: Documentation string.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    name: str | None
    domain: str
    op_type: str
    overload: str
    inputs: Sequence[ValueProtocol]
    outputs: Sequence[ValueProtocol]
    attributes: OrderedDict[str, AttributeProtocol | ReferenceAttributeProtocol]
    version: int | None
    doc_string: str | None
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]

    def replace_input_with(self, index: int, value: ValueProtocol | None) -> None:
        """Set the input at the given index to the given value, replacing the original value."""
        ...


@typing.runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for graphs.

    Graph represents a computation graph. In addition to the ONNX specification
    specified fields, it also contains a mapping of :attr:`opset_imports`. This
    allows different subgraphs to import different opsets. It is the responsibility
    of the deserializer to reconcile the different opsets.

    The nodes are not guaranteed to be topologically sorted. But the
    iteration order should be deterministic across different runs. It is the
    responsibility of the user to maintain a topological order of the nodes.

    Note that there is not a ``node`` attribute in the Graph. The Graph can be
    seen as a Sequence of nodes and should be used as such. For example, to obtain
    all nodes as a list, call ``list(graph)``.

    Attributes:
        name: The name of the graph.
        inputs: The input values of the graph.
        outputs: The output values of the graph.
        initializers: The initializers in the graph.
        doc_string: Documentation string.
        opset_imports: Opsets imported by the graph.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    # TODO(justinchuby): Support quantization_annotation
    name: str | None
    inputs: MutableSequence[ValueProtocol]
    outputs: MutableSequence[ValueProtocol]
    initializers: MutableMapping[str, ValueProtocol]
    doc_string: str
    opset_imports: MutableMapping[str, int]
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]

    def __getitem__(self, index: int) -> NodeProtocol: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NodeProtocol]: ...
    def __reversed__(self) -> Iterator[NodeProtocol]: ...

    # Mutation methods
    def append(self, node: NodeProtocol, /) -> None:
        """Append a node to the graph."""
        ...

    def extend(self, nodes: Iterable[NodeProtocol], /) -> None:
        """Extend the graph with the given nodes."""
        ...

    def remove(self, node: NodeProtocol, /) -> None:
        """Remove a node from the graph."""
        ...

    def insert_after(self, node: NodeProtocol, new_nodes: Iterator[NodeProtocol], /) -> None:
        """Insert new nodes after the given node."""
        ...

    def insert_before(self, node: NodeProtocol, new_nodes: Iterator[NodeProtocol], /) -> None:
        """Insert new nodes before the given node."""
        ...

    def sort(self) -> None:
        """Topologically sort the nodes in the graph."""
        ...


@typing.runtime_checkable
class GraphViewProtocol(Protocol):
    """Protocol for a read-only view on a graph.

    The GraphView is useful for analysis of a subgraph. It can be initialized
    with a subset of nodes from a :class:`Graph`. Creating GraphView does not
    change the ownership of the nodes, and so it is possible to create multiple
    GraphViews that contain the same nodes.

    Attributes:
        name: The name of the graph.
        inputs: The input values of the graph.
        outputs: The output values of the graph.
        initializers: The initializers in the graph.
        doc_string: Documentation string.
        opset_imports: Opsets imported by the graph.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    name: str | None
    inputs: Sequence[ValueProtocol]
    outputs: Sequence[ValueProtocol]
    initializers: Mapping[str, ValueProtocol]
    doc_string: str
    opset_imports: Mapping[str, int]
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]

    def __getitem__(self, index: int) -> NodeProtocol: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NodeProtocol]: ...
    def __reversed__(self) -> Iterator[NodeProtocol]: ...


@typing.runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models.

    A model is a container for a graph and metadata. It is the top-level object
    that represents an ONNX model.

    Attributes:
        graph: The graph of the model.
        ir_version: The version of the IR.
        producer_name: The name of the producer.
        producer_version: The version of the producer.
        domain: The domain of the model.
        model_version: The version of the model.
        doc_string: Documentation string.
        functions: The functions defined in the model.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    graph: GraphProtocol
    ir_version: int
    producer_name: str | None
    producer_version: str | None
    domain: str | None
    model_version: int | None
    doc_string: str | None
    functions: MutableMapping[str, FunctionProtocol]
    # TODO(justinchuby): Add training_info
    opset_imports: MutableMapping[str, int]
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]


@typing.runtime_checkable
class AttributeProtocol(Protocol):
    """Protocol for ONNX attributes.

    Attributes:
        name: The name of the attribute.
        type: The type of the attribute.
        value: The value of the attribute.
        doc_string: Documentation string.
    """

    name: str
    type: _enums.AttributeType
    value: Any
    doc_string: str | None


@typing.runtime_checkable
class ReferenceAttributeProtocol(Protocol):
    """Protocol for a reference attribute.

    A reference attribute can only appear inside the definition body of a function.

    Attributes:
        name: The name of the attribute.
        ref_attr_name: The name of the attribute definition this attribute refers to.
        type: The type of the attribute.
        doc_string: Documentation string.
    """

    name: str
    ref_attr_name: str
    type: _enums.AttributeType
    doc_string: str | None


@typing.runtime_checkable
class SparseTensorProtocol(Protocol):
    values: TensorProtocol
    indices: TensorProtocol
    dims: Sequence[int]


@typing.runtime_checkable
class SymbolicDimProtocol(Protocol):
    """Value of a single symbolic/dynamic dimension in a shape.

    Attributes:
        value: The value of the dimension.
    """

    value: str | None  # TODO(justinchuby): Maybe support sympy


@typing.runtime_checkable
class ShapeProtocol(Protocol):
    """Protocol for ONNX shapes.

    A shape is a sequence of dimensions.

    Attributes:
        dims: The dimensions of the shape.
    """

    dims: Sequence[int | SymbolicDimProtocol]

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int | SymbolicDimProtocol]: ...
    @typing.overload
    def __getitem__(self, index: int) -> int | SymbolicDimProtocol: ...
    @typing.overload
    def __getitem__(self, index: slice) -> tuple[int | SymbolicDimProtocol, ...]: ...
    def __setitem__(
        self, index: int, value: int | SymbolicDimProtocol | str | None
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, value: object) -> bool: ...
    def get_denotation(self, index: int) -> str | None: ...
    def set_denotation(self, index: int, denotation: str | None) -> None: ...
    def numpy(self) -> Sequence[int]: ...
    def rank(self) -> int: ...


@typing.runtime_checkable
class TypeProtocol(Protocol):
    """Protocol for ONNX tensors, Sequence tensors, Optional tensors and Sparse tensors.

    These three types of tensors share the same attribute "elem_type" so they are
    merged in the same interface. Unlike the ONNX TensorProto, shapes are not included
    in the type and should be stored in the :class:`Value`.

    Attributes:
        denotation: An optional denotation can be used to denote the whole
            type with a standard semantic description as to what is
            stored inside.
            Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
            for pre-defined type denotations.
        elem_type: The type of its elements for nested types like Sequence[Optional] tensors.
            Or the DataType if the type is not nested.
        dtype: The data type of the tensor or the nested tensor.
    """

    denotation: str | None
    elem_type: TypeProtocol | _enums.DataType
    dtype: _enums.DataType

    def __eq__(self, value: object, /) -> bool: ...


@typing.runtime_checkable
class MapTypeProtocol(Protocol):
    """Protocol for ONNX map types.

    TODO: This protocol is not yet implemented in the ONNX IR.
    """

    key_type: typing.Literal[
        _enums.DataType.STRING,
        _enums.DataType.INT64,
        _enums.DataType.INT32,
        _enums.DataType.INT16,
        _enums.DataType.INT8,
        _enums.DataType.UINT64,
        _enums.DataType.UINT32,
        _enums.DataType.UINT16,
        _enums.DataType.UINT8,
    ]
    value_type: _enums.DataType


@typing.runtime_checkable
class FunctionProtocol(Protocol):
    """Protocol for ONNX functions.

    Like a graph, a function can have nodes that are not topologically sorted. It is
    the responsibility of the user to maintain a topological order of the nodes.

    Note that there is not a ``node`` attribute in the Function. The Function can be
    seen as a Sequence of nodes and should be used as such. For example, to obtain
    all nodes as a list, call ``list(function)``.

    Attributes:
        name: The function name.
        domain: The domain this function is defined in.
        overload: The overload name when the function is overloaded.
        inputs: The input values of the function.
        attributes: The attributes this function defines.
        outputs: The output values of the function.
        opset_imports: Opsets imported by the function.
        doc_string: Documentation string.
        metadata_props: Metadata that will be serialized to the ONNX file.
        meta: Metadata store for graph transform passes.
    """

    name: str
    domain: str
    overload: str
    inputs: Sequence[ValueProtocol]
    attributes: OrderedDict[str, AttributeProtocol]
    outputs: Sequence[ValueProtocol]
    doc_string: str
    opset_imports: MutableMapping[str, int]
    metadata_props: MutableMapping[str, str]
    meta: MutableMapping[str, Any]

    def __getitem__(self, index: int) -> NodeProtocol: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NodeProtocol]: ...
    def __reversed__(self) -> Iterator[NodeProtocol]: ...
    def identifier(self) -> OperatorIdentifier:
        """Return the unique identifier of the function."""
        ...

    # Mutation methods
    # End Block
    def append(self, node: NodeProtocol, /) -> None:
        """Append a node to the function."""
        ...

    def extend(self, nodes: Iterable[NodeProtocol], /) -> None:
        """Extend the function with the given nodes."""
        ...

    def remove(self, node: NodeProtocol, /) -> None:
        """Remove a node from the function."""
        ...

    def insert_after(self, node: NodeProtocol, new_nodes: Iterator[NodeProtocol], /) -> None:
        """Insert new nodes after the given node."""
        ...

    def insert_before(self, node: NodeProtocol, new_nodes: Iterator[NodeProtocol], /) -> None:
        """Insert new nodes before the given node."""
        ...

    def sort(self) -> None:
        """Topologically sort the nodes in the function."""
        ...
