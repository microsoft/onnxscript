"""Protocols for the ONNX IR.

This file defines the interfaces for tools to interact with the IR. The interfaces
are designed such that tools leveraging the IR can be decoupled from the IR
implementation. This allows for the implementation to evolve independently of the
tools.

The file contains two sets of interfaces:
1. Topologically immutable interfaces:
    These interfaces provide a complete view of the ONNX model and allows mutation
    against any metadata fields like shape, type, and node attributes. However, the
    interfaces are topologically immutable, meaning that the structure of the graph
    cannot be changed. This is useful for tools that need to analyze the model
    without modifying how nodes are connected.
2. Mutable interfaces:
    These interfaces provide a mutable view of the ONNX model. They allow for
    modification of the graph structure. This is useful for tools that need to
    transform the model.
"""

from __future__ import annotations

import typing
from typing import (
    AbstractSet,
    Any,
    Iterator,
    Mapping,
    OrderedDict,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from onnxscript.ir import _enums

if typing.TYPE_CHECKING:
    import numpy as np
    from typing_extensions import TypeAlias

SimpleDim: TypeAlias = Union[int, str, None]
SimpleShape: TypeAlias = Sequence[SimpleDim]

# An identifier that will uniquely identify an operator. E.g (domain, op_type, overload)
OperatorIdentifier: TypeAlias = Tuple[str, str, str]


@typing.runtime_checkable
class ArrayCompatible(Protocol):
    """Protocol for array-like objects."""

    def __array__(self, dtype: Any) -> np.ndarray: ...


@typing.runtime_checkable
class DLPackCompatible(Protocol):
    """Protocol objects that can support dlpack."""

    def __dlpack__(self, *, stream: Any = ...) -> Any:
        """Return PyCapsule."""
        ...


@typing.runtime_checkable
class TensorProtocol(ArrayCompatible, Protocol):
    """Concrete tensor backed by data.

    Attributes:
        name: The name of the tensor.
        shape: The shape of the tensor.
        dtype: The data type of the elements of the tensor.
        doc_string: Documentation string.
        raw: The raw data behind this tensor. It can be anything.
        value: The tensor as a numpy array.
    """

    name: str
    shape: ShapeProtocol
    dtype: _enums.DataType
    doc_string: str | None
    raw: Any

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array."""
        ...

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the tensor as a numpy array, compatible with np.array."""
        ...

    def tobytes(self) -> bytes:
        """Return the tensor as a byte string conformed to the ONNX specification, in little endian."""
        ...


@typing.runtime_checkable
class ValueProtocol(Protocol):
    """Protocol for ONNX values.

    A value is a named entity that can be used as an input or output of an operator.

    Attributes:
        name: The name of the value.
        def_node: The node that produces this value.
        def_index: The index of the output of the node that produces this value.
        shape: The shape of the value.
        type: The type of the value.
        metadata_props: Metadata.
    """

    name: str
    def_node: NodeProtocol | None
    def_index: int | None
    shape: ShapeProtocol | None
    type: TypeProtocol | None
    metadata_props: Mapping[str, str]

    def users(self) -> AbstractSet[tuple[NodeProtocol, int]]:
        """The set of (node, input_index) with node being those that use this value as an input."""
        ...

    def is_graph_output(self) -> bool:
        """Whether this value is an output of a graph."""
        ...


@typing.runtime_checkable
class NodeProtocol(Protocol):
    """Protocol for ONNX nodes.

    A node represents an operation in the computation graph.

    Attributes:
        domain: The domain of the operator. E.g. "" for ONNX operators.
        version: The version of the operator.
        op_type: The operator name.
        overload: The overload name when the node is invoking a function.
        inputs: Input values.
        outputs: Output values.
        attributes: The attributes of the operator.
        doc_string: Documentation string.
        metadata_props: Metadata.
    """

    name: str | None
    domain: str
    version: int | None
    op_type: str
    overload: str
    inputs: Sequence[ValueProtocol]
    outputs: Sequence[ValueProtocol]
    attributes: OrderedDict[str, AttributeProtocol | ReferenceAttributeProtocol]
    doc_string: str | None
    metadata_props: Mapping[str, str]


@typing.runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for ONNX graphs.

    Graph represents a computation graph.

    Attributes:
        name: The name of the graph.
        inputs: The input values of the graph.
        outputs: The output values of the graph.
        nodes: All nodes this graph directly owns. They do not have to be sorted.
        initializers: The initializers in the graph.
        doc_string: Documentation string.
        opset_imports: Opsets imported by the graph.
        metadata_props: Metadata.
    """

    # TODO(justinchuby): Support quantization_annotation and metadata_props
    name: str | None
    inputs: Sequence[ValueProtocol]
    outputs: Sequence[ValueProtocol]
    nodes: Sequence[NodeProtocol]
    initializers: Mapping[str, TensorProtocol]
    doc_string: str
    opset_imports: Mapping[str, int]
    metadata_props: Mapping[str, str]

    def topologically_sorted_nodes(self) -> Sequence[NodeProtocol]:
        """Return the nodes in topological order."""
        ...


@typing.runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for ONNX models.

    A model is a container for a graph and metadata.

    Attributes:
        graph: The graph of the model.
        ir_version: The version of the IR.
        producer_name: The name of the producer.
        producer_version: The version of the producer.
        domain: The domain of the model.
        model_version: The version of the model.
        doc_string: Documentation string.
        functions: The functions defined in the model.
        metadata_props: Metadata.
    """

    graph: GraphProtocol
    ir_version: int
    producer_name: str | None
    producer_version: str | None
    domain: str | None
    model_version: int | None
    doc_string: str | None
    functions: Mapping[str, FunctionProtocol]
    # TODO(justinchuby): Add training_info
    opset_imports: Mapping[str, int]
    metadata_props: Mapping[str, str]


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
class DimensionProtocol(Protocol):
    """Value of a single dimension in a shape.

    Attributes:
        value: The value of the dimension.
        denotation: The denotation of the dimension.
            Standard denotation can optionally be used to denote tensor
            dimensions with standard semantic descriptions to ensure
            that operations are applied to the correct axis of a tensor.
            Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
            for pre-defined dimension denotations.
    """

    value: int | str | None
    denotation: str | None


@typing.runtime_checkable
class ShapeProtocol(Protocol):
    """Protocol for ONNX shapes.

    A shape is a sequence of dimensions.

    Attributes:
        dims: The dimensions of the shape.
    """

    dims: Sequence[DimensionProtocol]

    def __iter__(self) -> Iterator[DimensionProtocol]: ...

    def __getitem__(self, index: int) -> DimensionProtocol: ...

    def simple(self) -> SimpleShape: ...

    def numpy(self) -> Sequence[int]: ...


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
            Or None if the type is not nested.
        dtype: The data type of the tensor or the nested tensor.
    """

    denotation: str | None
    elem_type: TypeProtocol | None
    dtype: _enums.DataType

    def __eq__(self, __value: object) -> bool: ...


@typing.runtime_checkable
class MapTypeProtocol(Protocol):
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

    Attributes:
        name: The function name.
        domain: The domain this function is defined in.
        overload: The overload name when the function is overloaded.
        inputs: The input values of the function.
        attributes: The attributes this function defines.
        outputs: The output values of the function.
        opset_imports: Opsets imported by the function.
        doc_string: Documentation string.
        nodes: All nodes this function directly owns. They do not have to be sorted.
        metadata_props: Metadata.
    """

    name: str
    domain: str
    overload: str
    inputs: Sequence[ValueProtocol]
    attributes: OrderedDict[str, AttributeProtocol]
    outputs: Sequence[ValueProtocol]
    doc_string: str
    # opset_import is stored in a model, not a graph. However,
    # In ONNX IR we store it in a graph to unify it with
    # the function. This way a materialized function can still
    # be used as a subgraph even if it imports a different opset.
    opset_imports: Mapping[str, int]
    nodes: Sequence[NodeProtocol]
    metadata_props: Mapping[str, str]

    def identifier(self) -> OperatorIdentifier:
        """Return the unique identifier of the function."""
        ...

    def topologically_sorted_nodes(self) -> Sequence[NodeProtocol]:
        """Return the nodes in topological order."""
        ...
