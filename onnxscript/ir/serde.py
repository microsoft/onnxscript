# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Serialize and deserialize the intermediate representation to/from ONNX protos."""

# NOTES for developers:
# NOTE: Do not import pathlib in the IR. It is slow. Use os.path methods instead.
#
# NOTE: Protobuf serialization
#     Initializing a protobuf message with initialized protobuf messages incurs
#     a copy and is slow. Instead, use proto.add() to add to a repeated field.
#     or initialize the message first and then set the fields if the fields are
#     plain Python objects.

from __future__ import annotations

import functools

__all__ = [
    # Tensors
    "TensorProtoTensor",
    # Deserialization
    "from_proto",
    "deserialize_attribute",
    "deserialize_dimension",
    "deserialize_function",
    "deserialize_graph",
    "deserialize_metadata_props",
    "deserialize_model",
    "deserialize_node",
    "deserialize_opset_import",
    "deserialize_tensor",
    "deserialize_type_proto_for_shape",
    "deserialize_type_proto_for_type",
    "deserialize_value_info_proto",
    # Serialization
    "to_proto",
    "serialize_attribute_into",
    "serialize_attribute",
    "serialize_dimension_into",
    "serialize_function_into",
    "serialize_function",
    "serialize_graph_into",
    "serialize_graph",
    "serialize_model_into",
    "serialize_model",
    "serialize_node_into",
    "serialize_node",
    "serialize_shape_into",
    "serialize_reference_attribute_into",
    "serialize_tensor_into",
    "serialize_tensor",
    "serialize_type_into",
    "serialize_type",
    "serialize_value_into",
    "serialize_value",
    "SerdeError",
]

import collections
import logging
import os
import typing
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
import onnx
import onnx.external_data_helper

from onnxscript.ir import _core, _enums, _metadata, _protocols, _type_casting

if typing.TYPE_CHECKING:
    import google.protobuf.internal.containers as proto_containers
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_PLEASE_CONTRIBUTE = (
    "Please contribute by creating a PR at https://github.com/microsoft/onnxscript."
)
_FUNCTION_VALUE_INFO_SUPPORTED_VERSION = (
    10  # ONNX IR version where value info in functions was introduced
)
_T = typing.TypeVar("_T", bound=Callable[..., Any])


class SerdeError(RuntimeError):
    """Error during serialization or deserialization."""


def _capture_errors(arg_capturer: Callable[..., str]) -> Callable[[_T], _T]:
    """Decorator to capture errors and display the stack."""

    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise SerdeError(
                    f"Error calling {func.__name__} with: {arg_capturer(*args, **kwargs)}"
                ) from e

        return wrapper  # type: ignore

    return decorator


def _little_endian_dtype(dtype) -> np.dtype:
    """Create a small endian dtype on all platforms.

    This is useful because ONNX always stores raw_data in small endian. On big
    endian platforms, we still need to interpret the raw_data in small endian.
    """
    return np.dtype(dtype).newbyteorder("<")


def _unflatten_complex(
    array: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.complex64 | np.complex128]:
    """Convert the real representation of a complex dtype to the complex dtype."""
    return array[::2] + 1j * array[1::2]


def from_proto(
    proto: onnx.ModelProto
    | onnx.GraphProto
    | onnx.NodeProto
    | onnx.TensorProto
    | onnx.AttributeProto
    | onnx.ValueInfoProto
    | onnx.TypeProto
    | onnx.FunctionProto,
) -> Any:
    """Deserialize an ONNX proto message to an IR object."""
    if isinstance(proto, onnx.ModelProto):
        return deserialize_model(proto)
    if isinstance(proto, onnx.GraphProto):
        return deserialize_graph(proto)
    if isinstance(proto, onnx.NodeProto):
        return deserialize_node(proto)
    if isinstance(proto, onnx.TensorProto):
        return deserialize_tensor(proto)
    if isinstance(proto, onnx.AttributeProto):
        return deserialize_attribute(proto)
    if isinstance(proto, onnx.ValueInfoProto):
        return deserialize_value_info_proto(proto, None)
    if isinstance(proto, onnx.TypeProto):
        return _core.TypeAndShape(
            deserialize_type_proto_for_type(proto),
            deserialize_type_proto_for_shape(proto),
        )
    if isinstance(proto, onnx.FunctionProto):
        return deserialize_function(proto)
    raise NotImplementedError(
        f"Deserialization of {type(proto)} in from_proto is not implemented. "
        "Use a specific ir.serde.deserialize* function instead."
    )


def to_proto(
    ir_object: _protocols.ModelProtocol
    | _protocols.GraphProtocol
    | _protocols.NodeProtocol
    | _protocols.ValueProtocol
    | _protocols.AttributeProtocol
    | _protocols.ReferenceAttributeProtocol
    | _protocols.TensorProtocol
    | _protocols.TypeProtocol
    | _protocols.GraphViewProtocol
    | _protocols.FunctionProtocol,
) -> Any:
    """Serialize an IR object to a proto."""
    if isinstance(ir_object, _protocols.ModelProtocol):
        return serialize_model(ir_object)
    if isinstance(ir_object, _protocols.GraphProtocol):
        return serialize_graph(ir_object)
    if isinstance(ir_object, _protocols.NodeProtocol):
        return serialize_node(ir_object)
    if isinstance(ir_object, _protocols.TensorProtocol):
        return serialize_tensor(ir_object)
    if isinstance(ir_object, _protocols.ValueProtocol):
        return serialize_value(ir_object)
    if isinstance(ir_object, _protocols.AttributeProtocol):
        return serialize_attribute(ir_object)
    if isinstance(ir_object, _protocols.ReferenceAttributeProtocol):
        return serialize_reference_attribute_into(onnx.AttributeProto(), ir_object)
    if isinstance(ir_object, _protocols.TypeProtocol):
        return serialize_type_into(onnx.TypeProto(), ir_object)
    if isinstance(ir_object, _protocols.GraphViewProtocol):
        return serialize_graph(ir_object)
    if isinstance(ir_object, _protocols.FunctionProtocol):
        return serialize_function(ir_object)
    raise NotImplementedError(
        f"Serialization of {type(ir_object)} in to_proto is not implemented. "
        "Use a specific ir.serde.serialize* function instead."
    )


class TensorProtoTensor(_core.TensorBase):  # pylint: disable=too-many-ancestors
    """A tensor initialized from a tensor proto."""

    def __init__(self, proto: onnx.TensorProto) -> None:
        self._proto = proto
        self._metadata_props: dict[str, str] | None = deserialize_metadata_props(
            proto.metadata_props
        )
        self._metadata: _metadata.MetadataStore | None = None

    @property
    def name(self) -> str:
        return self._proto.name

    @name.setter
    def name(self, value: str | None) -> None:
        if value is None:
            self._proto.ClearField("name")
        else:
            self._proto.name = value

    @property
    def shape(self) -> _core.Shape:
        return _core.Shape(self._proto.dims, frozen=True)

    @property
    def dtype(self) -> _enums.DataType:
        return _enums.DataType(self._proto.data_type)

    @property
    def doc_string(self) -> str:
        return self._proto.doc_string

    @property
    def raw(self) -> onnx.TensorProto:
        return self._proto

    def __repr__(self) -> str:
        # It is a little hard to display the content when there can be types
        # unsupported by numpy
        # Preferably we should display some content when the tensor is small
        return f"{self._repr_base()}(name={self.name!r})"

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the tensor as a numpy array, compatible with np.array."""
        return self.numpy().__array__(dtype)

    def __dlpack__(self, *, stream: Any = None) -> Any:
        return self.numpy().__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.numpy().__dlpack_device__()

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array.

        This is an improved version of onnx.numpy_helper.to_array.
        It first reads the data using the dtype corresponding to the tensor
        proto data field, then converts it to the correct dtype and shape.
        Special cases are bfloat16, complex and int4 where we need to
        reinterpret the data. Other types can simply be casted.

        When the data type is not supported by numpy, the dtypes from the ``ml_dtype``
        package are used. The values can be reinterpreted as bit representations
        using the ``.view()`` method.

        When the data type is a string, this method returns a numpy array
        of bytes instead of a numpy array of strings, to follow the ONNX
        specification.

        External tensors are not supported by this class. Use
        :class:`onnxscript.ir.ExternalTensor` instead.

        Raises:
            ValueError: If the data type is UNDEFINED.
        """
        dtype = self.dtype
        if dtype == _enums.DataType.UNDEFINED:
            raise ValueError("Cannot convert UNDEFINED tensor to numpy array.")
        if self._proto.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(
                "Cannot convert external tensor to numpy array. "
                "Use ir.ExternalTensor instead."
            )

        if self._proto.HasField("raw_data"):
            array = np.frombuffer(self._proto.raw_data, dtype=dtype.numpy().newbyteorder("<"))
            # Cannot return now, because we may need to unpack 4bit tensors
        elif dtype == _enums.DataType.STRING:
            return np.array(self._proto.string_data).reshape(self._proto.dims)
        elif self._proto.int32_data:
            array = np.array(self._proto.int32_data, dtype=_little_endian_dtype(np.int32))
            if dtype in {_enums.DataType.FLOAT16, _enums.DataType.BFLOAT16}:
                # Reinterpret the int32 as float16 or bfloat16
                array = array.astype(np.uint16).view(dtype.numpy())
            elif dtype in {
                _enums.DataType.FLOAT8E4M3FN,
                _enums.DataType.FLOAT8E4M3FNUZ,
                _enums.DataType.FLOAT8E5M2,
                _enums.DataType.FLOAT8E5M2FNUZ,
            }:
                array = array.astype(np.uint8).view(dtype.numpy())
        elif self._proto.int64_data:
            array = np.array(self._proto.int64_data, dtype=_little_endian_dtype(np.int64))
        elif self._proto.uint64_data:
            array = np.array(self._proto.uint64_data, dtype=_little_endian_dtype(np.uint64))
        elif self._proto.float_data:
            array = np.array(self._proto.float_data, dtype=_little_endian_dtype(np.float32))
            if dtype == _enums.DataType.COMPLEX64:
                array = _unflatten_complex(array)
        elif self._proto.double_data:
            array = np.array(self._proto.double_data, dtype=_little_endian_dtype(np.float64))
            if dtype == _enums.DataType.COMPLEX128:
                array = _unflatten_complex(array)
        else:
            # Empty tensor
            if not self._proto.dims:
                # When dims not precent and there is no data, we return an empty array
                return np.array([], dtype=dtype.numpy())
            else:
                # Otherwise we return a size 0 array with the correct shape
                return np.zeros(self._proto.dims, dtype=dtype.numpy())

        if dtype == _enums.DataType.INT4:
            return _type_casting.unpack_int4(array.astype(np.uint8), self._proto.dims)
        elif dtype == _enums.DataType.UINT4:
            return _type_casting.unpack_uint4(array.astype(np.uint8), self._proto.dims)
        else:
            # Otherwise convert to the correct dtype and reshape
            # Note we cannot use view() here because the storage dtype may not be the same size as the target
            return array.astype(dtype.numpy()).reshape(self._proto.dims)

    def tobytes(self) -> bytes:
        """Return the tensor as a byte string conformed to the ONNX specification, in little endian.

        Raises:
            ValueError: If the tensor is a string tensor or an external tensor.
            ValueError: If the tensor is of UNDEFINED data type.
        """
        if self._proto.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(
                "Cannot convert external tensor to bytes. Use ir.ExternalTensor instead."
            )
        if self.dtype == _enums.DataType.STRING:
            raise ValueError("Cannot convert string tensor to bytes.")
        if self.dtype == _enums.DataType.UNDEFINED:
            raise ValueError("Cannot convert UNDEFINED tensor to bytes.")

        if self._proto.HasField("raw_data"):
            return self._proto.raw_data
        if self._proto.float_data:
            return np.array(
                self._proto.float_data, dtype=_little_endian_dtype(np.float32)
            ).tobytes()
        if self._proto.int32_data:
            array = np.array(self._proto.int32_data, dtype=np.int32)
            if self.dtype in {
                _enums.DataType.INT16,
                _enums.DataType.UINT16,
                _enums.DataType.FLOAT16,
                _enums.DataType.BFLOAT16,
            }:
                return array.astype(_little_endian_dtype(np.uint16)).tobytes()
            if self.dtype in {
                _enums.DataType.INT8,
                _enums.DataType.UINT8,
                _enums.DataType.BOOL,
                _enums.DataType.FLOAT8E4M3FN,
                _enums.DataType.FLOAT8E4M3FNUZ,
                _enums.DataType.FLOAT8E5M2,
                _enums.DataType.FLOAT8E5M2FNUZ,
                _enums.DataType.INT4,
                _enums.DataType.UINT4,
            }:
                # uint4 and int4 values are already packed, even when stored as int32
                # so we don't need to pack them again
                return array.astype(_little_endian_dtype(np.uint8)).tobytes()
            assert self.dtype == _enums.DataType.INT32
            return array.tobytes()
        if self._proto.int64_data:
            return np.array(
                self._proto.int64_data, dtype=_little_endian_dtype(np.int64)
            ).tobytes()
        if self._proto.double_data:
            return np.array(
                self._proto.double_data, dtype=_little_endian_dtype(np.float64)
            ).tobytes()
        if self._proto.uint64_data:
            array = np.array(self._proto.uint64_data, dtype=_little_endian_dtype(np.uint64))
            if self.dtype == _enums.DataType.UINT32:
                return array.astype(_little_endian_dtype(np.uint32)).tobytes()
            assert self.dtype == _enums.DataType.UINT64
            return array.tobytes()
        # The repeating fields can be empty and still valid.
        # For example, int32_data can be empty and still be a valid tensor.
        return b""

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attr:`metadata_props` if you would like the metadata to be serialized
        to the ONNX proto.
        """
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props


def _get_field(proto: Any, field: str) -> Any:
    if proto.HasField(field):
        return getattr(proto, field)
    return None


# Deserialization


def deserialize_opset_import(
    protos: Sequence[onnx.OperatorSetIdProto],
) -> dict[str, int]:
    return {opset.domain: opset.version for opset in protos}


def _parse_experimental_function_value_info_name(
    name: str,
) -> tuple[str, str, str] | None:
    """Get the function domain, name and value name if the value info is for a function.

    The experimental format is:
    {function_domain}::{function_name}/{value_name}

    Args:
        name: The name stored in the value info.

    Returns:
        A tuple of the function domain, function name and value name if the value info is for a function.
        None otherwise.
    """
    parts = name.split("/")
    expected_parts = 2
    if len(parts) != expected_parts:
        return None
    function, value_name = parts
    parts = function.split("::")
    if len(parts) != expected_parts:
        return None
    # NOTE: There will not be overload because overloads are introduced in ONNX IR v10, which also
    # introduces the ValueInfoProto for functions
    function_domain, function_name = parts
    return function_domain, function_name, value_name


def deserialize_model(proto: onnx.ModelProto) -> _core.Model:
    graph = _deserialize_graph(proto.graph, [])
    graph.opset_imports.update(deserialize_opset_import(proto.opset_import))

    functions = []
    for func in proto.functions:
        functions.append(deserialize_function(func))

    model = _core.Model(
        graph,
        ir_version=proto.ir_version,
        producer_name=_get_field(proto, "producer_name"),
        producer_version=_get_field(proto, "producer_version"),
        domain=_get_field(proto, "domain"),
        model_version=_get_field(proto, "model_version"),
        doc_string=_get_field(proto, "doc_string"),
        functions=functions,
        meta_data_props=deserialize_metadata_props(proto.metadata_props),
    )

    # Handle experimental value info for functions created by the dynamo exporter in IR version 9
    if model.ir_version < _FUNCTION_VALUE_INFO_SUPPORTED_VERSION:
        _deserialized_experimental_value_info_for_function_ir9(
            model.functions, proto.graph.value_info
        )

    return model


def _deserialized_experimental_value_info_for_function_ir9(
    functions: Mapping[_protocols.OperatorIdentifier, _core.Function],
    value_info_protos: Sequence[onnx.ValueInfoProto],
) -> None:
    """Deserialize value info for functions when they are stored in an experimental format.

    The experimental format is:
    {function_domain}::{function_name}/{value_name}
    """
    # Parse value info for functions from the main graph
    function_value_value_info_mapping: collections.defaultdict[
        _protocols.OperatorIdentifier,
        dict[str, onnx.ValueInfoProto],
    ] = collections.defaultdict(dict)
    for value_info_proto in value_info_protos:
        if (
            parsed := _parse_experimental_function_value_info_name(value_info_proto.name)
        ) is None:
            continue
        function_domain, function_name, value_name = parsed
        function_overload = ""
        # TODO(justinchuby): Create a constructor for OperatorIdentifier so we don't create tuples manually
        function_id = (function_domain, function_name, function_overload)
        function = functions.get(function_id)
        if function is None:
            # Function not found
            logger.debug(
                "Function with ID '%s' not found in model functions. Value info '%s' will be ignored.",
                function_id,
                value_info_proto.name,
            )
            continue
        function_value_value_info_mapping[function_id][value_name] = value_info_proto
    for function_id, function in functions.items():
        for input in function.inputs:
            if input.name in function_value_value_info_mapping[function_id]:
                deserialize_value_info_proto(
                    function_value_value_info_mapping[function_id][input.name], input
                )
        for node in function:
            for output in node.outputs:
                if output.name in function_value_value_info_mapping[function_id]:
                    deserialize_value_info_proto(
                        function_value_value_info_mapping[function_id][output.name],
                        output,
                    )
            # The function outputs are handled as well because they are also node outputs


def deserialize_graph(proto: onnx.GraphProto) -> _core.Graph:
    """Deserialize a graph proto, recursively if needed.

    Args:
        proto: The graph proto to deserialize.

    Returns:
        IR Graph.
    """
    return _deserialize_graph(proto, [])


@_capture_errors(lambda proto, scoped_values: proto.name)
def _deserialize_graph(
    proto: onnx.GraphProto, scoped_values: list[dict[str, _core.Value]]
) -> _core.Graph:
    """Deserialize a graph proto, recursively if needed.

    Args:
        proto: The graph proto to deserialize.
        scoped_values: A list of dictionaries mapping value names to their corresponding Value objects.
            Every time we enter a new graph, a new scope is created and appended to this list to include
            all values defined in the scope.
        scoped_value_info: A list of dictionaries mapping value names to their corresponding ValueInfoProto.

    Returns:
        IR Graph.
    """
    # Create values for initializers and inputs
    initializer_tensors = [deserialize_tensor(tensor) for tensor in proto.initializer]
    inputs = [_core.Input(info.name) for info in proto.input]
    for info, value in zip(proto.input, inputs):
        deserialize_value_info_proto(info, value)

    # Initialize the values dictionary for this graph scope with the inputs and initializers
    values: dict[str, _core.Value] = {v.name: v for v in inputs}  # type: ignore[misc]
    scoped_values.append(values)
    initializer_values = []
    for tensor in initializer_tensors:
        if tensor.name in values:
            # The initializer is for an input
            initializer_value = values[tensor.name]
            initializer_value.const_value = tensor
        else:
            # The initializer is for some other value. Create this value first
            initializer_value = _core.Value(
                None,
                index=None,
                name=tensor.name,
                # TODO(justinchuby): Fix type hinting for shape and dtype
                shape=tensor.shape,  # type: ignore
                type=_core.TensorType(tensor.dtype),
                const_value=tensor,
            )
            values[tensor.name] = initializer_value  # type: ignore[index]
        initializer_values.append(initializer_value)

    # Add ValueInfos for this graph scope
    value_info = {info.name: info for info in proto.value_info}

    # Deserialize nodes with all known values
    nodes = [_deserialize_node(node, scoped_values, value_info) for node in proto.node]

    # Fill in values for graph outputs
    outputs = [deserialize_value_info_proto(info, values[info.name]) for info in proto.output]
    scoped_values.pop()
    return _core.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializer_values,
        doc_string=_get_field(proto, "doc_string"),
        name=_get_field(proto, "name"),
        metadata_props=deserialize_metadata_props(proto.metadata_props),
    )


@_capture_errors(lambda proto: proto.name)
def deserialize_function(proto: onnx.FunctionProto) -> _core.Function:
    inputs = [_core.Input(name) for name in proto.input]
    values: dict[str, _core.Value] = {v.name: v for v in inputs}  # type: ignore[misc]
    value_info = {info.name: info for info in getattr(proto, "value_info", [])}

    # TODO(justinchuby): Handle unsorted nodes
    nodes = [_deserialize_node(node, [values], value_info=value_info) for node in proto.node]
    outputs = [values[name] for name in proto.output]
    graph = _core.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=(),
        doc_string=_get_field(proto, "doc_string"),
        opset_imports=deserialize_opset_import(proto.opset_import),
        name=(
            f"{proto.name}_{proto.domain}" + f"__{proto.overload}"
            if hasattr(proto, "overload") and proto.overload
            else ""
        ),
    )
    attributes = [_deserialize_attribute(attr, []) for attr in proto.attribute_proto]
    # Attributes without defaults
    attributes += [
        _core.Attr(name, _enums.AttributeType.UNDEFINED, None) for name in proto.attribute
    ]
    return _core.Function(
        domain=proto.domain,
        name=proto.name,
        overload=getattr(proto, "overload", ""),
        graph=graph,
        attributes=typing.cast(List[_core.Attr], attributes),
        metadata_props=deserialize_metadata_props(proto.metadata_props),
    )


@_capture_errors(lambda proto, value: str(proto))
def deserialize_value_info_proto(
    proto: onnx.ValueInfoProto, value: _core.Value | None
) -> _core.Value:
    if value is None:
        value = _core.Value(name=proto.name)
    value.shape = deserialize_type_proto_for_shape(proto.type)
    value.type = deserialize_type_proto_for_type(proto.type)
    metadata_props = deserialize_metadata_props(proto.metadata_props)
    if metadata_props is not None:
        value.metadata_props.update(metadata_props)
    value.doc_string = _get_field(proto, "doc_string")
    return value


@_capture_errors(str)
def deserialize_type_proto_for_shape(proto: onnx.TypeProto) -> _core.Shape | None:
    if proto.HasField("tensor_type"):
        if (shape_proto := _get_field(proto.tensor_type, "shape")) is None:
            return None
        # This logic handles when the shape is [] as well
        dim_protos = shape_proto.dim
        deserialized_dim_denotations = [
            deserialize_dimension(dim_proto) for dim_proto in dim_protos
        ]
        dims = [dim for dim, _ in deserialized_dim_denotations]
        denotations = [denotation for _, denotation in deserialized_dim_denotations]
        return _core.Shape(dims, denotations=denotations, frozen=True)
    if proto.HasField("sparse_tensor_type"):
        if (shape_proto := _get_field(proto.sparse_tensor_type, "shape")) is None:
            return None
        dim_protos = shape_proto.dim
        deserialized_dim_denotations = [
            deserialize_dimension(dim_proto) for dim_proto in dim_protos
        ]
        dims = [dim for dim, _ in deserialized_dim_denotations]
        denotations = [denotation for _, denotation in deserialized_dim_denotations]
        return _core.Shape(dims, denotations=denotations, frozen=True)
    if proto.HasField("sequence_type"):
        if (elem_type := _get_field(proto.sequence_type, "elem_type")) is None:
            return None
        return deserialize_type_proto_for_shape(elem_type)
    if proto.HasField("optional_type"):
        if (elem_type := _get_field(proto.optional_type, "elem_type")) is None:
            return None
        return deserialize_type_proto_for_shape(elem_type)
    if proto.HasField("map_type"):
        # TODO(justinchuby): Do we need to support map types?
        raise NotImplementedError(f"Map types are not supported yet. {_PLEASE_CONTRIBUTE}")

    return None


@_capture_errors(str)
def deserialize_type_proto_for_type(
    proto: onnx.TypeProto,
) -> _protocols.TypeProtocol | None:
    denotation = _get_field(proto, "denotation")
    if proto.HasField("tensor_type"):
        if (elem_type := _get_field(proto.tensor_type, "elem_type")) is None:
            return None
        return _core.TensorType(_enums.DataType(elem_type), denotation=denotation)
    if proto.HasField("sparse_tensor_type"):
        if (elem_type := _get_field(proto.sparse_tensor_type, "elem_type")) is None:
            return None
        return _core.SparseTensorType(_enums.DataType(elem_type), denotation=denotation)
    if proto.HasField("sequence_type"):
        # FIXME(justinchuby): Allow nested types being None
        if (elem_type := _get_field(proto.sequence_type, "elem_type")) is None:
            raise ValueError(f"SequenceTypeProto must have elem_type set: {proto}")
        nested_type = deserialize_type_proto_for_type(elem_type)
        if nested_type is None:
            raise ValueError(f"SequenceType must have elem_type set: {proto}")
        return _core.SequenceType(nested_type, denotation=denotation)
    if proto.HasField("optional_type"):
        # FIXME(justinchuby): Allow nested types being None
        if (elem_type := _get_field(proto.optional_type, "elem_type")) is None:
            raise ValueError(f"SequenceTypeProto must have elem_type set: {proto}")
        nested_type = deserialize_type_proto_for_type(elem_type)
        if nested_type is None:
            raise ValueError(f"SequenceType must have elem_type set: {proto}")
        return _core.OptionalType(nested_type, denotation=denotation)
    if proto.HasField("map_type"):
        # TODO(justinchuby): Do we need to support map types?
        raise NotImplementedError(f"Map types are not supported yet. {_PLEASE_CONTRIBUTE}")

    return None


@_capture_errors(str)
def deserialize_dimension(
    proto: onnx.TensorShapeProto.Dimension,
) -> tuple[int | _core.SymbolicDim, str | None]:
    """Deserialize a dimension proto into (dimension, denotation).

    Args:
        proto: The dimension proto to deserialize.

    Returns:
        A tuple of the dimension and its denotation.
    """
    value_field = proto.WhichOneof("value")
    denotation = _get_field(proto, "denotation")
    if value_field is not None:
        value = getattr(proto, value_field)
        if value_field == "dim_value":
            return value, denotation
        if value_field == "dim_param":
            return _core.SymbolicDim(value), denotation
    return _core.SymbolicDim(None), denotation


@_capture_errors(lambda proto, base_path: proto.name)
def deserialize_tensor(
    proto: onnx.TensorProto, base_path: str | os.PathLike = ""
) -> _protocols.TensorProtocol:
    # TODO: Sanitize base_path
    if proto.data_location == onnx.TensorProto.EXTERNAL:
        external_info = onnx.external_data_helper.ExternalDataInfo(proto)
        return _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=_enums.DataType(proto.data_type),
            base_dir=base_path,
            name=_get_field(proto, "name"),
            shape=_core.Shape(proto.dims),
            doc_string=_get_field(proto, "doc_string"),
            metadata_props=deserialize_metadata_props(proto.metadata_props),
        )
    if proto.data_type == _enums.DataType.STRING:
        name = _get_field(proto, "name")
        doc_string = _get_field(proto, "doc_string")
        metadata_props = deserialize_metadata_props(proto.metadata_props)
        return _core.StringTensor(
            proto.string_data,
            shape=_core.Shape(proto.dims),
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
    return TensorProtoTensor(proto)


def deserialize_metadata_props(
    proto: Sequence[onnx.StringStringEntryProto],
) -> dict[str, str] | None:
    if len(proto) == 0:
        # Avoid creating an empty dictionary to save memory
        return None
    return {entry.key: entry.value for entry in proto}


def deserialize_attribute(proto: onnx.AttributeProto) -> _core.Attr | _core.RefAttr:
    return _deserialize_attribute(proto, [])


@_capture_errors(lambda proto, scoped_values: str(proto))
def _deserialize_attribute(
    proto: onnx.AttributeProto, scoped_values: list[dict[str, _core.Value]]
) -> _core.Attr | _core.RefAttr:
    name = proto.name
    doc_string = _get_field(proto, "doc_string")
    type_ = _enums.AttributeType(proto.type)
    ref_attr_name = _get_field(proto, "ref_attr_name")
    if ref_attr_name:
        return _core.RefAttr(name, ref_attr_name, type_, doc_string=doc_string)

    if type_ == _enums.AttributeType.INT:
        return _core.AttrInt64(name, proto.i, doc_string=doc_string)
    if type_ == _enums.AttributeType.FLOAT:
        return _core.AttrFloat32(name, proto.f, doc_string=doc_string)
    if type_ == _enums.AttributeType.STRING:
        return _core.AttrString(name, proto.s.decode("utf-8"), doc_string=doc_string)
    if type_ == _enums.AttributeType.INTS:
        return _core.AttrInt64s(name, proto.ints, doc_string=doc_string)
    if type_ == _enums.AttributeType.FLOATS:
        return _core.AttrFloat32s(name, proto.floats, doc_string=doc_string)
    if type_ == _enums.AttributeType.STRINGS:
        return _core.AttrStrings(
            name, [s.decode("utf-8") for s in proto.strings], doc_string=doc_string
        )
    if type_ == _enums.AttributeType.TENSOR:
        return _core.AttrTensor(name, deserialize_tensor(proto.t), doc_string=doc_string)
    if type_ == _enums.AttributeType.GRAPH:
        return _core.AttrGraph(
            name, _deserialize_graph(proto.g, scoped_values), doc_string=doc_string
        )
    if type_ == _enums.AttributeType.TENSORS:
        return _core.AttrTensors(
            name,
            [deserialize_tensor(t) for t in proto.tensors],
            doc_string=doc_string,
        )
    if type_ == _enums.AttributeType.GRAPHS:
        return _core.AttrGraphs(
            name,
            [_deserialize_graph(g, scoped_values) for g in proto.graphs],
            doc_string=doc_string,
        )
    if type_ == _enums.AttributeType.SPARSE_TENSOR:
        raise NotImplementedError(
            f"Sparse tensors are not supported yet. {_PLEASE_CONTRIBUTE}"
        )
    if type_ == _enums.AttributeType.SPARSE_TENSORS:
        raise NotImplementedError(
            f"Sparse tensors are not supported yet. {_PLEASE_CONTRIBUTE}"
        )
    if type_ == _enums.AttributeType.TYPE_PROTO:
        ir_type = deserialize_type_proto_for_type(proto.tp)
        shape = deserialize_type_proto_for_shape(proto.tp)
        return _core.AttrTypeProto(
            name, _core.TypeAndShape(ir_type, shape), doc_string=doc_string
        )
    if type_ == _enums.AttributeType.TYPE_PROTOS:
        type_and_shapes = []
        for type_proto in proto.type_protos:
            ir_type = deserialize_type_proto_for_type(type_proto)
            shape = deserialize_type_proto_for_shape(type_proto)
            type_and_shapes.append(_core.TypeAndShape(ir_type, shape))
        return _core.AttrTypeProtos(name, type_and_shapes, doc_string=doc_string)
    if type_ == _enums.AttributeType.UNDEFINED:
        return _core.Attr(name, type_, None, doc_string=doc_string)
    raise ValueError(f"Unsupported attribute type: '{type_}'")


def deserialize_node(proto: onnx.NodeProto) -> _core.Node:
    return _deserialize_node(proto, scoped_values=[], value_info={})


@_capture_errors(lambda proto, scoped_values, value_info: str(proto))
def _deserialize_node(
    proto: onnx.NodeProto,
    scoped_values: list[dict[str, _core.Value]],
    value_info: dict[str, onnx.ValueInfoProto],
) -> _core.Node:
    node_inputs: list[_core.Value | None] = []
    for input_name in proto.input:
        if input_name == "":
            # Empty input
            node_inputs.append(None)
            continue

        # Find the input in all value scopes
        found = False
        for values in reversed(scoped_values):
            if input_name not in values:
                continue
            node_inputs.append(values[input_name])
            found = True
            del values  # Remove the reference so it is not used by mistake
            break
        if not found:
            # If the input is not found, we know the graph may be unsorted and
            # the input may be a supposed-to-be initializer or an output of a node that comes later.
            # Here we create the value with the name and add it to the current scope.
            # Nodes need to check the value pool for potentially initialized outputs
            logger.warning(
                "Input '%s' of node '%s(%s::%s:%s)' not found in any scope. "
                "The graph may be unsorted. Creating a new input (current depth: %s) .",
                input_name,
                proto.name,
                proto.domain,
                proto.op_type,
                getattr(proto, "overload", ""),
                len(scoped_values),
            )
            if len(scoped_values) > 1:
                logger.warning(
                    "Caveat: The value is created in the subgraph. If "
                    "the node is referencing a value that is not in the current graph, "
                    "it is impossible to create it in the correct scope.",
                )
            value = _core.Value(name=input_name)
            # Fill in shape/type information if they exist
            if input_name in value_info:
                deserialize_value_info_proto(value_info[input_name], value)
            node_inputs.append(value)
            # We can only create the value in the current scope. If the subgraph is
            # referencing a value that is not in the current scope, it is impossible
            # to create it in the correct scope.
            scoped_values[-1][input_name] = value

    # Build the output values for the node.
    node_outputs: list[_core.Value] = []
    for output_name in proto.output:
        if output_name == "":
            # Empty output
            node_outputs.append(_core.Value(name=""))
            continue

        # 1. When the graph is unsorted, we may be able to find the output already created
        # as an input to some other nodes in the current scope.
        # Note that a value is always owned by the producing node. Even though a value
        # can be created when parsing inputs of other nodes, the new node created here
        # that produces the value will assume ownership. It is then impossible to transfer
        # the ownership to any other node.

        # The output can only be found in the current scope. It is impossible for
        # a node to produce an output that is not in its own scope.
        current_scope = scoped_values[-1]
        if output_name in current_scope:
            value = current_scope[output_name]
        else:
            # 2. Common scenario: the graph is sorted and this is the first time we see the output.
            # Create the value and add it to the current scope.
            value = _core.Value(name=output_name)
            current_scope[output_name] = value
        # Fill in shape/type information if they exist
        if output_name in value_info:
            deserialize_value_info_proto(value_info[output_name], value)
        else:
            logger.debug(
                "ValueInfoProto not found for output '%s' in node '%s' of type '%s'",
                output_name,
                proto.name,
                proto.op_type,
            )
        node_outputs.append(value)
    return _core.Node(
        proto.domain,
        proto.op_type,
        node_inputs,
        [_deserialize_attribute(a, scoped_values) for a in proto.attribute],
        overload=getattr(proto, "overload", ""),
        outputs=node_outputs,
        name=proto.name,
        doc_string=_get_field(proto, "doc_string"),
        metadata_props=deserialize_metadata_props(proto.metadata_props),
    )


# Serialization


def serialize_model(model: _protocols.ModelProtocol) -> onnx.ModelProto:
    return serialize_model_into(onnx.ModelProto(), from_=model)


@_capture_errors(
    lambda model_proto, from_: (
        f"ir_version={from_.ir_version}, producer_name={from_.producer_name}, "
        f"producer_version={from_.producer_version}, domain={from_.domain}, "
    )
)
def serialize_model_into(
    model_proto: onnx.ModelProto, from_: _protocols.ModelProtocol
) -> onnx.ModelProto:
    """Serialize an IR model to an ONNX model proto."""
    model_proto.ir_version = from_.ir_version
    if from_.producer_name:
        model_proto.producer_name = from_.producer_name
    if from_.producer_version:
        model_proto.producer_version = from_.producer_version
    if from_.domain:
        model_proto.domain = from_.domain
    if from_.model_version:
        model_proto.model_version = from_.model_version
    if from_.doc_string:
        model_proto.doc_string = from_.doc_string
    # Sort names for deterministic serialization
    _serialize_opset_imports_into(model_proto.opset_import, from_.opset_imports)
    if from_.metadata_props:
        _serialize_metadata_props_into(model_proto.metadata_props, from_.metadata_props)
    serialize_graph_into(model_proto.graph, from_.graph)

    create_value_info_in_functions = from_.ir_version >= _FUNCTION_VALUE_INFO_SUPPORTED_VERSION
    for func in from_.functions.values():
        serialize_function_into(
            model_proto.functions.add(),
            from_=func,
            create_value_info=create_value_info_in_functions,
        )
        if not create_value_info_in_functions:
            # Create them in the main graph instead
            _serialize_experimental_value_info_for_function_ir9_into(model_proto.graph, func)
    return model_proto


def _should_create_value_info_for_value(value: _protocols.ValueProtocol) -> bool:
    """Check if value info should be created for a value.

    Args:
        value: The value to check.

    Returns:
        True if value info should be created for the value.
    """
    # No need to serialize value info if it is not set
    return not (value.shape is None and value.type is None)


def _serialize_experimental_value_info_for_function_ir9_into(
    graph_proto: onnx.GraphProto, function: _protocols.FunctionProtocol
) -> None:
    """Serialize value info for functions in an experimental format for IR version 9.

    Because IRv9 and older does not have ValueInfoProto for functions, we give the value info
    special names and store them in the main graph instead.

    The experimental format is:
    {function_domain}::{function_name}/{value_name}

    Args:
        graph_proto: The graph proto to create ValueInfoProto in.
        function: The function to serialize.
    """
    # TODO(justinchuby): In the future, we can decide if it is a good idea to simply iterate over
    # all values in the function and call serialize_value_into instead.
    function_qualified_name = f"{function.domain}::{function.name}"

    def format_name(value_name: str) -> str:
        return f"{function_qualified_name}/{value_name}"

    for input in function.inputs:
        if not input.name:
            logging.warning(
                "Function '%s': Value name not set for function input: %s",
                function_qualified_name,
                input,
            )
            continue
        if not _should_create_value_info_for_value(input):
            # No need to serialize value info if it is not set
            continue
        serialize_value_into(graph_proto.value_info.add(), input, name=format_name(input.name))
    for node in function:
        for node_output in node.outputs:
            if not node_output.name:
                logging.warning(
                    "Function '%s': Value name not set for node output: %s",
                    function_qualified_name,
                    node_output,
                )
                continue
            if not _should_create_value_info_for_value(node_output):
                # No need to serialize value info if it is not set
                continue
            serialize_value_into(
                graph_proto.value_info.add(),
                node_output,
                name=format_name(node_output.name),
            )


def _serialize_opset_imports_into(
    opset_ids: proto_containers.RepeatedCompositeFieldContainer[onnx.OperatorSetIdProto],
    from_: Mapping[str, int],
) -> None:
    """Serialize opset imports into a repeated field of OperatorSetId protos.

    Args:
        opset_ids: The repeated field to serialize into.
        from_: The mapping of opset domains to versions to serialize.
    """
    # Sort names for deterministic serialization
    for domain, version in from_.items():
        opset_ids.add(domain=domain, version=version)


def _serialize_metadata_props_into(
    string_string_entries: proto_containers.RepeatedCompositeFieldContainer[
        onnx.StringStringEntryProto
    ],
    from_: Mapping[str, str],
) -> None:
    """Serialize metadata properties into a repeated field of string-string entries.

    Args:
        string_string_entries: The repeated field to serialize into.
        from_: The mapping of metadata properties to serialize.
    """
    # Sort names for deterministic serialization
    for key in sorted(from_):
        string_string_entries.add(key=key, value=from_[key])


def serialize_graph(
    graph: _protocols.GraphProtocol | _protocols.GraphViewProtocol,
) -> onnx.GraphProto:
    """Serializes the given graph into an :class:`onnx.GraphProto`.

    When the graph initializers do not have `const_value` set, they will be skipped.

    Args:
        graph: The graph to be serialized.

    Returns:
        The serialized ONNX GraphProto object.
    """
    graph_proto = onnx.GraphProto()
    serialize_graph_into(graph_proto, from_=graph)
    return graph_proto


@_capture_errors(
    lambda graph_proto, from_: (
        f"name={from_.name}, doc_string={from_.doc_string}, "
        f"len(inputs)={len(from_.inputs)}, len(initializers)={len(from_.initializers)}, "
        f"len(nodes)={len(from_)}, len(outputs)={len(from_.outputs)}, metadata_props={from_.metadata_props}"
    )
)
def serialize_graph_into(
    graph_proto: onnx.GraphProto,
    from_: _protocols.GraphProtocol | _protocols.GraphViewProtocol,
) -> None:
    if from_.name:
        graph_proto.name = from_.name
    if from_.doc_string:
        graph_proto.doc_string = from_.doc_string
    for input_ in from_.inputs:
        serialize_value_into(graph_proto.input.add(), input_)
    # TODO(justinchuby): Support sparse_initializer
    for initializer in from_.initializers.values():
        if initializer.const_value is None:
            # Skip initializers without constant values
            logger.warning(
                "Initializer '%s' does not have a constant value set.", initializer.name
            )
            continue
        # Make sure the tensor's name is the same as the value's name
        initializer.const_value.name = initializer.name
        serialize_tensor_into(graph_proto.initializer.add(), from_=initializer.const_value)
    for node in from_:
        serialize_node_into(graph_proto.node.add(), from_=node)
        for node_output in node.outputs:
            if not _should_create_value_info_for_value(node_output):
                # No need to serialize value info if it is not set
                continue
            if node_output.is_graph_output():
                # No need to serialize value info for these outputs because they are also graph outputs
                continue
            serialize_value_into(graph_proto.value_info.add(), node_output)
    for output in from_.outputs:
        serialize_value_into(graph_proto.output.add(), from_=output)
    if from_.metadata_props:
        _serialize_metadata_props_into(graph_proto.metadata_props, from_.metadata_props)


def serialize_function(
    function: _protocols.FunctionProtocol, *, create_value_info: bool = True
) -> onnx.FunctionProto:
    """Serialize an IR function as a FunctionProto.

    Args:
        function: The function to serialize.
        create_value_info: Whether to create ValueInfoProto for nodes in the function. This is supported
            starting from ONNX IR version 10.
    """
    function_proto = onnx.FunctionProto()
    serialize_function_into(
        function_proto, from_=function, create_value_info=create_value_info
    )
    return function_proto


@_capture_errors(lambda function_proto, from_, create_value_info: repr(from_))
def serialize_function_into(
    function_proto: onnx.FunctionProto,
    from_: _protocols.FunctionProtocol,
    *,
    create_value_info: bool = True,
) -> None:
    """Serialize an IR function into a FunctionProto.

    Args:
        function_proto: The proto to serialize into.
        from_: The function to serialize.
        create_value_info: Whether to create ValueInfoProto for nodes in the function. This is supported
            starting from ONNX IR version 10.
    """
    if from_.domain:
        function_proto.domain = from_.domain
    if from_.name:
        function_proto.name = from_.name
    if from_.overload:
        function_proto.overload = from_.overload
    if from_.doc_string:
        function_proto.doc_string = from_.doc_string
    if from_.opset_imports:
        # A valid ONNX graph should have at least one opset import, that is
        # the default ONNX opset.
        # Here we check for emptiness before serializing to keep the logic consistent
        _serialize_opset_imports_into(function_proto.opset_import, from_.opset_imports)
    if from_.metadata_props:
        _serialize_metadata_props_into(function_proto.metadata_props, from_.metadata_props)
    for input_ in from_.inputs:
        function_proto.input.append(input_.name)
        if not _should_create_value_info_for_value(input_):
            # No need to serialize value info if it is not set
            continue
        if not create_value_info:
            continue
        serialize_value_into(function_proto.value_info.add(), input_)
    for attr in from_.attributes.values():
        if attr.value is not None:
            serialize_attribute_into(function_proto.attribute_proto.add(), from_=attr)
        else:
            # ONNX does not record type information if the attribute does not have a default
            function_proto.attribute.append(attr.name)
    for func_output in from_.outputs:
        function_proto.output.append(func_output.name)
        # No need to serialize value info for function outputs because they are
        # also node outputs
    for node in from_:
        serialize_node_into(function_proto.node.add(), from_=node)
        # Record value info for outputs
        for node_output in node.outputs:
            if not _should_create_value_info_for_value(node_output):
                # No need to serialize value info if it is not set
                continue
            if not create_value_info:
                continue
            serialize_value_into(function_proto.value_info.add(), node_output)


def serialize_node(node: _protocols.NodeProtocol) -> onnx.NodeProto:
    node_proto = onnx.NodeProto()
    serialize_node_into(node_proto, from_=node)
    return node_proto


@_capture_errors(lambda node_proto, from_: repr(from_))
def serialize_node_into(node_proto: onnx.NodeProto, from_: _protocols.NodeProtocol) -> None:
    node_proto.op_type = from_.op_type
    if from_.domain:
        # If the domain is "", we can assume the default domain and not set it
        node_proto.domain = from_.domain
    if from_.name:
        node_proto.name = from_.name
    if from_.overload:
        node_proto.overload = from_.overload
    if from_.doc_string:
        node_proto.doc_string = from_.doc_string
    if from_.metadata_props:
        _serialize_metadata_props_into(node_proto.metadata_props, from_.metadata_props)
    for input_ in from_.inputs:
        if input_ is None:
            node_proto.input.append("")
        else:
            node_proto.input.append(input_.name)
    for output in from_.outputs:
        node_proto.output.append(output.name)
    for attr in from_.attributes.values():
        if isinstance(attr, _core.Attr):
            serialize_attribute_into(node_proto.attribute.add(), from_=attr)
        elif isinstance(attr, _core.RefAttr):
            serialize_reference_attribute_into(node_proto.attribute.add(), from_=attr)
        # Handle protocol attributes for completeness. We do not check them first because
        # calling isinstance on a protocol can be slow.
        # Most of the time, we will have Attr or RefAttr so the two branches below
        # will not be taken.
        elif isinstance(attr, _protocols.AttributeProtocol):
            serialize_attribute_into(node_proto.attribute.add(), from_=attr)
        elif isinstance(attr, _protocols.ReferenceAttributeProtocol):
            serialize_reference_attribute_into(node_proto.attribute.add(), from_=attr)
        else:
            raise TypeError(f"Unsupported attribute type: {type(attr)}")


def serialize_tensor(tensor: _protocols.TensorProtocol) -> onnx.TensorProto:
    tensor_proto = onnx.TensorProto()
    serialize_tensor_into(tensor_proto, from_=tensor)
    return tensor_proto


@_capture_errors(lambda tensor_proto, from_: repr(from_))
def serialize_tensor_into(
    tensor_proto: onnx.TensorProto, from_: _protocols.TensorProtocol
) -> None:
    if isinstance(from_, TensorProtoTensor):
        # Directly copy from the tensor proto if it is available
        tensor_proto.CopyFrom(from_.raw)
        if from_.metadata_props:
            _serialize_metadata_props_into(tensor_proto.metadata_props, from_.metadata_props)
        return

    if from_.name:
        tensor_proto.name = from_.name
    if from_.doc_string:
        tensor_proto.doc_string = from_.doc_string
    tensor_proto.data_type = from_.dtype.value
    tensor_proto.dims.extend(from_.shape.numpy())
    if isinstance(from_, _core.ExternalTensor):
        # Store external tensors as is
        tensor_proto.data_location = onnx.TensorProto.EXTERNAL
        for k, v in {
            "location": os.fspath(from_.location),
            "offset": from_.offset,
            "length": from_.length,
        }.items():
            if v is not None:
                entry = tensor_proto.external_data.add()
                entry.key = k
                entry.value = str(v)
    elif isinstance(from_, _core.StringTensor):
        tensor_proto.string_data.extend(from_.string_data())
    else:
        tensor_proto.raw_data = from_.tobytes()
    _serialize_metadata_props_into(tensor_proto.metadata_props, from_.metadata_props)


def serialize_attribute(attribute: _protocols.AttributeProtocol) -> onnx.AttributeProto:
    attribute_proto = onnx.AttributeProto()
    serialize_attribute_into(attribute_proto, from_=attribute)
    return attribute_proto


@_capture_errors(lambda attribute_proto, from_: repr(from_))
def serialize_attribute_into(
    attribute_proto: onnx.AttributeProto, from_: _protocols.AttributeProtocol
) -> None:
    attribute_proto.name = from_.name
    if from_.doc_string:
        attribute_proto.doc_string = from_.doc_string
    _fill_in_value_for_attribute(attribute_proto, from_.type, from_.value)


def _fill_in_value_for_attribute(
    attribute_proto: onnx.AttributeProto, type_: _enums.AttributeType, value: Any
) -> None:
    if type_ == _enums.AttributeType.INT:
        # value: int
        attribute_proto.i = value
        attribute_proto.type = onnx.AttributeProto.INT
    elif type_ == _enums.AttributeType.FLOAT:
        # value: float
        attribute_proto.f = value
        attribute_proto.type = onnx.AttributeProto.FLOAT
    elif type_ == _enums.AttributeType.STRING:
        # value: str
        attribute_proto.s = value.encode("utf-8")
        attribute_proto.type = onnx.AttributeProto.STRING
    elif type_ == _enums.AttributeType.INTS:
        # value: Sequence[int]
        attribute_proto.ints.extend(value)
        attribute_proto.type = onnx.AttributeProto.INTS
    elif type_ == _enums.AttributeType.FLOATS:
        # value: Sequence[float]
        attribute_proto.floats.extend(value)
        attribute_proto.type = onnx.AttributeProto.FLOATS
    elif type_ == _enums.AttributeType.STRINGS:
        # value: Sequence[str]
        attribute_proto.strings.extend([s.encode("utf-8") for s in value])
        attribute_proto.type = onnx.AttributeProto.STRINGS
    elif type_ == _enums.AttributeType.TENSOR:
        # value: _protocols.TensorProtocol
        serialize_tensor_into(attribute_proto.t, value)
        attribute_proto.type = onnx.AttributeProto.TENSOR
    elif type_ == _enums.AttributeType.GRAPH:
        # value: _protocols.GraphProtocol
        serialize_graph_into(attribute_proto.g, value)
        attribute_proto.type = onnx.AttributeProto.GRAPH
    elif type_ == _enums.AttributeType.TENSORS:
        # value: Sequence[_protocols.TensorProtocol]
        for tensor in value:
            serialize_tensor_into(attribute_proto.tensors.add(), tensor)
        attribute_proto.type = onnx.AttributeProto.TENSORS
    elif type_ == _enums.AttributeType.GRAPHS:
        # value: Sequence[_protocols.GraphProtocol]
        for graph in value:
            serialize_graph_into(attribute_proto.graphs.add(), graph)
        attribute_proto.type = onnx.AttributeProto.GRAPHS
    elif type_ == _enums.AttributeType.SPARSE_TENSOR:
        raise NotImplementedError(
            f"Sparse tensors are not supported yet. {_PLEASE_CONTRIBUTE}"
        )
    elif type_ == _enums.AttributeType.SPARSE_TENSORS:
        raise NotImplementedError(
            f"Sparse tensors are not supported yet. {_PLEASE_CONTRIBUTE}"
        )
    elif type_ == _enums.AttributeType.TYPE_PROTO:
        # value: _core.TypeAndShape
        if value.type is not None:
            serialize_type_into(attribute_proto.tp, value.type)
        # Need to create the type _before_ writing the shape
        if value.shape is not None:
            serialize_shape_into(attribute_proto.tp, value.shape)
        attribute_proto.type = onnx.AttributeProto.TYPE_PROTO
    elif type_ == _enums.AttributeType.TYPE_PROTOS:
        for ir_type in value:
            # ir_type: _core.TypeAndShape
            type_proto = attribute_proto.type_protos.add()
            if ir_type.type is not None:
                serialize_type_into(type_proto, ir_type.type)
            # Need to create the type _before_ writing the shape so that the shape can be written to the leaf type proto
            if ir_type.shape is not None:
                serialize_shape_into(type_proto, ir_type.shape)
        attribute_proto.type = onnx.AttributeProto.TYPE_PROTOS
    else:
        raise TypeError(f"Unsupported attribute type: {type_}")


@_capture_errors(lambda attribute_proto, from_: repr(from_))
def serialize_reference_attribute_into(
    attribute_proto: onnx.AttributeProto, from_: _protocols.ReferenceAttributeProtocol
) -> None:
    attribute_proto.name = from_.name
    attribute_proto.ref_attr_name = from_.ref_attr_name
    if from_.doc_string:
        attribute_proto.doc_string = from_.doc_string
    attribute_proto.type = typing.cast(onnx.AttributeProto.AttributeType, from_.type.value)


def serialize_value(value: _protocols.ValueProtocol, *, name: str = "") -> onnx.ValueInfoProto:
    """Serialize a value into a ValueInfoProto.

    Args:
        value: The proto to serialize into.
        from_: The value to serialize.
        name: A custom name to set for the value info. If not provided, the name from the value will be used.
    """
    value_info_proto = onnx.ValueInfoProto()
    serialize_value_into(value_info_proto, value, name=name)
    return value_info_proto


@_capture_errors(lambda value_info_proto, from_: repr(from_))
def serialize_value_into(
    value_info_proto: onnx.ValueInfoProto,
    from_: _protocols.ValueProtocol,
    *,
    name: str = "",
) -> None:
    """Serialize a value into a ValueInfoProto.

    Args:
        value_info_proto: The proto to serialize into.
        from_: The value to serialize.
        name: A custom name to set for the value info. If not provided, the name from the value will be used.
    """
    if name:
        value_info_proto.name = name
    else:
        value_info_proto.name = from_.name
    if from_.metadata_props:
        _serialize_metadata_props_into(value_info_proto.metadata_props, from_.metadata_props)
    if from_.type is not None:
        serialize_type_into(value_info_proto.type, from_.type)
    # Need to create the type _before_ writing the shape so that the shape can be written to the leaf type proto
    if from_.shape is not None:
        serialize_shape_into(value_info_proto.type, from_.shape)
    if from_.doc_string:
        value_info_proto.doc_string = from_.doc_string


@_capture_errors(lambda type_proto, from_: repr(from_))
def serialize_type_into(type_proto: onnx.TypeProto, from_: _protocols.TypeProtocol) -> None:
    if from_.denotation:
        type_proto.denotation = from_.denotation
    if isinstance(from_, _core.TensorType):
        tensor_type_proto = type_proto.tensor_type
        tensor_type_proto.elem_type = from_.dtype.value
    elif isinstance(from_, _core.SparseTensorType):
        sparse_tensor_type_proto = type_proto.sparse_tensor_type
        sparse_tensor_type_proto.elem_type = from_.dtype.value
    elif isinstance(from_, _core.SequenceType):
        sequence_type_proto = type_proto.sequence_type
        serialize_type_into(sequence_type_proto.elem_type, from_.elem_type)
    elif isinstance(from_, _core.OptionalType):
        optional_type_proto = type_proto.optional_type
        serialize_type_into(optional_type_proto.elem_type, from_.elem_type)
    else:
        raise TypeError(f"Unsupported type: {from_}")


def serialize_type(type_protocol: _protocols.TypeProtocol) -> onnx.TypeProto:
    type_proto = onnx.TypeProto()
    serialize_type_into(type_proto, from_=type_protocol)
    return type_proto


@_capture_errors(lambda type_proto, from_: repr(from_))
def serialize_shape_into(type_proto: onnx.TypeProto, from_: _protocols.ShapeProtocol) -> None:
    value_field = type_proto.WhichOneof("value")
    tensor_type = getattr(type_proto, value_field)
    while not isinstance(tensor_type.elem_type, int):
        # Find the leaf type that has the shape field
        type_proto = tensor_type.elem_type
        value_field = type_proto.WhichOneof("value")
        tensor_type = getattr(type_proto, value_field)
    # When from is empty, we still need to set the shape field to an empty list by touching it
    tensor_type.shape.ClearField("dim")
    for i, dim in enumerate(from_):
        denotation = from_.get_denotation(i)
        serialize_dimension_into(tensor_type.shape.dim.add(), dim, denotation)


@_capture_errors(lambda dim_proto, dim, denotation: repr(dim_proto))
def serialize_dimension_into(
    dim_proto: onnx.TensorShapeProto.Dimension,
    dim: int | _protocols.SymbolicDimProtocol,
    denotation: str | None = None,
) -> None:
    if denotation:
        dim_proto.denotation = denotation
    if isinstance(dim, int):
        dim_proto.dim_value = dim
    elif isinstance(dim, (_core.SymbolicDim, _protocols.SymbolicDimProtocol)):
        if dim.value is not None:
            # TODO(justinchuby): None is probably not a valid value for dim_param
            dim_proto.dim_param = str(dim.value)
