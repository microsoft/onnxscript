# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""data structures for the intermediate representation."""

# NOTES for developers:
# NOTE: None of these classes will have a "to_onnx" or "from_protobuf" method because
# We cannot assume that the build tool chain has protoc installed and would like
# to keep this module protobuf free. This way we separate the concerns of the IR
# and the serialization/deserialization.
#
# NOTE: Do not import pathlib in the IR. It is slow. Use os.path methods instead.

from __future__ import annotations

import abc
import contextlib
import math
import mmap
import os
import sys
import textwrap
import typing
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    OrderedDict,
    Sequence,
    Union,
)

import numpy as np

from onnxscript.ir import (
    _display,
    _enums,
    _linked_list,
    _metadata,
    _protocols,
)

if typing.TYPE_CHECKING:
    from typing_extensions import TypeGuard

TArrayCompatible = typing.TypeVar(
    "TArrayCompatible",
    bound=Union[_protocols.ArrayCompatible, _protocols.DLPackCompatible],
)

# System is little endian
IS_LITTLE_ENDIAN = sys.byteorder == "little"


def _compatible_with_numpy(obj: Any) -> TypeGuard[_protocols.ArrayCompatible]:
    """Use this function to check if an object is compatible with numpy.

    Avoid isinstance checks with the ArrayCompatible protocol for performance reasons.
    """
    return hasattr(obj, "__array__")


def _compatible_with_dlpack(obj: Any) -> TypeGuard[_protocols.DLPackCompatible]:
    """Use this function to check if an object is compatible with DLPack.

    Avoid isinstance checks with the DLPackCompatible protocol for performance reasons.
    """
    return hasattr(obj, "__dlpack__")


class TensorBase(abc.ABC, _protocols.TensorProtocol, _display.PrettyPrintable):
    """Convenience Shared methods for classes implementing TensorProtocol."""

    __slots__ = ()

    def _printable_type_shape(self) -> str:
        """Return a string representation of the shape and data type."""
        return f"{self.dtype},{self.shape}"

    def _repr_base(self) -> str:
        """Base string for the repr method.

        Example: Tensor<FLOAT:=1,5x42>
        """
        return f"{self.__class__.__name__}<{self._printable_type_shape()}>"

    @property
    def size(self) -> int:
        """The number of elements in the tensor."""
        return np.prod(self.shape.numpy())  # type: ignore[return-value,attr-defined]

    @property
    def nbytes(self) -> int:
        """The number of bytes in the tensor."""
        # Use math.ceil because when dtype is INT4, the itemsize is 0.5
        return math.ceil(self.dtype.itemsize * self.size)

    def display(self, *, page: bool | None = None) -> None:
        rich = _display.require_rich()

        if rich is None:
            status_manager = contextlib.nullcontext()
        else:
            import rich.status  # pylint: disable=import-outside-toplevel

            status_manager = rich.status.Status(f"Computing tensor stats for {self!r}")

        from onnxscript._thirdparty import (  # pylint: disable=import-outside-toplevel
            asciichartpy,
        )

        with status_manager:
            # Construct the text to display
            lines = []
            array = self.numpy().flatten()
            lines.append(repr(self))
            lines.append("")
            nan_values = np.isnan(array)
            nan_count = np.count_nonzero(nan_values)
            inf_count = np.count_nonzero(np.isinf(array))
            numbers = array[~nan_values]
            lines.append(
                f"Min: {np.min(numbers)}, Max: {np.max(numbers)}, "
                f"NaN count: {nan_count}, "
                f"Inf count: {inf_count}"
            )
            # Compute sparsity
            sparse_threathold = 1e-6
            # NOTE: count_nonzero() is faster than sum() for boolean arrays
            sparsity = np.count_nonzero(np.abs(array) < sparse_threathold) / array.size
            lines.append(f"Sparsity (abs<{sparse_threathold}): {sparsity:.2f}")

            # Compute histogram
            finite_numbers = array[np.isfinite(array)]
            lines.append("Histogram:")
            hist, bin_edges = np.histogram(finite_numbers, bins=80, density=False)
            lines.append(
                asciichartpy.plot(
                    hist, bin_edges=bin_edges, cfg={"height": 8, "format": "{:8.0f}"}
                )
            )

            text = "\n".join(lines)

        if rich is None:
            print(text)
        elif page:
            import rich.console  # pylint: disable=import-outside-toplevel

            console = rich.console.Console()
            with console.pager(styles=True):
                console.print(text)
        else:
            rich.print(text)


class Tensor(TensorBase, _protocols.TensorProtocol, Generic[TArrayCompatible]):
    """An immutable concrete value."""

    __slots__ = ("_raw", "_dtype", "_shape", "name", "doc_string")

    def __init__(
        self,
        value: TArrayCompatible,
        dtype: _enums.DataType,
        *,
        shape: Shape | None = None,
        name: str = "",
        doc_string: str | None = None,
    ) -> None:
        # NOTE: We should not do any copying here for performance reasons
        if not _compatible_with_numpy(value) and not _compatible_with_dlpack(value):
            raise TypeError(f"Expected an array compatible object, got {type(value)}")
        if not hasattr(value, "shape") and shape is None:
            raise ValueError(
                f"Expected an object with a shape attribute, but {type(value)} does not have shape. "
                "Please specify the shape explicitly."
            )
        self._raw = value
        self._dtype = dtype
        self._shape = Shape(getattr(value, "shape"))  # noqa: B009
        self.name = name
        self.doc_string = doc_string

    def __array__(self, dtype: Any = None) -> np.ndarray:
        # TODO(justinchuby): Support numpy unsupported types
        if isinstance(self._raw, np.ndarray) or _compatible_with_numpy(self._raw):
            return self._raw.__array__(dtype)
        assert _compatible_with_dlpack(
            self._raw
        ), f"Bug: Expected DLPack or Numpy compatible objects, got {type(self._raw)}"
        return np.from_dlpack(self._raw)

    def __dlpack__(self, *, stream: Any = None) -> Any:
        if _compatible_with_dlpack(self._raw):
            return self._raw.__dlpack__(stream=stream)
        return self.__array__().__dlpack__(stream=stream)

    def __repr__(self) -> str:
        return f"{self._repr_base()}({self._raw!r})"

    @property
    def dtype(self) -> _enums.DataType:
        """The data type of the tensor. Immutable."""
        return self._dtype

    @property
    def shape(self) -> Shape:
        """The shape of the tensor. Immutable."""
        return self._shape

    @property
    def raw(self) -> TArrayCompatible:
        """Backing data of the tensor. Immutable."""
        return self._raw  # type: ignore[return-value]

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array."""
        if isinstance(self._raw, np.ndarray):
            return self._raw
        # We do not cache the value to save memory
        return self.__array__()

    def tobytes(self) -> bytes:
        """Returns the value as bytes encoded in little endian.

        Override this method for more efficient serialization when the raw
        value is not a numpy array.
        """
        # TODO(justinchuby): Support DLPack
        array = self.numpy()
        if not IS_LITTLE_ENDIAN:
            return array.view(array.dtype.newbyteorder("<")).tobytes()
        return array.tobytes()


class ExternalTensor(TensorBase, _protocols.TensorProtocol):
    """An immutable concrete tensor with its data store on disk.

    This class uses memory mapping to avoid loading the tensor into memory,
    when the data type is supported by numpy. Otherwise, the tensor is loaded
    into memory lazily when accessed.

    Calling :attr:`shape` does not incur IO. Checking shape before loading
    the tensor is recommended if IO overhead and memory usage is a concern.

    To obtain an array, call :meth:`numpy`. To obtain the bytes,
    call :meth:`tobytes`.

    The :attribute:`path` can be a relative path or an absolute path.
    Serializers should handle the path correctly to conform with the ONNX spec.

    Attributes:
        path: The path to the data file. This can be a relative path or an absolute path.
        offset: The offset in bytes from the start of the file.
        length: The length of the data in bytes.
        dtype: The data type of the tensor.
        shape: The shape of the tensor.
        name: The name of the tensor. It must be specified.
        doc_string: The documentation string.
    """

    __slots__ = (
        "_path",
        "_offset",
        "_length",
        "_dtype",
        "_shape",
        "name",
        "doc_string",
        "_array",
        "raw",
    )

    def __init__(
        self,
        path: os.PathLike | str,
        offset: int | None,
        length: int | None,
        dtype: _enums.DataType,
        *,
        shape: Shape,
        name: str,
        doc_string: str | None = None,
    ) -> None:
        self._path = path
        self._offset: int | None = offset
        self._length: int | None = length
        self._dtype: _enums.DataType = dtype
        self.name: str = name  # mutable
        self._shape: Shape = shape
        self.doc_string: str | None = doc_string  # mutable
        self._array: np.ndarray | None = None
        self.raw: mmap.mmap | None = None

    @property
    def path(self) -> str | os.PathLike:
        # Immutable
        return self._path

    @property
    def offset(self) -> int | None:
        # Immutable
        return self._offset

    @property
    def length(self) -> int | None:
        # Immutable
        return self._length

    @property
    def dtype(self) -> _enums.DataType:
        # Immutable
        return self._dtype

    @property
    def shape(self) -> Shape:
        # Immutable
        return self._shape

    def _load(self):
        assert self._array is None, "Bug: The array should be loaded only once."
        # Map the whole file into the memory
        # TODO(justinchuby): Verify if this would exhaust the memory address space
        with open(self._path, "rb") as f:
            self.raw = mmap.mmap(
                f.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
        # Handle the byte order correctly by always using little endian
        dt = np.dtype(self.dtype.numpy()).newbyteorder("<")
        self._array = np.frombuffer(
            self.raw, dtype=dt, offset=self.offset or 0, count=self.size
        ).reshape(self.shape.numpy())

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self._array is None:
            self._load()
        assert self._array is not None
        return self._array.__array__(dtype)

    def __dlpack__(self, *, stream: Any = None) -> Any:
        return self.numpy().__dlpack__(stream=stream)

    def __repr__(self) -> str:
        return f"{self._repr_base()}(path='{self._path}', name={self.name!r}, offset={self._offset!r}), length={self._length!r})"

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array.

        The data will be memory mapped into memory and will not taken up physical memory space.
        """
        if self._array is None:
            self._load()
        assert self._array is not None
        return self._array

    def tobytes(self) -> bytes:
        """Return the bytes of the tensor.

        This will load the tensor into memory.
        """
        if self.raw is None:
            self._load()
        assert self.raw is not None
        offset = self._offset or 0
        length = self._length or self.nbytes
        return self.raw[offset : offset + length]


class Dimension(_protocols.DimensionProtocol, _display.PrettyPrintable):
    __slots__ = ("_value", "_denotation")

    def __init__(self, value: int | str | None, denotation: str | None = None) -> None:
        self._value = value
        self._denotation = denotation

    def __int__(self) -> int:
        if not isinstance(self.value, int):
            raise TypeError(
                f"The value of this Dimension is not int, but {type(self.value)} ({self.value})"
            )
        return self.value

    def __index__(self) -> int:
        return int(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, str)) or other is None:
            return self.value == other
        if not isinstance(other, Dimension):
            return False
        return self.value == other.value

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, (int, Dimension)):
            raise TypeError(f"Expected other to be Dimension or int, got {type(other)}")
        return int(self) < int(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, (int, Dimension)):
            raise TypeError(f"Expected other to be Dimension or int, got {type(other)}")
        return int(self) <= int(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, (int, Dimension)):
            raise TypeError(f"Expected other to be Dimension or int, got {type(other)}")
        return int(self) > int(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, (int, Dimension)):
            raise TypeError(f"Expected other to be Dimension or int, got {type(other)}")
        return int(self) >= int(other)

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def value(self) -> int | str | None:
        return self._value

    @property
    def denotation(self) -> str | None:
        return self._denotation

    def __repr__(self) -> str:
        return f"{self._value}"


class Shape(_protocols.ShapeProtocol, _display.PrettyPrintable):
    __slots__ = ("_dims",)

    def __init__(self, dims: _protocols.SimpleShape | Sequence[Dimension]) -> None:
        # TODO: Support symbolic shapes with expressions?
        for dim in dims:
            if dim is not None and not isinstance(dim, (int, str, Dimension)):
                raise TypeError(f"Expected int, str, None or Dimension, got '{type(dim)}'")
        self._dims: list[Dimension] = [
            dim if isinstance(dim, Dimension) else Dimension(dim) for dim in dims
        ]

    @property
    def dims(self) -> tuple[Dimension, ...]:
        """All dimensions in the shape.

        This property is read-only. Use __getitem__ and __setitem__ to modify the shape or create a new shape.
        """
        return tuple(self._dims)

    def rank(self) -> int:
        """The rank of the shape."""
        return len(self._dims)

    def simple(self) -> _protocols.SimpleShape:
        return tuple(dim.value for dim in self._dims)

    def numpy(self) -> tuple[int, ...]:
        if any(not isinstance(dim.value, int) for dim in self._dims):
            raise ValueError(f"Cannot convert the shape {self} to a tuple of ints")
        return tuple(dim.value for dim in self._dims)  # type: ignore

    def __len__(self) -> int:
        return len(self._dims)

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self._dims)

    def __getitem__(self, index: int) -> Dimension:
        return self._dims[index]

    def __setitem__(
        self, index: int, value: _protocols.DimensionProtocol | int | str | None
    ) -> None:
        if isinstance(value, Dimension):
            self._dims[index] = value
            return
        if isinstance(value, (int, str, type(None))):
            self._dims[index] = Dimension(value)
            return

        raise TypeError(
            f"Value must be int, str, None or DimensionProtocol. Got '{type(value)}'"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dims})"

    def __str__(self) -> str:
        """Return a string representation of the shape.

        E.g. [n,1,3]
        """
        return f"[{','.join([str(dim) for dim in self._dims])}]"

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError("Not implemented yet")


def _quoted(string: str) -> str:
    """Return a quoted string.

    This function is used to quote value/node names in the IR for better readability.
    """
    return f'"{string}"'


class Node(_protocols.NodeProtocol, _display.PrettyPrintable):
    """IR Node.

    If the ``graph`` is provided, the node will be added to the graph. Otherwise,
    user is responsible to call ``graph.append(node)`` (or other mutation methods
    in :class:`Graph`) to add the node to the graph.

    After the node is initialized, it will add itself as a user of the input values.

    The output values of the node are created during node initialization and are immutable.
    To change the output values, create a new node and replace the output.users.inputs with
    the new output values by calling :meth:`replace_input_with` on the user nodes
    of this node's outputs.

    """

    __slots__ = (
        "_name",
        "_domain",
        "_op_type",
        "_inputs",
        "_outputs",
        "_attributes",
        "_overload",
        "_version",
        "doc_string",
        "_metadata",
        "_metadata_props",
        "_graph",
    )

    def __init__(
        self,
        domain: str,
        op_type: str,
        inputs: Sequence[Value | None],
        attributes: Sequence[Attr | RefAttr] = (),
        *,
        overload: str = "",
        num_outputs: int = 1,
        version: int | None = None,
        graph: Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
    ):
        """Initialize a node and add it as a user of the input values.

        Args:
            domain: The domain of the operator. For onnx operators, this is an empty string.
            op_type: The name of the operator.
            inputs: The input values. When an input is None, it is an empty input.
            attributes: The attributes. RefAttr can be used only when the node is defined in a Function.
            overload: The overload name when the node is invoking a function.
            num_outputs: The number of outputs of the node.
            version: The version of the operator. If None, the version is unspecified and will follow that of the graph.
            graph: The graph that the node belongs to. If None, the node is not added to any graph.
                A `Node` must belong to zero or one graph.
            name: The name of the node. If None, the node is anonymous.
            doc_string: The documentation string.
        """
        self._name = name
        self._domain: str = domain
        self._op_type: str = op_type
        # NOTE: Make inputs immutable with the assumption that they are not mutated
        # very often. This way all mutations can be tracked.
        # If necessary, we can cache the inputs and outputs as tuples.
        self._inputs: tuple[Value | None, ...] = tuple(inputs)
        # Values belong to their defining nodes. The values list is immutable
        self._outputs: tuple[Value, ...] = tuple(
            Value(self, def_index=i) for i in range(num_outputs)
        )
        self._attributes: OrderedDict[str, Attr | RefAttr] = OrderedDict(
            (attr.name, attr) for attr in attributes
        )
        self._overload: str = overload
        # TODO(justinchuby): Potentially support a version range
        self._version: int | None = version
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None
        self._graph: Graph | None = graph
        self.doc_string = doc_string

        # Add the node as a user of the inputs
        for i, input_value in enumerate(self._inputs):
            if input_value is not None:
                input_value.add_user(self, i)

        # Add the node to the graph if graph is specified
        if self._graph is not None:
            self._graph.append(self)

    def __str__(self) -> str:
        node_type_text = f"{self._domain}::{self._op_type}" + f":{self._overload}" * (
            self._overload != ""
        )
        inputs_text = (
            "("
            + ", ".join(
                [
                    (
                        f"%{_quoted(x.name) if x.name else 'anonymous:' + str(id(x))}"
                        if x
                        else "None"
                    )
                    for x in self._inputs
                ]
            )
            + ")"
        )
        attributes_text = (
            (" {" + ", ".join([f"{k}={v}" for k, v in self._attributes.items()]) + "}")
            if self._attributes
            else ""
        )
        outputs_text = ", ".join(str(x) for x in self._outputs)

        return f"{outputs_text} ⬅️ {node_type_text}{inputs_text}{attributes_text}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, domain={self._domain!r}, "
            f"op_type={self._op_type!r}, inputs={self._inputs!r}, attributes={self._attributes!r}, "
            f"overload={self._overload!r}, outputs={self._outputs!r}, "
            f"version={self._version!r}, doc_string={self.doc_string!r})"
        )

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    @property
    def domain(self) -> str:
        return self._domain

    @domain.setter
    def domain(self, value: str) -> None:
        self._domain = value

    @property
    def version(self) -> int | None:
        return self._version

    @version.setter
    def version(self, value: int | None) -> None:
        self._version = value

    @property
    def op_type(self) -> str:
        return self._op_type

    @op_type.setter
    def op_type(self, value: str) -> None:
        self._op_type = value

    @property
    def overload(self) -> str:
        return self._overload

    @overload.setter
    def overload(self, value: str) -> None:
        self._overload = value

    @property
    def inputs(self) -> Sequence[Value | None]:
        return self._inputs

    @inputs.setter
    def inputs(self, _: Any) -> None:
        raise AttributeError(
            "Directly mutating the input sequence is unsupported. Please use Node.replace_input_with() instead."
        )

    def replace_input_with(self, index: int, value: Value | None) -> None:
        """Replace an input with a new value."""
        if index < 0 or index >= len(self.inputs):
            raise ValueError(f"Index out of range: {index}")
        old_input = self.inputs[index]
        self._inputs = tuple(
            value if i == index else old_input for i, old_input in enumerate(self.inputs)
        )
        if old_input is not None:
            old_input.remove_user(self, index)
        if value is not None:
            value.add_user(self, index)

    def prepend(self, /, nodes: Node | Iterable[Node]) -> None:
        """Insert a node before this node in the list of nodes in the graph.

        It is the same as calling ``graph.insert_before(self, nodes)``.

        Example::

            Before: previous_node -> self
                    previous_node' -> node -> next_node'
            After:  previous_node -> node -> self
                    previous_node' -> next_node'

        Args:
            nodes: A node or a sequence of nodes to put before this node.
        """
        if self._graph is None:
            raise ValueError("The node to prepend to does not belong to any graph.")
        self._graph.insert_before(self, nodes)

    def append(self, /, nodes: Node | Iterable[Node]) -> None:
        """Insert a node after this node in the list of nodes in the graph.

        It is the same as calling ``graph.insert_after(self, nodes)``.

        Example::

            Before: previous_node -> self
                    previous_node' -> node -> next_node'
            After:  previous_node -> self -> node
                    previous_node' -> next_node'

        Args:
            nodes:  A node or a sequence of nodes to put after this node.
        """
        if self._graph is None:
            raise ValueError("The node to append to does not belong to any graph.")
        self._graph.insert_after(self, nodes)

    @property
    def outputs(self) -> Sequence[Value]:
        return self._outputs

    @outputs.setter
    def outputs(self, _: Sequence[Value]) -> None:
        raise AttributeError("outputs is immutable. Please create a new node instead.")

    @property
    def attributes(self) -> OrderedDict[str, Attr | RefAttr]:
        return self._attributes

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for this node."""
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    @property
    def graph(self) -> Graph | None:
        return self._graph

    @graph.setter
    def graph(self, value: Graph | None) -> None:
        self._graph = value

    def op_identifier(self) -> _protocols.OperatorIdentifier:
        return self.domain, self.op_type, self.overload

    def display(self, *, page: bool | None = None) -> None:
        # Add the node's name to the displayed text
        print(f"Node: {self.name!r}")
        if self.doc_string:
            print(f"Doc: {self.doc_string}")
        super().display(page=page)


class _TensorTypeBase(_protocols.TypeProtocol, _display.PrettyPrintable):
    """Tensor types that are non recursive types."""

    __slots__ = ("_dtype", "denotation")

    def __init__(self, dtype: _enums.DataType, *, denotation: str | None = None) -> None:
        self._dtype = dtype
        self.denotation = denotation

    @property
    def dtype(self) -> _enums.DataType:
        return self._dtype

    @dtype.setter
    def dtype(self, value: _enums.DataType) -> None:
        self._dtype = value

    @property
    def elem_type(self) -> _enums.DataType:
        """Return the element type of the tensor type"""
        return self.dtype

    def __eq__(self, other: object) -> bool:
        if self.__class__ is not other.__class__:
            return False
        return self.dtype == other.dtype  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        # Remove "Type" from name for display
        short_name = self.__class__.__name__[:-4]
        return f"{short_name}({self.dtype!r})"


class TensorType(_TensorTypeBase):
    """A type that represents a tensor."""

    def __str__(self) -> str:
        return f"{self.dtype}"


class SparseTensorType(_TensorTypeBase):
    """A type that represents a sparse tensor."""


class _RecursiveTypeBase(_protocols.TypeProtocol, _display.PrettyPrintable):
    """Base for recursive types like Optional and Sequence."""

    __slots__ = ("_elem_type", "denotation")

    def __init__(
        self, elem_type: _protocols.TypeProtocol, *, denotation: str | None = None
    ) -> None:
        self._elem_type = elem_type
        self.denotation = denotation

    @property
    def dtype(self) -> _enums.DataType:
        return self._elem_type.dtype

    @dtype.setter
    def dtype(self, value: _enums.DataType) -> None:
        self._elem_type.dtype = value

    @property
    def elem_type(self) -> _protocols.TypeProtocol:
        return self._elem_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RecursiveTypeBase):
            return False
        if self.__class__ != other.__class__:
            return False
        # Recursively compare the type of the elements
        return self.elem_type == other.elem_type

    def __repr__(self) -> str:
        # Remove "Type" from name for display
        short_name = self.__class__.__name__[:-4]
        return f"{short_name}({self.elem_type!r})"


class SequenceType(_RecursiveTypeBase):
    """A type that represents a sequence of elements."""


class OptionalType(_RecursiveTypeBase):
    """A type that represents an optional element."""


class Value(_protocols.ValueProtocol, _display.PrettyPrintable):
    """IR Value."""

    __slots__ = (
        "_def_node",
        "_def_index",
        "_metadata",
        "_metadata_props",
        "_name",
        "_shape",
        "_type",
        "_const_value",
        "_users",
    )

    def __init__(
        self,
        def_node: Node | None,
        *,
        def_index: int | None,
        name: str | None = None,
        shape: Shape | None = None,
        type: _protocols.TypeProtocol | None = None,
        const_value: _protocols.TensorProtocol
        | Sequence[_protocols.TensorProtocol]
        | None = None,
    ) -> None:
        # def_node is None when the value is an input or an initializer
        self._def_node: Node | None = def_node
        self._def_index: int | None = def_index
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None

        self._name: str | None = name
        self._shape: Shape | None = shape
        self._type: _protocols.TypeProtocol | None = type
        # TODO(justinchuby): Handle initialization when a const value is provided
        # We can get shape and type information from the const value
        self._const_value = const_value
        # Use a collection of (Node, int) to store users. This is needed
        # because a single user can use the same value multiple times.
        self._users: set[tuple[Node, int]] = set()

    def __repr__(self) -> str:
        value_name = self.name if self.name else "anonymous:" + str(id(self))
        def_node = self.def_node()
        def_node_text = (
            def_node.name or "anonymous_node:" + str(id(def_node))
            if def_node is not None
            else None
        )
        return f"{self.__class__.__name__}({value_name!r}, type={self.type!r}, shape={self.shape}, def_node={def_node_text}, def_index={self.def_index()})"

    def __str__(self) -> str:
        value_name = self.name if self.name else "anonymous:" + str(id(self))
        shape_text = str(self.shape) if self.shape is not None else "?"
        type_text = str(self.type) if self.type is not None else "?"

        # Quote the name because in reality the names can have invalid characters
        # that make them hard to read
        return f"%{_quoted(value_name)}<{type_text},{shape_text}>"

    def def_node(self) -> Node | None:
        """The node that produces this value."""
        return self._def_node

    def def_index(self) -> int | None:
        """The index of the output of the defining node."""
        return self._def_index

    def users(self) -> frozenset[tuple[Node, int]]:
        return frozenset(self._users)

    def add_user(self, user: Node, index: int) -> None:
        self._users.add((user, index))

    def remove_user(self, user: Node, index: int) -> None:
        """Reduce a user node."""
        self._users.remove((user, index))

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    @property
    def type(self) -> _protocols.TypeProtocol | None:
        """The type of the tensor.

        Example types can be ``TensorType``, ``SparseTensorType``, ``SequenceType``, ``OptionalType``.
        To obtain the data type of the tensor, use ``type.dtype`` or conveniently
        :attribute:`dtype`.
        """
        return self._type

    @type.setter
    def type(self, value: _protocols.TypeProtocol | None) -> None:
        self._type = value

    @property
    def dtype(self) -> _enums.DataType | None:
        """The data type of the tensor."""
        if self._type is None:
            return None
        return self._type.dtype

    @dtype.setter
    def dtype(self, value: _enums.DataType) -> None:
        """Set the data type of the tensor.

        If the type is not set, it will be initialized to a new TensorType. To
        set the type as other types like ``SequenceType``, initialize the type
        then set :attribute:`type` instead.
        """
        if self._type is None:
            self._type = TensorType(value)
        else:
            self._type.dtype = value

    @property
    def shape(self) -> Shape | None:
        return self._shape

    @shape.setter
    def shape(self, value: _protocols.SimpleShape | Shape | None) -> None:
        if value is None:
            self._shape = None
            return
        if isinstance(value, Shape):
            self._shape = value
            return
        self._shape = Shape(value)

    @property
    def const_value(
        self,
    ) -> _protocols.TensorProtocol | Sequence[_protocols.TensorProtocol] | None:
        """A concrete value.

        The value can be backed by different raw data types, such as numpy arrays.
        The only guarantee is that it conforms TensorProtocol.
        """
        return self._const_value

    @const_value.setter
    def const_value(
        self,
        value: _protocols.TensorProtocol | Sequence[_protocols.TensorProtocol] | None,
    ) -> None:
        self._const_value = value

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attribute:`metadata_props` if you would like the metadata to be serialized
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

    def is_graph_output(self) -> bool:
        """Whether the value is an output of a graph."""
        def_node = self.def_node()
        if def_node is None:
            return False
        if def_node.graph is None:
            return False
        return self in def_node.graph.outputs


class Input(Value):
    """Input of a Graph or a Function."""

    # Slots already defined in Value
    __slots__ = ()

    def __init__(
        self,
        name: str | None = None,
        shape: Shape | None = None,
        type: _protocols.TypeProtocol | None = None,
    ) -> None:
        super().__init__(None, def_index=None)
        self._name = name
        self._shape = shape
        self._type = type


class Graph(_protocols.GraphProtocol, Sequence[Node], _display.PrettyPrintable):
    """IR Graph.

    The graph can be used as a sequence of nodes::

        for node in graph:
            print(node)
    """

    __slots__ = (
        "name",
        "_inputs",
        "_outputs",
        "_initializers",
        "_doc_string",
        "_opset_imports",
        "_nodes",
        "_metadata",
        "_metadata_props",
    )

    def __init__(
        self,
        inputs: Sequence[Input],
        outputs: Sequence[Value],
        *,
        nodes: Iterable[Node],
        initializers: Sequence[_protocols.TensorProtocol] = (),
        doc_string: str | None = None,
        opset_imports: dict[str, int] | None = None,
        name: str | None = None,
    ):
        self.name = name

        # Private fields that are not to be accessed by any other classes
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        for initializer in initializers:
            if initializer.name is None:
                raise ValueError(f"Initializer must have a name: {initializer}")
        self._initializers = {tensor.name: tensor for tensor in initializers}
        self._doc_string = doc_string
        self._opset_imports = opset_imports or {}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None
        self._nodes: _linked_list.DoublyLinkedSet[Node] = _linked_list.DoublyLinkedSet()
        # Call self.extend not self._nodes.extend so the graph reference is added to the nodes
        self.extend(nodes)

    @property
    def inputs(self) -> list[Input]:
        return self._inputs

    @property
    def outputs(self) -> list[Value]:
        return self._outputs

    @property
    def initializers(self) -> dict[str, _protocols.TensorProtocol]:
        return self._initializers

    @property
    def doc_string(self) -> str | None:
        return self._doc_string

    @doc_string.setter
    def doc_string(self, value: str | None) -> None:
        self._doc_string = value

    @property
    def opset_imports(self) -> dict[str, int]:
        return self._opset_imports

    @property
    def nodes(self) -> Sequence[Node]:
        return tuple(self._nodes)

    def __getitem__(self, index: int) -> Node:
        return self._nodes[index]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __reversed__(self) -> Iterator[Node]:
        return reversed(self._nodes)

    def _set_node_graph_to_self(self, node: Node) -> Node:
        """Set the graph reference for the node."""
        if node.graph is not None and node.graph is not self:
            raise ValueError(
                f"The node {node} belongs to another graph. Please remove it first with Graph.remove()."
            )
        node.graph = self
        return node

    # Mutation methods
    def append(self, node: Node, /) -> None:
        """Append a node to the graph in O(1) time.

        Args:
            node: The node to append.

        Raises:
            ValueError: If the node belongs to another graph.
        """
        self._set_node_graph_to_self(node)
        self._nodes.append(node)

    def extend(self, nodes: Iterable[Node], /) -> None:
        """Extend the graph with the given nodes in O(#new_nodes) time.

        Args:
            nodes: The nodes to extend the graph with.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        nodes = [self._set_node_graph_to_self(node) for node in nodes]
        self._nodes.extend(nodes)

    def remove(self, node: Node, /) -> None:
        """Remove a node from the graph in O(1) time.

        Args:
            node: The node to remove.

        Raises:
            ValueError: If the node does not belong to this graph.
        """
        if node.graph is not self:
            raise ValueError(f"The node {node} does not belong to this graph.")
        node.graph = None
        self._nodes.remove(node)

    def insert_after(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        """Insert new nodes after the given node in O(#new_nodes) time.

        Args:
            node: The node to insert after.
            new_nodes: The new nodes to insert.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        if isinstance(new_nodes, Node):
            new_nodes = (new_nodes,)
        new_nodes = [self._set_node_graph_to_self(node) for node in new_nodes]
        self._nodes.insert_after(node, new_nodes)

    def insert_before(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        """Insert new nodes before the given node in O(#new_nodes) time.

        Args:
            node: The node to insert before.
            new_nodes: The new nodes to insert.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        if isinstance(new_nodes, Node):
            new_nodes = (new_nodes,)
        new_nodes = [self._set_node_graph_to_self(node) for node in new_nodes]
        self._nodes.insert_before(node, new_nodes)

    def sort(self) -> None:
        """Topologically sort the nodes in the graph."""
        raise NotImplementedError("Not implemented yet")

    # End of mutation methods

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attribute:`metadata_props` if you would like the metadata to be serialized
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

    def __str__(self) -> str:
        return _graph_str(self)

    def __repr__(self) -> str:
        return _graph_repr(self)


def _graph_str(graph: Graph | GraphView) -> str:
    """Return a string representation of the graph."""
    # TODO(justinchuby): Show docstrings and metadata
    inputs_text = "\n" + ",\n".join(str(x) for x in graph.inputs)
    outputs_text = "\n" + ",\n".join(str(x) for x in graph.outputs)
    initializers_text = ",\n".join(str(x) for x in graph.initializers.values())
    if initializers_text:
        initializers_text = (
            "\ninitializers=(\n" + textwrap.indent(initializers_text, " " * 4) + "\n),"
        )
    signature = f"""\
graph(
name={graph.name or 'anonymous_graph:' + str(id(graph))},
inputs=({textwrap.indent(inputs_text, ' '*8)}
),
outputs=({textwrap.indent(outputs_text, ' '*8)}
),{textwrap.indent(initializers_text, ' '*4)}
)"""
    node_count = len(graph)
    number_width = len(str(node_count))
    node_lines = []
    for i, node in enumerate(graph.nodes):
        node_name = node.name if node.name else f":anonymous_node:{id(node)}"
        node_text = f"# {node_name}\n{node}"
        indented_node_text = textwrap.indent(node_text, " " * (number_width + 4))
        # Remove the leading spaces
        indented_node_text = indented_node_text.strip()
        node_lines.append(f"{i:>{number_width}} |  {indented_node_text}")
    returns = ", ".join(str(x) for x in graph.outputs)
    body = (
        "{\n"
        + textwrap.indent("\n".join(node_lines), " " * 4)
        + textwrap.indent(f"\nreturn {returns}", " " * 4)
        + "\n}"
    )

    return f"{signature} {body}"


def _graph_repr(graph: Graph | GraphView) -> str:
    """Return an repr string of the graph."""
    inputs_text = "\n" + ",\n".join(str(x) for x in graph.inputs)
    outputs_text = "\n" + ",\n".join(str(x) for x in graph.outputs)
    initializers_text = ",\n".join(str(x) for x in graph.initializers.values())
    if initializers_text:
        initializers_text = (
            "\ninitializers=(\n" + textwrap.indent(initializers_text, " " * 4) + "\n),"
        )
    return f"""\
{graph.__class__.__name__}(
    name={graph.name or 'anonymous_graph:' + str(id(graph))!r},
    inputs=({textwrap.indent(inputs_text, ' '*8)}
    ),
    outputs=({textwrap.indent(outputs_text, ' '*8)}
    ),{textwrap.indent(initializers_text, ' '*4)}
    len()={len(graph)}
)"""


class GraphView(Sequence[Node], _display.PrettyPrintable):
    """A read-only view on a graph.

    The GraphView is useful for analysis of a subgraph. It can be initialized
    with a subset of nodes from a :class:`Graph`. Creating GraphView does not
    change the ownership of the nodes, and so it is possible to create multiple
    GraphViews that contain the same nodes. If the underlying nodes / connections
    are mutated, the mutation will be reflected in all views as well.

    The graph view can be serialized to ONNX::

            graph_proto = ir.serde.serialize_graph(graph_view)

    It can also be used to create a model::

            model = ir.Model(graph_view, ir_version=8)
            model_proto = ir.serde.serialize_model(model)

    Attributes:
        name: The name of the graph.
        inputs: The input values of the graph.
        outputs: The output values of the graph.
        nodes: All nodes visible in this view. They do not have to be sorted.
        initializers: The initializers in the graph.
        doc_string: Documentation string.
        opset_imports: Opsets imported by the graph.
        metadata_props: Metadata.
    """

    __slots__ = (
        "name",
        "inputs",
        "outputs",
        "initializers",
        "doc_string",
        "opset_imports",
        "nodes",
        "_metadata",
        "_metadata_props",
    )

    def __init__(
        self,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        *,
        nodes: Iterable[Node],
        initializers: Sequence[_protocols.TensorProtocol] = (),
        doc_string: str | None = None,
        opset_imports: dict[str, int] | None = None,
        name: str | None = None,
    ):
        self.name = name
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        for initializer in initializers:
            if initializer.name is None:
                raise ValueError(f"Initializer must have a name: {initializer}")
        self.initializers = {tensor.name: tensor for tensor in initializers}
        self.doc_string = doc_string
        self.opset_imports = opset_imports or {}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None
        self.nodes: list[Node] = list(nodes)

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __reversed__(self) -> Iterator[Node]:
        return reversed(self.nodes)

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attribute:`metadata_props` if you would like the metadata to be serialized
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

    def __str__(self) -> str:
        return _graph_str(self)

    def __repr__(self) -> str:
        return _graph_repr(self)


class Model(_protocols.ModelProtocol, _display.PrettyPrintable):
    __slots__ = (
        "graph",
        "ir_version",
        "producer_name",
        "producer_version",
        "domain",
        "model_version",
        "doc_string",
        "_functions",
        "_metadata",
        "_metadata_props",
    )
    """IR Model.

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

    def __init__(
        self,
        graph: Graph | GraphView,
        *,
        ir_version: int,
        producer_name: str | None = None,
        producer_version: str | None = None,
        domain: str | None = None,
        model_version: int | None = None,
        doc_string: str | None = None,
        functions: Sequence[Function] = (),
    ) -> None:
        self.graph: Graph | GraphView = graph  # type: ignore[assignment]
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.producer_version = producer_version
        self.domain = domain
        self.model_version = model_version
        self.doc_string = doc_string
        self._functions = {func.identifier(): func for func in functions}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None

    @property
    def functions(self) -> dict[_protocols.OperatorIdentifier, Function]:
        return self._functions

    @property
    def opset_imports(self) -> dict[str, int]:
        return self.graph.opset_imports

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for this node."""
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    def __str__(self) -> str:
        # TODO(justinchuby): Show docstrings and metadata
        signature = f"""\
<
    ir_version={self.ir_version!r},
    opset_imports={self.opset_imports!r},
    producer_name={self.producer_name!r},
    producer_version={self.producer_version!r},
    domain={self.domain!r},
    model_version={self.model_version!r},
>"""
        graph_text = str(self.graph)
        functions_text = ",\n\n".join(str(func) for func in self.functions.values())
        return f"{signature}\n{graph_text}" + f"\n\n{functions_text}" * len(self.functions)

    def __repr__(self) -> str:
        return f"""\
Model(
    ir_version={self.ir_version!r},
    opset_imports={self.opset_imports!r},
    producer_name={self.producer_name!r},
    producer_version={self.producer_version!r},
    domain={self.domain!r},
    model_version={self.model_version!r},
    functions={self.functions!r},
    graph={textwrap.indent(repr(self.graph), ' ' * 4).strip()}
)"""


class Function(_protocols.FunctionProtocol, _display.PrettyPrintable):
    __slots__ = (
        "_domain",
        "_name",
        "_overload",
        "_graph",
        "_attributes",
        "_metadata",
        "_metadata_props",
    )

    def __init__(
        self,
        domain: str,
        name: str,
        overload: str = "",
        *,
        # Ensure the inputs and outputs of the function belong to a graph
        # and not from an outer scope
        graph: Graph,
        attributes: Sequence[Attr],
    ) -> None:
        self._domain = domain
        self._name = name
        self._overload = overload
        self._graph = graph
        self._attributes = OrderedDict((attr.name, attr) for attr in attributes)
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None

    def identifier(self) -> _protocols.OperatorIdentifier:
        return self.domain, self.name, self.overload

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def domain(self) -> str:
        return self._domain

    @domain.setter
    def domain(self, value: str) -> None:
        self._domain = value

    @property
    def overload(self) -> str:
        return self._overload

    @overload.setter
    def overload(self, value: str) -> None:
        self._overload = value

    @property
    def inputs(self) -> list[Input]:
        return self._graph.inputs

    @property
    def outputs(self) -> list[Value]:
        return self._graph.outputs

    @property
    def attributes(self) -> OrderedDict[str, Attr]:
        return self._attributes

    @property
    def nodes(self) -> Sequence[Node]:
        return self._graph.nodes

    def __getitem__(self, index: int) -> Node:
        return self._graph.__getitem__(index)

    def __len__(self) -> int:
        return self._graph.__len__()

    def __iter__(self) -> Iterator[Node]:
        return self._graph.__iter__()

    def __reversed__(self) -> Iterator[Node]:
        return self._graph.__reversed__()

    @property
    def doc_string(self) -> str | None:
        return self._graph.doc_string

    @doc_string.setter
    def doc_string(self, value: str | None) -> None:
        self._graph.doc_string = value

    @property
    def opset_imports(self) -> dict[str, int]:
        return self._graph.opset_imports

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for this node."""
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    # Mutation methods
    def append(self, node: Node, /) -> None:
        """Append a node to the function in O(1) time."""
        self._graph.append(node)

    def extend(self, nodes: Iterable[Node], /) -> None:
        """Extend the function with the given nodes in O(#new_nodes) time."""
        self._graph.extend(nodes)

    def remove(self, node: Node, /) -> None:
        """Remove a node from the function in O(1) time."""
        self._graph.remove(node)

    def insert_after(self, node: Node, new_nodes: Iterable[Node], /) -> None:
        """Insert new nodes after the given node in O(#new_nodes) time."""
        self._graph.insert_after(node, new_nodes)

    def insert_before(self, node: Node, new_nodes: Iterable[Node], /) -> None:
        """Insert new nodes before the given node in O(#new_nodes) time."""
        self._graph.insert_before(node, new_nodes)

    def sort(self) -> None:
        """Topologically sort the nodes in the function."""
        self._graph.sort()

    # End of mutation methods

    def __str__(self) -> str:
        full_name = f"{self.domain}::{self.name}" + f":{self.overload}" * (self.overload != "")
        inputs_text = ",\n".join(str(x) for x in self.inputs)
        outputs_text = ",\n".join(str(x) for x in self.outputs)
        attributes_text = ",\n".join(
            attr.name + f": {attr.type}" + f" = {attr.value}" * (attr.value is None)
            for attr in self.attributes.values()
        )
        if attributes_text:
            attributes_text = (
                "\nattributes={\n" + textwrap.indent(attributes_text, " " * 4) + "\n}"
            )
        signature = f"""\
<
    opset_imports={self.opset_imports!r},
>
def {full_name}(
    inputs=(
{textwrap.indent(inputs_text, ' '*8)}
    ),{textwrap.indent(attributes_text, ' '*4)}
    outputs=(
{textwrap.indent(outputs_text, ' '*8)}
    ),
)"""
        node_count = len(self.nodes)
        number_width = len(str(node_count))
        node_lines = []
        for i, node in enumerate(self.nodes):
            node_name = node.name if node.name else f":anonymous_node:{id(node)}"
            node_text = f"# {node_name}\n{node}"
            indented_node_text = textwrap.indent(node_text, " " * (number_width + 4))
            # Remove the leading spaces
            indented_node_text = indented_node_text.strip()
            node_lines.append(f"{i:>{number_width}} |  {indented_node_text}")
        returns = ", ".join(str(x) for x in self.outputs)
        body = (
            "{\n"
            + textwrap.indent("\n".join(node_lines), " " * 4)
            + textwrap.indent(f"\nreturn {returns}", " " * 4)
            + "\n}"
        )

        return f"{signature} {body}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.domain!r}, {self.name!r}, {self.overload!r}, inputs={self.inputs!r}, attributes={self.attributes!r}), outputs={self.outputs!r})"


class RefAttr(_protocols.ReferenceAttributeProtocol, _display.PrettyPrintable):
    """Reference attribute."""

    __slots__ = ("_name", "_ref_attr_name", "_type", "doc_string")

    def __init__(
        self,
        name: str,
        ref_attr_name: str,
        type: _enums.AttributeType,
        *,
        doc_string: str | None = None,
    ) -> None:
        self._name = name
        self._ref_attr_name = ref_attr_name
        self._type = type
        self.doc_string = doc_string

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def ref_attr_name(self) -> str:
        return self._ref_attr_name

    @ref_attr_name.setter
    def ref_attr_name(self, value: str) -> None:
        self._ref_attr_name = value

    @property
    def type(self) -> _enums.AttributeType:
        return self._type

    @type.setter
    def type(self, value: _enums.AttributeType) -> None:
        self._type = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name!r}, {self._type!r}, ref_attr_name={self.ref_attr_name!r})"


class Attr(_protocols.AttributeProtocol, _display.PrettyPrintable):
    """Base class for ONNX attributes."""

    __slots__ = ("name", "type", "value", "doc_string")

    def __init__(
        self,
        name: str,
        type: _enums.AttributeType,
        value: Any,
        *,
        doc_string: str | None = None,
    ):
        self.name = name
        self.type = type
        self.value = value
        self.doc_string = doc_string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _protocols.AttributeProtocol):
            return False

        if self.name != other.name:
            return False
        if self.type != other.type:
            return False
        if self.value != other.value:
            return False
        if self.doc_string != other.doc_string:
            return False
        return True

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, {self.type!r}, {self.value!r})"


# NOTE: The following classes are just supporting classes (partially applied) for convenience
# But I think they would be useful to have in the IR by having the type info
# explicitly in the class type.
class AttrFloat32(Attr):
    def __init__(self, name: str, value: float, doc_string: str | None = None):
        super().__init__(
            name,
            _enums.AttributeType.FLOAT,
            value,
            doc_string=doc_string,
        )


class AttrInt64(Attr):
    def __init__(self, name: str, value: int, doc_string: str | None = None):
        super().__init__(
            name,
            _enums.AttributeType.INT,
            value,
            doc_string=doc_string,
        )


class AttrString(Attr):
    def __init__(self, name: str, value: str, doc_string: str | None = None):
        super().__init__(
            name,
            _enums.AttributeType.STRING,
            value,
            doc_string=doc_string,
        )


class AttrTensor(Attr):
    def __init__(
        self,
        name: str,
        value: _protocols.TensorProtocol,
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.TENSOR,
            value,
            doc_string=doc_string,
        )


class AttrGraph(Attr):
    def __init__(
        self,
        name: str,
        value: Graph,
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.GRAPH,
            value,
            doc_string=doc_string,
        )

    def __str__(self) -> str:
        return textwrap.indent("\n" + super().__str__(), " " * 4)


class AttrFloat32s(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[float],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.FLOATS,
            value,
            doc_string=doc_string,
        )


class AttrInt64s(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[int],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.INTS,
            value,
            doc_string=doc_string,
        )


class AttrStrings(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[str],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.STRINGS,
            value,
            doc_string=doc_string,
        )


class AttrTensors(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.TensorProtocol],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.TENSORS,
            value,
            doc_string=doc_string,
        )


class AttrGraphs(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[Graph],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.GRAPHS,
            value,
            doc_string=doc_string,
        )


# NOTE: SparseTensor should be a sparse tensor proto
class AttrSparseTensor(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.SparseTensorProtocol],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.SPARSE_TENSOR,
            value,
            doc_string=doc_string,
        )


class AttrSparseTensors(Attr):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.SparseTensorProtocol],
        doc_string: str | None = None,
    ):
        super().__init__(
            name,
            _enums.AttributeType.SPARSE_TENSORS,
            value,
            doc_string=doc_string,
        )


class AttrTypeProto(Attr):
    def __init__(
        self,
        name: str,
        value: _protocols.TypeProtocol,
        doc_string: str | None = None,
    ):
        # TODO(justinchuby): Include shape as well
        super().__init__(
            name,
            _enums.AttributeType.TYPE_PROTO,
            value,
            doc_string=doc_string,
        )
