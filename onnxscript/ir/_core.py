# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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
import dataclasses
import heapq
import math
import mmap
import os
import sys
import textwrap
import typing
from typing import (
    AbstractSet,
    Any,
    Collection,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    OrderedDict,
    Sequence,
    Union,
)

import ml_dtypes
import numpy as np

import onnxscript
from onnxscript.ir import (
    _display,
    _enums,
    _linked_list,
    _metadata,
    _name_authority,
    _protocols,
    _type_casting,
)

if typing.TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import TypeGuard

TArrayCompatible = typing.TypeVar(
    "TArrayCompatible",
    bound=Union[_protocols.ArrayCompatible, _protocols.DLPackCompatible],
)

# System is little endian
_IS_LITTLE_ENDIAN = sys.byteorder == "little"
# Data types that are not supported by numpy
_NON_NUMPY_NATIVE_TYPES = frozenset(
    (
        _enums.DataType.BFLOAT16,
        _enums.DataType.FLOAT8E4M3FN,
        _enums.DataType.FLOAT8E4M3FNUZ,
        _enums.DataType.FLOAT8E5M2,
        _enums.DataType.FLOAT8E5M2FNUZ,
        _enums.DataType.INT4,
        _enums.DataType.UINT4,
        _enums.DataType.FLOAT4E2M1,
    )
)


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

        Example: Tensor<FLOAT,[5,42]>
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

    def display(self, *, page: bool = False) -> None:
        rich = _display.require_rich()

        if rich is None:
            status_manager = contextlib.nullcontext()
        else:
            import rich.status  # type: ignore[import-not-found, no-redef]  # pylint: disable=import-outside-toplevel

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
            import rich.console  # type: ignore[import-not-found, no-redef]  # pylint: disable=import-outside-toplevel

            console = rich.console.Console()
            with console.pager():
                console.print(text)
        else:
            rich.print(text)


def _check_numpy_representation_type(array: np.ndarray, dtype: _enums.DataType) -> None:
    """Check if the numpy array dtype matches the IR data type.

    When the dtype is not one of the numpy native dtypes, the value needs need to be:

    - ``int8`` or ``uint8`` for int4, with the sign bit extended to 8 bits.
    - ``uint8`` for uint4 or float4.
    - ``uint8`` for 8-bit data types.
    - ``uint16`` for bfloat16

    or corresponding dtypes from the ``ml_dtype`` package.
    """
    if dtype in _NON_NUMPY_NATIVE_TYPES:
        if dtype.itemsize == 2 and array.dtype not in (np.uint16, ml_dtypes.bfloat16):
            raise TypeError(
                f"The numpy array dtype must be uint16 or ml_dtypes.bfloat16 (not {array.dtype}) for IR data type {dtype}."
            )
        if dtype.itemsize == 1 and array.dtype not in (
            np.uint8,
            ml_dtypes.float8_e4m3b11fnuz,
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.float8_e5m2fnuz,
            ml_dtypes.float8_e5m2,
        ):
            raise TypeError(
                f"The numpy array dtype must be uint8 or ml_dtypes.float8* (not {array.dtype}) for IR data type {dtype}."
            )
        if dtype == _enums.DataType.INT4:
            if array.dtype not in (np.int8, np.uint8, ml_dtypes.int4):
                raise TypeError(
                    f"The numpy array dtype must be int8 or uint8 or ml_dtypes.int4 (not {array.dtype}) for IR data type {dtype}."
                )
        if dtype == _enums.DataType.UINT4:
            if array.dtype not in (np.uint8, ml_dtypes.uint4):
                raise TypeError(
                    f"The numpy array dtype must be uint8 or or ml_dtypes.uint4 (not {array.dtype}) for IR data type {dtype}."
                )
        if dtype == _enums.DataType.FLOAT4E2M1:
            if array.dtype not in (np.uint8, ml_dtypes.float4_e2m1fn):
                raise TypeError(
                    f"The numpy array dtype must be uint8 or ml_dtypes.float4_e2m1fn (not {array.dtype}) for IR data type {dtype}."
                )
        return

    try:
        dtype_numpy = _enums.DataType.from_numpy(array.dtype)
    except TypeError as e:
        raise TypeError(
            "Failed to convert the numpy dtype to an IR data type. "
            "If you are using a non-native dtype, be sure to specify the corresponding IR dtype when "
            "creating a Tensor."
        ) from e

    if dtype_numpy != dtype:
        raise TypeError(
            f"The numpy array dtype {array.dtype} does not match the IR data type {dtype}."
        )


def _maybe_view_np_array_with_ml_dtypes(
    array: np.ndarray, dtype: _enums.DataType
) -> np.ndarray:
    """Reinterpret the array when it is a bit representation of a dtype not supported by numpy.

    Args:
        array: The numpy array to reinterpret.
        dtype: The data type to reinterpret the array as.

    Returns:
        The array reinterpreted as the dtype.
    """
    if dtype == _enums.DataType.BFLOAT16:
        return array.view(ml_dtypes.bfloat16)
    if dtype == _enums.DataType.FLOAT8E4M3FN:
        return array.view(ml_dtypes.float8_e4m3fn)
    if dtype == _enums.DataType.FLOAT8E4M3FNUZ:
        return array.view(ml_dtypes.float8_e4m3fnuz)
    if dtype == _enums.DataType.FLOAT8E5M2:
        return array.view(ml_dtypes.float8_e5m2)
    if dtype == _enums.DataType.FLOAT8E5M2FNUZ:
        return array.view(ml_dtypes.float8_e5m2fnuz)
    if dtype == _enums.DataType.INT4:
        return array.view(ml_dtypes.int4)
    if dtype == _enums.DataType.UINT4:
        return array.view(ml_dtypes.uint4)
    if dtype == _enums.DataType.FLOAT4E2M1:
        return array.view(ml_dtypes.float4_e2m1fn)
    return array


class Tensor(TensorBase, _protocols.TensorProtocol, Generic[TArrayCompatible]):  # pylint: disable=too-many-ancestors
    """An immutable concrete tensor.

    This class is a wrapper around the raw tensor data. The raw tensor data can be a numpy array
    compatible object (e.g. ``np.ndarray``, ``torch.Tensor``) or a ``DLPack`` compatible object.
    The tensor is immutable and the data is not copied at initialization.

    To create a tensor from a numpy array::

        >>> import numpy as np
        >>> array = np.array([1, 2, 3])
        >>> tensor = Tensor(array)
        >>> # The tensor itself can be treated as a numpy array because it implements the __array__ method
        >>> np.allclose(tensor, array)
        True

    To get a numpy array from the tensor, call :meth:`numpy`. To convert the tensor
    to a byte string for serialization, call :meth:`tobytes`.

    It is recommended to check the size of the tensor first before accessing the
    underlying data, because accessing the data may be expensive and incur IO
    overhead.

    Subclass this class to efficiently handle different types of tensors from different frameworks.

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

    __slots__ = (
        "_dtype",
        "_metadata",
        "_metadata_props",
        "_raw",
        "_shape",
        "doc_string",
        "name",
    )

    def __init__(
        self,
        value: TArrayCompatible,
        dtype: _enums.DataType | None = None,
        *,
        shape: Shape | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        """Initialize a tensor.

        Args:
            value: The backing data of the tensor. It can be a numpy array compatible object or a DLPack compatible object.
                When the dtype is not one of the numpy native dtypes, the value needs
                to be ``uint8`` for 4-bit and 8-bit data types, and ``uint16`` for bfloat16
                when the value is a numpy array; :param:`dtype` must be specified in this case.
            dtype: The data type of the tensor. It can be None only when value is a numpy array.
                Users are responsible for making sure the dtype matches the value when value is not a numpy array.
            shape: The shape of the tensor. If None, the shape is obtained from the value.
            name: The name of the tensor.
            doc_string: The documentation string.
            metadata_props: The metadata properties.

        Raises:
            TypeError: If the value is not a numpy array compatible or a DLPack compatible object.
            TypeError: If the value is a numpy array and the dtype is specified but does not match the dtype of the array.
            ValueError: If the shape is not specified and the value does not have a shape attribute.
            ValueError: If the dtype is not specified and the value is not a numpy array.
        """
        # NOTE: We should not do any copying here for performance reasons
        if not _compatible_with_numpy(value) and not _compatible_with_dlpack(value):
            raise TypeError(f"Expected an array compatible object, got {type(value)}")
        if shape is None:
            # Obtain the shape from the value
            if not hasattr(value, "shape"):
                raise ValueError(
                    f"Expected an object with a shape attribute, but {type(value)} does not have shape. "
                    "Please specify the shape explicitly."
                )
            self._shape = Shape(getattr(value, "shape"), frozen=True)  # noqa: B009
        else:
            self._shape = shape
            self._shape._frozen = True
        if dtype is None:
            if isinstance(value, np.ndarray):
                self._dtype = _enums.DataType.from_numpy(value.dtype)
            else:
                raise ValueError(
                    "The dtype must be specified when the value is not a numpy array."
                )
        else:
            if isinstance(value, np.ndarray):
                # Make sure the dtype matches the value
                _check_numpy_representation_type(value, dtype)
            # Users are responsible for making sure the dtype matches the value
            # when value is not a numpy array
            self._dtype = dtype

        # View the bfloat16, float8 and int4 types using ml_dtypes
        if isinstance(value, np.ndarray):
            value = _maybe_view_np_array_with_ml_dtypes(value, self._dtype)  # type: ignore[assignment]

        self._raw = value
        self.name = name
        self.doc_string = doc_string
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props = metadata_props

    def __array__(self, dtype: Any = None) -> np.ndarray:
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

    def __dlpack_device__(self) -> tuple[int, int]:
        if _compatible_with_dlpack(self._raw):
            return self._raw.__dlpack_device__()
        return self.__array__().__dlpack_device__()

    def __repr__(self) -> str:
        return f"{self._repr_base()}({self._raw!r}, name={self.name!r})"

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
        """Return the tensor as a numpy array.

        When the data type is not supported by numpy, the dtypes from the ``ml_dtype``
        package are used. The values can be reinterpreted as bit representations
        using the ``.view()`` method.
        """
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
        if self.dtype in {
            _enums.DataType.INT4,
            _enums.DataType.UINT4,
            _enums.DataType.FLOAT4E2M1,
        }:
            # Pack the array into int4
            array = _type_casting.pack_int4(array)
        else:
            assert self.dtype.itemsize == array.itemsize, "Bug: The itemsize should match"
        if not _IS_LITTLE_ENDIAN:
            array = array.view(array.dtype.newbyteorder("<"))
        return array.tobytes()

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attr:`metadata_props` if you would like the metadata to be serialized
        to the ONNX proto.
        """
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata


class ExternalTensor(TensorBase, _protocols.TensorProtocol):  # pylint: disable=too-many-ancestors
    """An immutable concrete tensor with its data store on disk.

    This class uses memory mapping to avoid loading the tensor into memory,
    when the data type is supported by numpy. Otherwise, the tensor is loaded
    into memory lazily when accessed.

    Calling :attr:`shape` does not incur IO. Checking shape before loading
    the tensor is recommended if IO overhead and memory usage is a concern.

    To obtain an array, call :meth:`numpy`. To obtain the bytes,
    call :meth:`tobytes`.

    The :attr:`location` must be a relative path conforming to the ONNX
    specification. Given the correct :attr:`base_dir`, the :attr:`path` is computed
    to be the full path to the data file. Users should expect that the :attr:`path`
    always leads to the correct file. At initialization, paths are not checked.
    It is the user's responsibility to ensure the paths are valid and accessible.

    Attributes:
        location: The location of the data file. It is the path relative to the base directory.
        base_dir: The base directory for the external data. It is used to resolve relative paths.
            At serialization, only the :attr:`location` is serialized into the "location" field of the ``TensorProto``.
        path: The path to the data file. This is computed by joining :attr:`base_dir` and :attr:`location`.
        offset: The offset in bytes from the start of the file.
        length: The length of the data in bytes.
        dtype: The data type of the tensor.
        shape: The shape of the tensor.
        name: The name of the tensor. It must be specified.
        doc_string: The documentation string.
        metadata_props: The metadata properties.
    """

    __slots__ = (
        "_array",
        "_base_dir",
        "_dtype",
        "_length",
        "_location",
        "_metadata",
        "_metadata_props",
        "_offset",
        "_shape",
        "doc_string",
        "name",
        "raw",
    )

    def __init__(
        self,
        location: os.PathLike | str,
        offset: int | None,
        length: int | None,
        dtype: _enums.DataType,
        *,
        shape: Shape,
        name: str,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
        base_dir: os.PathLike | str = "",
    ) -> None:
        """Initialize an external tensor.

        Args:
            location: The location of the data file. It is the path relative to the base directory.
            offset: The offset in bytes from the start of the file.
            length: The length of the data in bytes.
            dtype: The data type of the tensor.
            shape: The shape of the tensor.
            name: The name of the tensor..
            doc_string: The documentation string.
            metadata_props: The metadata properties.
            base_dir: The base directory for the external data. It is used to resolve relative paths.
        """
        # NOTE: Do not verify the location by default. This is because the location field
        # in the tensor proto can be anything and we would like deserialization from
        # proto to IR to not fail.
        if onnxscript.DEBUG:
            if os.path.isabs(location):
                raise ValueError(
                    "The location must be a relative path. Please specify base_dir as well."
                )
        self._location = location
        self._base_dir = base_dir
        self._offset: int | None = offset
        self._length: int | None = length
        self._dtype: _enums.DataType = dtype
        self.name: str = name  # mutable
        self._shape: Shape = shape
        self._shape._frozen = True
        self.doc_string: str | None = doc_string  # mutable
        self._array: np.ndarray | None = None
        self.raw: mmap.mmap | None = None
        self._metadata_props = metadata_props
        self._metadata: _metadata.MetadataStore | None = None

    @property
    def base_dir(self) -> str | os.PathLike:
        # Mutable
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: str | os.PathLike) -> None:
        self._base_dir = value

    @property
    def location(self) -> str | os.PathLike:
        # Immutable
        return self._location

    @property
    def path(self) -> str:
        # Immutable, computed
        return os.path.join(self._base_dir, self._location)

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
        if self.size == 0:
            # When the size is 0, mmap is impossible and meaningless
            self._array = np.empty(self.shape.numpy(), dtype=self.dtype.numpy())
            return
        # Map the whole file into the memory
        # TODO(justinchuby): Verify if this would exhaust the memory address space
        with open(self.path, "rb") as f:
            self.raw = mmap.mmap(
                f.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
        # Handle the byte order correctly by always using little endian
        dt = np.dtype(self.dtype.numpy()).newbyteorder("<")
        if self.dtype in {
            _enums.DataType.INT4,
            _enums.DataType.UINT4,
            _enums.DataType.FLOAT4E2M1,
        }:
            # Use uint8 to read in the full byte. Otherwise ml_dtypes.int4 will clip the values
            dt = np.dtype(np.uint8).newbyteorder("<")
            count = self.size // 2 + self.size % 2
        else:
            count = self.size
        self._array = np.frombuffer(self.raw, dtype=dt, offset=self.offset or 0, count=count)
        shape = self.shape.numpy()
        if self.dtype == _enums.DataType.INT4:
            # Unpack the int4 arrays
            self._array = _type_casting.unpack_int4(self._array, shape)
        elif self.dtype == _enums.DataType.UINT4:
            self._array = _type_casting.unpack_uint4(self._array, shape)
        elif self.dtype == _enums.DataType.FLOAT4E2M1:
            self._array = _type_casting.unpack_float4e2m1(self._array, shape)
        else:
            self._array = self._array.reshape(shape)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self._array is None:
            self._load()
        assert self._array is not None
        return self._array.__array__(dtype)

    def __dlpack__(self, *, stream: Any = None) -> Any:
        raise NotImplementedError(
            "ExternalTensor does not support DLPack because it uses memory mapping. "
            "Call numpy() to get a numpy array instead."
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        raise NotImplementedError(
            "ExternalTensor does not support DLPack because it uses memory mapping. "
            "Call numpy() to get a numpy array instead."
        )

    def __repr__(self) -> str:
        return (
            f"{self._repr_base()}(location='{self.location}', name={self.name!r}, "
            f"offset={self.offset!r}, length={self.length!r}, base_dir={self.base_dir!r})"
        )

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

    def release(self) -> None:
        """Delete all references to the memory buffer and close the memory-mapped file."""
        self._array = None
        if self.raw is not None:
            self.raw.close()
            self.raw = None

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attr:`metadata_props` if you would like the metadata to be serialized
        to the ONNX proto.
        """
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata


class StringTensor(TensorBase, _protocols.TensorProtocol):  # pylint: disable=too-many-ancestors
    """Multidimensional array of strings (as binary data to match the string_data field in TensorProto)."""

    __slots__ = (
        "_metadata",
        "_metadata_props",
        "_raw",
        "_shape",
        "doc_string",
        "name",
    )

    def __init__(
        self,
        value: Sequence[bytes] | npt.NDArray[np.bytes_],
        *,
        shape: Shape | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        """Initialize a tensor.

        Args:
            value: The backing data of the tensor. It can be a numpy array or a Sequence of bytes.
            shape: The shape of the tensor. If None, the shape is obtained from the value.
            name: The name of the tensor.
            doc_string: The documentation string.
            metadata_props: The metadata properties.
        """
        if shape is None:
            if not hasattr(value, "shape"):
                raise ValueError(
                    f"Expected an object with a shape attribute, but {type(value)} does not have shape. "
                    "Please specify the shape explicitly."
                )
            self._shape = Shape(getattr(value, "shape"), frozen=True)  # noqa: B009
        else:
            self._shape = shape
            self._shape._frozen = True
        self._raw = value
        self.name = name
        self.doc_string = doc_string
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props = metadata_props

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if isinstance(self._raw, np.ndarray):
            return self._raw
        assert isinstance(
            self._raw, Sequence
        ), f"Bug: Expected a sequence, got {type(self._raw)}"
        return np.array(self._raw, dtype=dtype).reshape(self.shape.numpy())

    def __dlpack__(self, *, stream: Any = None) -> Any:
        del stream  # unused
        raise TypeError("StringTensor does not support DLPack")

    def __dlpack_device__(self) -> tuple[int, int]:
        raise TypeError("StringTensor does not support DLPack")

    def __repr__(self) -> str:
        return f"{self._repr_base()}({self._raw!r}, name={self.name!r})"

    @property
    def dtype(self) -> _enums.DataType:
        """The data type of the tensor. Immutable."""
        return _enums.DataType.STRING

    @property
    def shape(self) -> Shape:
        """The shape of the tensor. Immutable."""
        return self._shape

    @property
    def raw(self) -> Sequence[bytes] | npt.NDArray[np.bytes_]:
        """Backing data of the tensor. Immutable."""
        return self._raw  # type: ignore[return-value]

    def numpy(self) -> npt.NDArray[np.bytes_]:
        """Return the tensor as a numpy array."""
        return self.__array__()

    def tobytes(self) -> bytes:
        raise ValueError("StringTensor does not support tobytes. Use 'string_data' instead.")

    def string_data(self) -> Sequence[bytes]:
        """Return the string data of the tensor."""
        if isinstance(self._raw, np.ndarray):
            return self._raw.flatten().tolist()
        return self._raw

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attr:`metadata_props` if you would like the metadata to be serialized
        to the ONNX proto.
        """
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata


class SymbolicDim(_protocols.SymbolicDimProtocol, _display.PrettyPrintable):
    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        """Initialize a symbolic dimension.

        Args:
            value: The value of the dimension. It should not be an int.
        """
        if isinstance(value, int):
            raise TypeError(
                "The value of a SymbolicDim cannot be an int. "
                "If you are creating a Shape, use int directly instead of SymbolicDim."
            )
        self._value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicDim):
            return self.value == other
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def value(self) -> str | None:
        return self._value

    def __str__(self) -> str:
        return f"{self._value}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


class Shape(_protocols.ShapeProtocol, _display.PrettyPrintable):
    __slots__ = ("_dims", "_frozen")

    def __init__(
        self,
        dims: Iterable[int | SymbolicDim | str | None],
        /,
        denotations: Iterable[str | None] | None = None,
        frozen: bool = False,
    ) -> None:
        """Initialize a shape.

        Args:
            dims: The dimensions of the shape. Each dimension can be an integer or a
                SymbolicDim or any Python object. When a ``dim`` is not an integer or a
                SymbolicDim, it is converted to a SymbolicDim.
            denotations: The denotations of the dimensions. If None, the denotations are not set.
                Standard denotation can optionally be used to denote tensor
                dimensions with standard semantic descriptions to ensure
                that operations are applied to the correct axis of a tensor.
                Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
                for pre-defined dimension denotations.
            frozen: If True, the shape is immutable and cannot be modified. This
                is useful when the shape is initialized by a Tensor.
        """
        self._dims: list[int | SymbolicDim] = [
            SymbolicDim(dim) if not isinstance(dim, (int, SymbolicDim)) else dim
            for dim in dims
        ]
        self._denotations: list[str | None] = (
            list(denotations) if denotations is not None else [None] * len(self._dims)
        )
        if len(self._denotations) != len(self._dims):
            raise ValueError(
                "The number of denotations, when provided, must be equal to the number of dimensions."
            )
        self._frozen: bool = frozen

    def copy(self):
        """Return a copy of the shape."""
        return Shape(self._dims, self._denotations, self._frozen)

    @property
    def dims(self) -> tuple[int | SymbolicDim, ...]:
        """All dimensions in the shape.

        This property is read-only. Use __getitem__ and __setitem__ to modify the shape or create a new shape.
        """
        return tuple(self._dims)

    def rank(self) -> int:
        """The rank of the shape."""
        return len(self._dims)

    def numpy(self) -> tuple[int, ...]:
        if any(not isinstance(dim, int) for dim in self._dims):
            raise ValueError(f"Cannot convert the shape {self} to a tuple of ints")
        return tuple(dim for dim in self._dims)  # type: ignore

    def __len__(self) -> int:
        return len(self._dims)

    def __iter__(self) -> Iterator[int | SymbolicDim]:
        return iter(self._dims)

    @typing.overload
    def __getitem__(self, index: int) -> int | SymbolicDim: ...

    @typing.overload
    def __getitem__(self, index: slice) -> tuple[int | SymbolicDim, ...]: ...

    def __getitem__(self, index):
        return tuple(self._dims)[index]

    def __setitem__(self, index: int, value: int | SymbolicDim | str | None) -> None:
        """Set the dimension at the index.

        Args:
            index: The index of the dimension.
            value: The value of the dimension.

        Raises:
            TypeError: If the shape is frozen and cannot be modified.
            TypeError: If the value is not an int or SymbolicDim.
        """
        if self._frozen:
            raise TypeError("The shape is frozen and cannot be modified.")
        if isinstance(value, str) or value is None:
            value = SymbolicDim(value)
        if not isinstance(value, (int, SymbolicDim)):
            raise TypeError(f"Expected int, str, None or SymbolicDim, got '{type(value)}'")

        self._dims[index] = value

    def get_denotation(self, index: int) -> str | None:
        """Return the denotation of the dimension at the index.

        Args:
            index: The index of the dimension.

        Returns:
            The denotation of the dimension.
        """
        return self._denotations[index]

    def set_denotation(self, index: int, denotation: str | None) -> None:
        """Set the denotation of the dimension at the index.

        Args:
            index: The index of the dimension.
            denotation: The denotation of the dimension.
        """
        self._denotations[index] = denotation

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dims!r})"

    def __str__(self) -> str:
        """Return a string representation of the shape.

        E.g. [n,1,3]
        """
        return f"[{','.join([str(dim) for dim in self._dims])}]"

    def __eq__(self, other: object) -> bool:
        """Return True if the shapes are equal.

        Two shapes are eqaul if all their dimensions are equal.
        """
        if isinstance(other, Shape):
            return self._dims == other._dims
        if not isinstance(other, Iterable):
            return False
        return self._dims == list(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


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
    To change the output values, create a new node and replace the each of the inputs of ``output.uses()`` with
    the new output values by calling :meth:`replace_input_with` on the using nodes
    of this node's outputs.
    """

    __slots__ = (
        "_attributes",
        "_domain",
        "_graph",
        "_inputs",
        "_metadata",
        "_metadata_props",
        "_name",
        "_op_type",
        "_outputs",
        "_overload",
        "_version",
        "doc_string",
    )

    def __init__(
        self,
        domain: str,
        op_type: str,
        inputs: Iterable[Value | None],
        attributes: Iterable[Attr | RefAttr] = (),
        *,
        overload: str = "",
        num_outputs: int | None = None,
        outputs: Sequence[Value] | None = None,
        version: int | None = None,
        graph: Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ):
        """Initialize a node and add it as a user of the input values.

        Args:
            domain: The domain of the operator. For onnx operators, this is an empty string.
            op_type: The name of the operator.
            inputs: The input values. When an input is None, it is an empty input.
            attributes: The attributes. RefAttr can be used only when the node is defined in a Function.
            overload: The overload name when the node is invoking a function.
            num_outputs: The number of outputs of the node. If not specified, the number is 1.
            outputs: The output values. If None, the outputs are created during initialization.
            version: The version of the operator. If None, the version is unspecified and will follow that of the graph.
            graph: The graph that the node belongs to. If None, the node is not added to any graph.
                A `Node` must belong to zero or one graph.
            name: The name of the node. If None, the node is anonymous.
            doc_string: The documentation string.
            metadata_props: The metadata properties.

        Raises:
            TypeError: If the attributes are not Attr or RefAttr.
            ValueError: If `num_outputs`, when not None, is not the same as the length of the outputs.
            ValueError: If an output value is None, when outputs is specified.
            ValueError: If an output value has a producer set already, when outputs is specified.
        """
        self._name = name
        self._domain: str = domain
        self._op_type: str = op_type
        # NOTE: Make inputs immutable with the assumption that they are not mutated
        # very often. This way all mutations can be tracked.
        # If necessary, we can cache the inputs and outputs as tuples.
        self._inputs: tuple[Value | None, ...] = tuple(inputs)
        # Values belong to their defining nodes. The values list is immutable
        self._outputs: tuple[Value, ...] = self._create_outputs(num_outputs, outputs)
        attributes = tuple(attributes)
        if attributes and not isinstance(attributes[0], (Attr, RefAttr)):
            raise TypeError(
                f"Expected the attributes to be Attr or RefAttr, got {type(attributes[0])}. "
                "If you are copying the attributes from another node, make sure you call "
                "node.attributes.values() because it is a dictionary."
            )
        self._attributes: OrderedDict[str, Attr | RefAttr] = OrderedDict(
            (attr.name, attr) for attr in attributes
        )
        self._overload: str = overload
        # TODO(justinchuby): Potentially support a version range
        self._version: int | None = version
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = metadata_props
        self._graph: Graph | None = graph
        self.doc_string = doc_string

        # Add the node as a use of the inputs
        for i, input_value in enumerate(self._inputs):
            if input_value is not None:
                input_value._add_usage(self, i)  # pylint: disable=protected-access

        # Add the node to the graph if graph is specified
        if self._graph is not None:
            self._graph.append(self)

    def _create_outputs(
        self, num_outputs: int | None, outputs: Sequence[Value] | None
    ) -> tuple[Value, ...]:
        """Check the parameters and create outputs for the node.

        Args:
            num_outputs: The number of outputs of the node.
            outputs: The output values of the node.

        Returns:
            The output values of the node.

        Raises:
            ValueError: If `num_outputs`, when not None, is not the same as the length of the outputs.
            ValueError: If an output value is None.
            ValueError: If an output value has a producer set already.
        """
        # Check num_outputs and outputs are consistent
        if num_outputs is not None and outputs is not None and num_outputs != len(outputs):
            raise ValueError(
                "num_outputs must be the same as len(outputs) when num_outputs is specified."
                f"num_outputs: {num_outputs}, outputs: {outputs}"
            )
        # 1. If outputs is specified (can be empty []), use the outputs
        if outputs is not None:
            # Check all output values are valid first
            for output in outputs:
                if output is None:
                    raise ValueError(f"Output value cannot be None. All outputs: {outputs}")
                if output.producer() is not None:
                    raise ValueError(
                        f"Supplied output value cannot have a producer when used for initializing a Node. "
                        f"Output: {output}. All outputs: {outputs}"
                    )
            result = []
            for i, output in enumerate(outputs):
                output._producer = self  # pylint: disable=protected-access
                output._index = i  # pylint: disable=protected-access
                result.append(output)
            return tuple(result)

        # 2. If num_outputs is specified, create num_outputs outputs
        if num_outputs is None:
            # Default to 1 output
            num_outputs = 1
        assert num_outputs is not None
        return tuple(Value(self, index=i) for i in range(num_outputs))

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
                        if x is not None
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
            old_input._remove_usage(self, index)  # pylint: disable=protected-access
        if value is not None:
            value._add_usage(self, index)  # pylint: disable=protected-access

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

    @property
    def graph(self) -> Graph | None:
        return self._graph

    @graph.setter
    def graph(self, value: Graph | None) -> None:
        self._graph = value

    def op_identifier(self) -> _protocols.OperatorIdentifier:
        return self.domain, self.op_type, self.overload

    def display(self, *, page: bool = False) -> None:
        # Add the node's name to the displayed text
        print(f"Node: {self.name!r}")
        if self.doc_string:
            print(f"Doc: {self.doc_string}")
        super().display(page=page)


class _TensorTypeBase(_protocols.TypeProtocol, _display.PrettyPrintable, Hashable):
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

    def __hash__(self) -> int:
        return hash(repr(self))

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


class _RecursiveTypeBase(_protocols.TypeProtocol, _display.PrettyPrintable, Hashable):
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

    def __hash__(self) -> int:
        return hash(repr(self))

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
    """IR Value.

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
        metadata_props: Metadata.
    """

    __slots__ = (
        "_const_value",
        "_index",
        "_metadata",
        "_metadata_props",
        "_name",
        "_producer",
        "_shape",
        "_type",
        "_uses",
        "doc_string",
    )

    def __init__(
        self,
        producer: Node | None = None,
        *,
        index: int | None = None,
        name: str | None = None,
        shape: Shape | None = None,
        type: _protocols.TypeProtocol | None = None,
        doc_string: str | None = None,
        const_value: _protocols.TensorProtocol | None = None,
    ) -> None:
        """Initialize a value.

        Args:
            producer: The node that produces the value.
                It can be ``None`` when the value is initialized first than its producer.
            index: The index of the output of the defining node.
            name: The name of the value.
            shape: The shape of the value.
            type: The type of the value.
            doc_string: The documentation string.
            const_value: The constant tensor if the value is constant.
        """
        self._producer: Node | None = producer
        self._index: int | None = index
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = None

        self._name: str | None = name
        self._shape: Shape | None = shape
        self._type: _protocols.TypeProtocol | None = type
        # TODO(justinchuby): Handle initialization when a const value is provided
        # We can get shape and type information from the const value
        self._const_value = const_value
        # Use a collection of (Node, int) to store uses. This is needed
        # because a single use can use the same value multiple times.
        # Use a dictionary to preserve insertion order so that the visiting order is deterministic
        self._uses: dict[tuple[Node, int], None] = {}
        self.doc_string = doc_string

    def __repr__(self) -> str:
        value_name = self.name if self.name else "anonymous:" + str(id(self))
        producer = self.producer()
        if producer is None:
            producer_text = "None"
        elif producer.name is not None:
            producer_text = producer.name
        else:
            producer_text = f"anonymous_node:{id(producer)}"
        return f"{self.__class__.__name__}({value_name!r}, type={self.type!r}, shape={self.shape}, producer={producer_text}, index={self.index()})"

    def __str__(self) -> str:
        value_name = self.name if self.name is not None else "anonymous:" + str(id(self))
        shape_text = str(self.shape) if self.shape is not None else "?"
        type_text = str(self.type) if self.type is not None else "?"

        # Quote the name because in reality the names can have invalid characters
        # that make them hard to read
        return f"%{_quoted(value_name)}<{type_text},{shape_text}>"

    def producer(self) -> Node | None:
        """The node that produces this value.

        When producer is ``None``, the value does not belong to a node, and is
        typically a graph input or an initializer.
        """
        return self._producer

    def index(self) -> int | None:
        """The index of the output of the defining node."""
        return self._index

    def uses(self) -> Collection[tuple[Node, int]]:
        """Return a set of uses of the value.

        The set contains tuples of ``(Node, index)`` where the index is the index of the input
        of the node. For example, if ``node.inputs[1] == value``, then the use is ``(node, 1)``.
        """
        return self._uses.keys()

    def _add_usage(self, use: Node, index: int) -> None:
        """Add a usage of this value.

        This is an internal method. It should only be called by the Node class.
        """
        self._uses[(use, index)] = None

    def _remove_usage(self, use: Node, index: int) -> None:
        """Remove a node from the uses of this value.

        This is an internal method. It should only be called by the Node class.
        """
        self._uses.pop((use, index))

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        if self._const_value is not None:
            self._const_value.name = value
        self._name = value

    @property
    def type(self) -> _protocols.TypeProtocol | None:
        """The type of the tensor.

        Example types can be ``TensorType``, ``SparseTensorType``, ``SequenceType``, ``OptionalType``.
        To obtain the data type of the tensor, use ``type.dtype`` or conveniently
        :attr:`dtype`.
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
        then set :attr:`type` instead.
        """
        if self._type is None:
            self._type = TensorType(value)
        else:
            self._type.dtype = value

    @property
    def shape(self) -> Shape | None:
        return self._shape

    @shape.setter
    def shape(self, value: Shape | None) -> None:
        if value is None:
            self._shape = None
            return
        if isinstance(value, Shape):
            self._shape = value
            return
        raise TypeError(f"Expected value to be a Shape or None, got '{type(value)}'")

    @property
    def const_value(
        self,
    ) -> _protocols.TensorProtocol | None:
        """A concrete value.

        The value can be backed by different raw data types, such as numpy arrays.
        The only guarantee is that it conforms TensorProtocol.
        """
        return self._const_value

    @const_value.setter
    def const_value(
        self,
        value: _protocols.TensorProtocol | None,
    ) -> None:
        if onnxscript.DEBUG:
            if value is not None and not isinstance(value, _protocols.TensorProtocol):
                raise TypeError(
                    f"Expected value to be a TensorProtocol or None, got '{type(value)}'"
                )
        self._const_value = value

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

    def is_graph_output(self) -> bool:
        """Whether the value is an output of a graph."""
        if (producer := self.producer()) is None:
            return False
        if (graph := producer.graph) is None:
            return False
        # Cannot use `in` because __eq__ may be defined by subclasses, even though
        # it is not recommended
        return any(output is self for output in graph.outputs)


def Input(
    name: str | None = None,
    shape: Shape | None = None,
    type: _protocols.TypeProtocol | None = None,
    doc_string: str | None = None,
) -> Value:
    """Create an input of a Graph or a Function.

    This is equivalent to calling ``Value(name=name, shape=shape, type=type, doc_string=doc_string)``.
    """

    # NOTE: The function name is capitalized to maintain API backward compatibility.

    return Value(name=name, shape=shape, type=type, doc_string=doc_string)


def _check_node_safe_to_remove(
    node: Node, to_remove: AbstractSet[Node], graph_outputs: AbstractSet[Value]
) -> None:
    """Check if a node is safe to remove.

    1. It checks to make sure there are no users of the node that are not
        to be removed before removing it.
    2. It checks the node does not contribute to any graph outputs.

    This check is typically O(1) assuming the number of uses of the node is small

    Args:
        node: The node to check.
        to_remove: A set of nodes that are to be removed.
            This set is used to check if the node is still being used by other
            nodes that are not to be removed.
        graph_outputs: A set of values that are outputs of the graph.

    Raises:
        ValueError: If the node does not belong to this graph or if there are users of the node.
        ValueError: If the node is still being used by other nodes not to be removed.
    """
    for output in node.outputs:
        if output in graph_outputs:
            raise ValueError(
                f"Node '{node!r}' is still an output of the graph and cannot be removed when safe=True."
            )
        uses_not_to_remove = [user for user, _ in output.uses() if user not in to_remove]
        if uses_not_to_remove:
            raise ValueError(
                f"Output value '{output!r}' is still being used by other nodes that are not to be "
                f"removed. All of its users that is not being removed: {uses_not_to_remove!r}. "
                "Please make sure these nodes are no longer using the output value."
            )


class Graph(_protocols.GraphProtocol, Sequence[Node], _display.PrettyPrintable):
    """IR Graph.

    Graph represents a computation graph. In addition to the ONNX specification
    specified fields, it also contains a mapping of :attr:`opset_imports`. This
    allows different subgraphs to import different opsets. It is the responsibility
    of the deserializer to reconcile the different opsets.

    The `nodes` are not guaranteed to be topologically sorted. But the
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

    __slots__ = (
        "_doc_string",
        "_initializers",
        "_inputs",
        "_metadata",
        "_metadata_props",
        "_name_authority",
        "_nodes",
        "_opset_imports",
        "_outputs",
        "name",
    )

    def __init__(
        self,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        *,
        nodes: Iterable[Node],
        initializers: Sequence[Value] = (),
        doc_string: str | None = None,
        opset_imports: dict[str, int] | None = None,
        name: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ):
        self.name = name

        # Private fields that are not to be accessed by any other classes
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._initializers = {}
        for initializer in initializers:
            if isinstance(initializer, str):
                raise TypeError(
                    "Initializer must be a Value, not a string. "
                    "If you are copying the initializers from another graph, "
                    "make sure you call graph.initializers.values() because it is a dictionary."
                )
            if initializer.name is None:
                raise ValueError(f"Initializer must have a name: {initializer}")
            self._initializers[initializer.name] = initializer
        self._doc_string = doc_string
        self._opset_imports = opset_imports or {}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = metadata_props
        self._nodes: _linked_list.DoublyLinkedSet[Node] = _linked_list.DoublyLinkedSet()
        # Be sure the initialize the name authority before extending the nodes
        # because it is used to name the nodes and their outputs
        self._name_authority = _name_authority.NameAuthority()
        # Call self.extend not self._nodes.extend so the graph reference is added to the nodes
        self.extend(nodes)

    @property
    def inputs(self) -> list[Value]:
        return self._inputs

    @property
    def outputs(self) -> list[Value]:
        return self._outputs

    @property
    def initializers(self) -> dict[str, Value]:
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

    def __getitem__(self, index: int) -> Node:
        return self._nodes[index]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __reversed__(self) -> Iterator[Node]:
        return reversed(self._nodes)

    def _set_node_graph_to_self_and_assign_names(self, node: Node) -> Node:
        """Set the graph reference for the node and assign names to it and its outputs if they don't have one."""
        if node.graph is not None and node.graph is not self:
            raise ValueError(
                f"The node '{node!r}' belongs to another graph. Please remove it first with Graph.remove()."
            )
        # Give the node and its output values names if they don't not have one
        self._name_authority.register_or_name_node(node)
        for value in node._outputs:  # pylint: disable=protected-access
            self._name_authority.register_or_name_value(value)
        node.graph = self
        return node

    def node(self, index_or_name: int | str, /) -> Node:
        """Get a node by index or name.

        This is an O(n) operation. Getting nodes on the ends of the graph (0 or -1) is O(1).

        .. note::
            If you need repeated random access, consider turning it into a list with ``list(graph)`` .
            Or a dictionary for repeated access by name: ``{node.name for node in graph}`` .

        When a name is provided and if there are multiple nodes with the same name,
        the first node with the name is returned.

        Args:
            index_or_name: The index or name of the node.

        Returns:
            The node if found.

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the node with the given name is not found.
        """
        # NOTE: This is a method specific to Graph, not required by the protocol unless proven
        if isinstance(index_or_name, int):
            return self[index_or_name]
        for node in self:
            if node.name == index_or_name:
                return node
        raise ValueError(f"Node with name '{index_or_name}' not found.")

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph in O(1) time.

        Note that this method returns the number of nodes this graph directly contains.
        It does not count nodes in subgraphs.

        This is an alias for ``len(graph)``. Use this if you prefer a more descriptive
        name for readability.
        """
        # NOTE: This is a method specific to Graph, not required by the protocol unless proven
        return len(self)

    # Mutation methods
    def append(self, node: Node, /) -> None:
        """Append a node to the graph in O(1) time.

        Unique names will be assigned to the node and its values if any name is ``None``.

        Args:
            node: The node to append.

        Raises:
            ValueError: If the node belongs to another graph.
        """
        self._set_node_graph_to_self_and_assign_names(node)
        self._nodes.append(node)

    def extend(self, nodes: Iterable[Node], /) -> None:
        """Extend the graph with the given nodes in O(#new_nodes) time.

        Unique names will be assigned to the node and its values if any name is ``None``.

        Args:
            nodes: The nodes to extend the graph with.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        nodes = [self._set_node_graph_to_self_and_assign_names(node) for node in nodes]
        self._nodes.extend(nodes)

    def remove(self, nodes: Node | Iterable[Node], /, safe: bool = False) -> None:
        """Remove nodes from the graph in O(#num of nodes to remove) time.

        If any errors are raise, to ensure the graph is not left in an inconsistent state,
        the graph is not modified.

        Args:
            nodes: The node to remove.
            safe: If True, performs the following actions before removal:

                1. It checks to make sure there are no users of the node that are not
                to be removed before removing it.
                2. It checks the node does not contribute to any graph outputs.
                3. It removes references to all inputs so it is no longer a user of other nodes.

        Raises:
            ValueError: If any node to remove does not belong to this graph.
            ValueError: (When ``safe=True``) If the node does not belong to this graph or if there are users of the node.
            ValueError: (When ``safe=True``) If the node is still being used by other nodes not to be removed.
        """
        if not isinstance(nodes, Iterable):
            nodes_set: AbstractSet[Node] = {nodes}
        else:
            nodes_set = frozenset(nodes)
        graph_outputs = frozenset(self.outputs)
        for node in nodes_set:
            if node.graph is not self:
                raise ValueError(f"The node '{node!r}' does not belong to this graph.")
            if safe:
                # Check 1, 2
                _check_node_safe_to_remove(node, nodes_set, graph_outputs)
        for node in nodes_set:
            if safe:
                # 3. Detach from all inputs so that it is no longer a user of other nodes
                for i in range(len(node.inputs)):
                    node.replace_input_with(i, None)
            # Set attributes to remove the node from this graph
            node.graph = None
            self._nodes.remove(node)

    def insert_after(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        """Insert new nodes after the given node in O(#new_nodes) time.

        Unique names will be assigned to the node and its values if any name is ``None``.

        Args:
            node: The node to insert after.
            new_nodes: The new nodes to insert.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        if isinstance(new_nodes, Node):
            new_nodes = (new_nodes,)
        new_nodes = [self._set_node_graph_to_self_and_assign_names(node) for node in new_nodes]
        self._nodes.insert_after(node, new_nodes)

    def insert_before(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        """Insert new nodes before the given node in O(#new_nodes) time.

        Unique names will be assigned to the node and its values if any name is ``None``.

        Args:
            node: The node to insert before.
            new_nodes: The new nodes to insert.

        Raises:
            ValueError: If any node belongs to another graph.
        """
        if isinstance(new_nodes, Node):
            new_nodes = (new_nodes,)
        new_nodes = [self._set_node_graph_to_self_and_assign_names(node) for node in new_nodes]
        self._nodes.insert_before(node, new_nodes)

    def sort(self) -> None:
        """Perform a topological sort of this graph and all subgraphs in O(#nodes + #values) time.

        This sort is stable. It preserves the original order as much as possible.

        Referece: https://github.com/madelson/MedallionTopologicalSort#stable-sort

        Raises:
            ValueError: If the graph contains a cycle, making topological sorting impossible.
        """
        # Obtain all nodes from the graph and its subgraphs for sorting
        nodes = list(onnxscript.ir.traversal.RecursiveGraphIterator(self))
        # Store the sorted nodes of each subgraph
        sorted_nodes_by_graph: dict[Graph, list[Node]] = {
            graph: [] for graph in {node.graph for node in nodes if node.graph is not None}
        }
        # TODO: Explain why we need to store direct predecessors and children and why
        # we only need to store the direct ones

        # The depth of a node is defined as the number of direct children it has
        node_depth: dict[Node, int] = dict.fromkeys(nodes, 0)
        # Direct predecessors of a node
        node_predecessors: dict[Node, list[Node]] = {node: [] for node in nodes}
        # Store the negative index of the nodes because heapq is a min heap and we
        # want to pop the node with largest index value first, effectively turning
        # it to a max heap
        neg_node_index: dict[Node, int] = {node: -i for i, node in enumerate(nodes)}

        def add_predecessor(child: Node, predecessor: Node | None) -> None:
            """Add a predecessor of a node, and increment the depth of the predecessor."""
            if predecessor is None:
                return
            node_predecessors[child].append(predecessor)
            node_depth[predecessor] += 1

        # 1. Build the direct predecessors of each node and the depth of each node
        # for sorting topolocally using Kahn's algorithm.
        # Note that when a node contains graph attributes (aka. has subgraphs),
        # we consider all nodes in the subgraphs *predecessors* of this node. This
        # way we ensure the implicit dependencies of the subgraphs are captured
        # as predecessors of the node.
        for node in nodes:
            # All producers of input values are considered as direct predecessors.
            for input_value in node.inputs:
                if input_value is None:
                    continue
                predecessor_node = input_value.producer()
                add_predecessor(node, predecessor_node)
            # All nodes in attribute graphs are considered as direct predecessors.
            for attr in node.attributes.values():
                if not isinstance(attr, Attr):
                    continue
                # A nice thing about this algorithm is that we only need to record
                # direct predecessors. This continues to be true even with subgraphs.
                # When a node in a subgraph (a) contains its own subgraphs (b), the
                # node in subgraphs (b) are guranteed to appear before the node
                # in (a).
                if attr.type == _enums.AttributeType.GRAPH:
                    for predecessor_node in attr.value:
                        add_predecessor(node, predecessor_node)
                elif attr.type == _enums.AttributeType.GRAPHS:
                    for attribute_graph in attr.value:
                        for predecessor_node in attribute_graph:
                            add_predecessor(node, predecessor_node)

        # 2. Priority Queue: Track nodes with zero direct children in a priority queue,
        # using NEGATIVE original index for ordering.
        # This ensures nodes appearing LATER in the original order are processed EARLIER.
        # We get REVERSED topological order of each subgraph.
        priority_queue: list[tuple[int, Node]] = [
            (neg_node_index[node], node) for node in nodes if node_depth[node] == 0
        ]
        heapq.heapify(priority_queue)

        # 3. Topological Sort:
        num_of_sorted_nodes = 0
        while priority_queue:
            # Pop the node with the most negative index and add it to the sorted nodes by subgraph.
            _, current_node = heapq.heappop(priority_queue)
            assert current_node.graph is not None
            sorted_nodes_by_graph[current_node.graph].append(current_node)
            num_of_sorted_nodes += 1
            # Decrement the depth of its predecessors. If any predecessor node has zero direct children, push it into the queue.
            for predecessor_node in node_predecessors[current_node]:
                node_depth[predecessor_node] -= 1
                if node_depth[predecessor_node] == 0:
                    heapq.heappush(
                        priority_queue, (neg_node_index[predecessor_node], predecessor_node)
                    )

        # 4. Cycle Check: Ensure all nodes are processed. If not, raise a ValueError indicating a cycle.
        if num_of_sorted_nodes != len(nodes):
            raise ValueError("Graph contains a cycle, topological sort is not possible.")

        # 5. Reverse: Reverse the sorted nodes of each subgraph to get the topological order.
        for graph, sorted_nodes in sorted_nodes_by_graph.items():
            # The graph container ensures all the nodes are unique so we can safely extend
            graph.extend(reversed(sorted_nodes))

    # End of mutation methods

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
    inputs=({textwrap.indent(inputs_text, ' ' * 8)}
    ),
    outputs=({textwrap.indent(outputs_text, ' ' * 8)}
    ),{textwrap.indent(initializers_text, ' ' * 4)}
)"""
    node_count = len(graph)
    number_width = len(str(node_count))
    node_lines = []
    for i, node in enumerate(graph):
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
    inputs=({textwrap.indent(inputs_text, ' ' * 8)}
    ),
    outputs=({textwrap.indent(outputs_text, ' ' * 8)}
    ),{textwrap.indent(initializers_text, ' ' * 4)}
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

    The model created with a GraphView will have a fixed topology, and its graph
    will remain read-only as a GraphView. No copying will be done during the
    initialization process.

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

    __slots__ = (
        "_metadata",
        "_metadata_props",
        "doc_string",
        "initializers",
        "inputs",
        "name",
        "nodes",
        "opset_imports",
        "outputs",
    )

    def __init__(
        self,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        *,
        nodes: Iterable[Node],
        initializers: Sequence[_protocols.ValueProtocol] = (),
        doc_string: str | None = None,
        opset_imports: dict[str, int] | None = None,
        name: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ):
        self.name = name
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        for initializer in initializers:
            if initializer.name is None:
                raise ValueError(f"Initializer must have a name: {initializer}")
        self.initializers = {tensor.name: tensor for tensor in initializers}
        self.doc_string = doc_string
        self.opset_imports = opset_imports or {}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = metadata_props
        self._nodes: tuple[Node, ...] = tuple(nodes)

    def __getitem__(self, index: int) -> Node:
        return self._nodes[index]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __reversed__(self) -> Iterator[Node]:
        return reversed(self._nodes)

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

    def __str__(self) -> str:
        return _graph_str(self)

    def __repr__(self) -> str:
        return _graph_repr(self)


class Model(_protocols.ModelProtocol, _display.PrettyPrintable):
    __slots__ = (
        "_functions",
        "_metadata",
        "_metadata_props",
        "doc_string",
        "domain",
        "graph",
        "ir_version",
        "model_version",
        "producer_name",
        "producer_version",
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
        graph: Graph,
        *,
        ir_version: int,
        producer_name: str | None = None,
        producer_version: str | None = None,
        domain: str | None = None,
        model_version: int | None = None,
        doc_string: str | None = None,
        functions: Sequence[Function] = (),
        meta_data_props: dict[str, str] | None = None,
    ) -> None:
        self.graph: Graph = graph
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.producer_version = producer_version
        self.domain = domain
        self.model_version = model_version
        self.doc_string = doc_string
        self._functions = {func.identifier(): func for func in functions}
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = meta_data_props

    @property
    def functions(self) -> dict[_protocols.OperatorIdentifier, Function]:
        return self._functions

    @property
    def opset_imports(self) -> dict[str, int]:
        return self.graph.opset_imports

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
        functions_text = "\n\n".join(str(func) for func in self.functions.values())
        return f"{signature}\n{graph_text}" + f"\n\n{functions_text}"

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


class Function(_protocols.FunctionProtocol, Sequence[Node], _display.PrettyPrintable):
    """IR functions.

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

    __slots__ = (
        "_attributes",
        "_domain",
        "_graph",
        "_metadata",
        "_metadata_props",
        "_name",
        "_overload",
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
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        self._domain = domain
        self._name = name
        self._overload = overload
        self._graph = graph
        self._attributes = OrderedDict((attr.name, attr) for attr in attributes)
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = metadata_props

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
    def inputs(self) -> list[Value]:
        return self._graph.inputs

    @property
    def outputs(self) -> list[Value]:
        return self._graph.outputs

    @property
    def attributes(self) -> OrderedDict[str, Attr]:
        return self._attributes

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

    # Mutation methods
    def append(self, node: Node, /) -> None:
        """Append a node to the function in O(1) time."""
        self._graph.append(node)

    def extend(self, nodes: Iterable[Node], /) -> None:
        """Extend the function with the given nodes in O(#new_nodes) time."""
        self._graph.extend(nodes)

    def remove(self, nodes: Node | Iterable[Node], /, safe: bool = False) -> None:
        """Remove nodes from the graph in O(#num of nodes) time.

        If any errors are raise, to ensure the graph is not left in an inconsistent state,
        the graph is not modified.

        Args:
            nodes: The node to remove.
            safe: If True, performs the following actions before removal:

                1. It checks to make sure there are no users of the node that are not
                to be removed before removing it.
                2. It checks the node does not contribute to any graph outputs.
                3. It removes references to all inputs so it is no longer a user of other nodes.

        Raises:
            ValueError: If any node to remove does not belong to this graph.
            ValueError: (When ``safe=True``) If the node does not belong to this graph or if there are users of the node.
            ValueError: (When ``safe=True``) If the node is still being used by other nodes not to be removed.
        """
        self._graph.remove(nodes, safe=safe)

    def insert_after(self, node: Node, new_nodes: Iterable[Node], /) -> None:
        """Insert new nodes after the given node in O(#new_nodes) time."""
        self._graph.insert_after(node, new_nodes)

    def insert_before(self, node: Node, new_nodes: Iterable[Node], /) -> None:
        """Insert new nodes before the given node in O(#new_nodes) time."""
        self._graph.insert_before(node, new_nodes)

    def sort(self) -> None:
        """Perform a topological sort of this graph and all subgraphs in O(#nodes + #values) time."""
        self._graph.sort()

    # End of mutation methods

    def __str__(self) -> str:
        full_name = f"{self.domain}::{self.name}" + f":{self.overload}" * (self.overload != "")
        inputs_text = ",\n".join(str(x) for x in self.inputs)
        outputs_text = ",\n".join(str(x) for x in self.outputs)
        attributes_text = ",\n".join(
            f"{attr.name}: {attr.type}" + f" = {attr.value}" * (attr.value is not None)
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
{textwrap.indent(inputs_text, ' ' * 8)}
    ),{textwrap.indent(attributes_text, ' ' * 4)}
    outputs=(
{textwrap.indent(outputs_text, ' ' * 8)}
    ),
)"""
        node_count = len(self)
        number_width = len(str(node_count))
        node_lines = []
        for i, node in enumerate(self):
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

    __slots__ = ("doc_string", "name", "type", "value")

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
        if self.type == _enums.AttributeType.GRAPH:
            return textwrap.indent("\n" + str(self.value), " " * 4)
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, {self.type!r}, {self.value!r})"


# NOTE: The following functions are just for convenience
def AttrFloat32(name: str, value: float, doc_string: str | None = None) -> Attr:
    """Create a float attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.FLOAT,
        value,
        doc_string=doc_string,
    )


def AttrInt64(name: str, value: int, doc_string: str | None = None) -> Attr:
    """Create an int attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.INT,
        value,
        doc_string=doc_string,
    )


def AttrString(name: str, value: str, doc_string: str | None = None) -> Attr:
    """Create a str attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.STRING,
        value,
        doc_string=doc_string,
    )


def AttrTensor(
    name: str, value: _protocols.TensorProtocol, doc_string: str | None = None
) -> Attr:
    """Create a tensor attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.TENSOR,
        value,
        doc_string=doc_string,
    )


def AttrGraph(name: str, value: Graph, doc_string: str | None = None) -> Attr:
    """Create a graph attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.GRAPH,
        value,
        doc_string=doc_string,
    )


def AttrFloat32s(name: str, value: Sequence[float], doc_string: str | None = None) -> Attr:
    """Create a float sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.FLOATS,
        value,
        doc_string=doc_string,
    )


def AttrInt64s(name: str, value: Sequence[int], doc_string: str | None = None) -> Attr:
    """Create an int sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.INTS,
        value,
        doc_string=doc_string,
    )


def AttrStrings(name: str, value: Sequence[str], doc_string: str | None = None) -> Attr:
    """Create a string sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.STRINGS,
        value,
        doc_string=doc_string,
    )


def AttrTensors(
    name: str, value: Sequence[_protocols.TensorProtocol], doc_string: str | None = None
) -> Attr:
    """Create a tensor sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.TENSORS,
        value,
        doc_string=doc_string,
    )


def AttrGraphs(name: str, value: Sequence[Graph], doc_string: str | None = None) -> Attr:
    """Create a graph sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.GRAPHS,
        value,
        doc_string=doc_string,
    )


# NOTE: SparseTensor should be a sparse tensor proto
def AttrSparseTensor(
    name: str, value: _protocols.SparseTensorProtocol, doc_string: str | None = None
) -> Attr:
    """Create a sparse tensor attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.SPARSE_TENSOR,
        value,
        doc_string=doc_string,
    )


def AttrSparseTensors(
    name: str, value: Sequence[_protocols.SparseTensorProtocol], doc_string: str | None = None
) -> Attr:
    """Create a sparse tensor sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.SPARSE_TENSORS,
        value,
        doc_string=doc_string,
    )


@dataclasses.dataclass
class TypeAndShape:
    """Type and shape.

    Useful for constructing a type proto.
    """

    type: _protocols.TypeProtocol | None
    shape: Shape | None


def AttrTypeProto(name: str, value: TypeAndShape, doc_string: str | None = None) -> Attr:
    """Create a type attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.TYPE_PROTO,
        value,
        doc_string=doc_string,
    )


def AttrTypeProtos(
    name: str, value: Sequence[TypeAndShape], doc_string: str | None = None
) -> Attr:
    """Create a type sequence attribute."""
    # NOTE: The function name is capitalized to maintain API backward compatibility.
    return Attr(
        name,
        _enums.AttributeType.TYPE_PROTOS,
        value,
        doc_string=doc_string,
    )
