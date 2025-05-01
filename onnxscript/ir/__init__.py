# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""In-memory intermediate representation for ONNX graphs."""

__all__ = [
    # Modules
    "serde",
    "traversal",
    "convenience",
    "external_data",
    "tape",
    # IR classes
    "Tensor",
    "ExternalTensor",
    "StringTensor",
    "LazyTensor",
    "SymbolicDim",
    "Shape",
    "TensorType",
    "OptionalType",
    "SequenceType",
    "SparseTensorType",
    "TypeAndShape",
    "Value",
    "Attr",
    "RefAttr",
    "Node",
    "Function",
    "Graph",
    "GraphView",
    "Model",
    # Constructors
    "AttrFloat32",
    "AttrFloat32s",
    "AttrGraph",
    "AttrGraphs",
    "AttrInt64",
    "AttrInt64s",
    "AttrSparseTensor",
    "AttrSparseTensors",
    "AttrString",
    "AttrStrings",
    "AttrTensor",
    "AttrTensors",
    "AttrTypeProto",
    "AttrTypeProtos",
    "Input",
    # Protocols
    "ArrayCompatible",
    "DLPackCompatible",
    "TensorProtocol",
    "ValueProtocol",
    "ModelProtocol",
    "NodeProtocol",
    "GraphProtocol",
    "GraphViewProtocol",
    "AttributeProtocol",
    "ReferenceAttributeProtocol",
    "SparseTensorProtocol",
    "SymbolicDimProtocol",
    "ShapeProtocol",
    "TypeProtocol",
    "MapTypeProtocol",
    "FunctionProtocol",
    # Enums
    "AttributeType",
    "DataType",
    # Types
    "OperatorIdentifier",
    # Protobuf compatible types
    "TensorProtoTensor",
    # Conversion functions
    "from_proto",
    "to_proto",
    # Convenience constructors
    "tensor",
    "node",
    # Pass infrastructure
    "passes",
    # IO
    "load",
    "save",
]

from onnxscript.ir import convenience, external_data, passes, serde, tape, traversal
from onnxscript.ir._convenience._constructors import node, tensor
from onnxscript.ir._core import (
    Attr,
    AttrFloat32,
    AttrFloat32s,
    AttrGraph,
    AttrGraphs,
    AttrInt64,
    AttrInt64s,
    AttrSparseTensor,
    AttrSparseTensors,
    AttrString,
    AttrStrings,
    AttrTensor,
    AttrTensors,
    AttrTypeProto,
    AttrTypeProtos,
    ExternalTensor,
    Function,
    Graph,
    GraphView,
    Input,
    LazyTensor,
    Model,
    Node,
    OptionalType,
    RefAttr,
    SequenceType,
    Shape,
    SparseTensorType,
    StringTensor,
    SymbolicDim,
    Tensor,
    TensorType,
    TypeAndShape,
    Value,
)
from onnxscript.ir._enums import (
    AttributeType,
    DataType,
)
from onnxscript.ir._io import load, save
from onnxscript.ir._protocols import (
    ArrayCompatible,
    AttributeProtocol,
    DLPackCompatible,
    FunctionProtocol,
    GraphProtocol,
    GraphViewProtocol,
    MapTypeProtocol,
    ModelProtocol,
    NodeProtocol,
    OperatorIdentifier,
    ReferenceAttributeProtocol,
    ShapeProtocol,
    SparseTensorProtocol,
    SymbolicDimProtocol,
    TensorProtocol,
    TypeProtocol,
    ValueProtocol,
)
from onnxscript.ir.serde import TensorProtoTensor, from_proto, to_proto


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
