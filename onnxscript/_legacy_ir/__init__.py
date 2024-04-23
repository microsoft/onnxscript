from __future__ import annotations

import dataclasses
from collections import deque
from typing import List, Tuple, Union

import numpy as np
import onnx


class Unknown:
    """A special value used to indicate that a value is not a statically known constant.

    We use this instead of None because None is a valid constant value (since ONNX
    supports the Optional type).
    """

    instance = None

    def __init__(self) -> None:
        if Unknown.instance is not None:
            raise ValueError("Unknown.instance is already set")
        Unknown.instance = self


# Singleton instance of Unknown
unknown = Unknown()
NotConstant = unknown

# ConcreteValue: This type represents constant values that an ONNX variable can take.
# TODO: Extend this to a recursive type to handle lists of tensors, etc., support optionals,
# maps, etc.
# TODO (rama): The value is sometimes stored as a numpy array, and sometimes as an ONNX TensorProto.
# A uniform representation would be helpful, but we should avoid unnecessary conversions for
# large tensors. Should be cleaned up in the new IR.
ConcreteValue = Union[onnx.TensorProto, np.ndarray, Unknown, None]

# SymbolicValue: This information is used to enable partial-evaluation and specialization
# of sequence operations, as well as elimination of redundant Identity ops.
# The symbolic value of a variable X can be:
# - a string with the value "Y", indicating that "X" is a copy of "Y"
# - a list of strings, indicating that "X" is a list of tensors, with their symbolic values
# Eg., the symbolic value ["A", "B", "C"] indicates that the value of X is equal to
# "SequenceConstruct(A, B, C)".
# TODO: Technically, SymbolicValue should be a recursive type to handle lists of lists of
# tensors, etc. However, we currently only handle lists of tensors.

SymbolicValue = Union[str, List[str]]

FunctionId = Tuple[str, str, str]


def get_function_id(function: onnx.FunctionProto) -> FunctionId:
    return (function.domain, function.name, getattr(function, "overload", ""))


def get_function_id_from_node(node: onnx.NodeProto) -> FunctionId:
    return (node.domain, node.op_type, getattr(node, "overload", ""))


@dataclasses.dataclass
class StaticValueInfo:
    name: str
    value: ConcreteValue = NotConstant
    type: onnx.TypeProto | None = None
    symbolic_value: SymbolicValue | None = None

    def is_copy(self) -> bool:
        return isinstance(self.symbolic_value, str)

    def tensor_shape_proto(self) -> onnx.TensorShapeProto | None:
        """Returns the shape of a tensor or None.

        A return value of None could mean that the type is unknown or that the type is not a tensor
        or that the tensor shape (that is, even the rank) is unknown.
        """
        type = self.type
        if type and type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
            return type.tensor_type.shape
        return None

    @property
    def shape(self) -> list[str | int | None] | None:
        """Returns the shape in a list.

        Str means that the shape is dynamic.
        """
        type = self.type
        if type and type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
            dims = []
            for dim in type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    dims.append(dim.dim_param)
                elif dim.HasField("dim_value"):
                    dims.append(dim.dim_value)
                else:
                    dims.append(None)
            return dims
        if self.value_as_np_array is not None:
            return list(self.value_as_np_array.shape)
        return None

    @property
    def element_type(self) -> int | None:
        """Returns the element type of a tensor, or None if type is not known or is not a tensor."""
        type = self.type
        if type and type.HasField("tensor_type"):
            return type.tensor_type.elem_type
        return None

    def identity_merge_from(self, other: StaticValueInfo) -> None:
        """Merge the value of other into self.

        This models the effect of an identity (copy) operation.
        This will update static-analysis information based on incoming value.
        """
        if not isinstance(other, StaticValueInfo):
            raise TypeError(f"Cannot merge {other} into {self}.")
        if other.value is not NotConstant:
            self.value = other.value
        # TODO: merge and combine best shape information from both types.
        if other.tensor_shape_proto() is not None and other.element_type is not None:
            self.type = other.type
        # We cannot copy symbolic value across different scopes.

    # WIP: Extensions towards new IR: Note that the default construction of StaticValueInfo
    # does not fill in the following fields. These fields are filled in by the IRBuilder
    # which constructs the IR from the ONNX model.
    node: Node | None = None
    uses: list[Node] = dataclasses.field(default_factory=list)
    output_index: int | None = None
    is_output: bool = False

    @property
    def const_value(self) -> ConcreteValue:
        return self.value

    @property
    def value_as_np_array(self) -> np.ndarray | None:
        if isinstance(self.value, np.ndarray):
            return self.value
        if isinstance(self.value, onnx.TensorProto):
            return onnx.numpy_helper.to_array(self.value)
        return None

    def def_node(self) -> Node | None:
        return self.node

    def def_index(self) -> int:
        return self.output_index  # type: ignore[return-value]

    def is_same_as(self, other: StaticValueInfo) -> bool:
        """Returns true if this value represents the same IR object as the other value.

        This is *not* value-equality, but rather object-equality.
        """
        return self is other

    def __str__(self) -> str:
        shape = self.shape
        if shape is not None:
            shape = [str(dim) for dim in shape]
            shape_str = f"[{', '.join(shape)}]"  # type: ignore[arg-type]
        else:
            shape_str = "None"
        return (
            f"StaticValueInfo({self.name}, shape:{shape_str}, dtype:{self.element_type}, "
            f"{'has const value' if self.value is not unknown else 'no const value'}.)"
        )


Value = StaticValueInfo


class Model:
    def __init__(self) -> None:
        self.gen_var_counter: int = 0

    def set(
        self,
        model_proto: onnx.ModelProto,
        graph: Graph,
        functions: list[Function],
        version_map: dict[str, int],
    ) -> None:
        """TODO. This is a temporary patch."""
        self.original_model_proto = model_proto
        self.graph = graph
        self.functions = functions
        self.version_map = version_map

    def make_new_name(self):
        # Temporary hack.
        self.gen_var_counter += 1
        return f"_gen_{self.gen_var_counter}"

    def __str__(self) -> str:
        # TODO: Naive string representation for debugging. Need to improve this.
        return "\n".join(
            [
                f"ModelGraph: {self.graph}",
                f"Functions: {self.functions}",
                f"VersionMap: {self.version_map}",
            ]
        )


class Graph:
    def __init__(self, graph_proto: onnx.GraphProto):
        self.original_graph_proto = graph_proto
        self.nodes: deque[Node] = deque()
        self.values: dict[str, Value] = {}

    @property
    def name(self) -> str:
        return self.original_graph_proto.name

    def __str__(self) -> str:
        return "\n".join(
            [
                "Graph",
                f"Nodes: {[str(n) for n in self.nodes]}",
                f"Values: {[str(v) for v in self.values]}",
            ]
        )

    @property
    def input_names(self) -> list[str]:
        return [_.name for _ in self.original_graph_proto.input]

    @property
    def output_names(self) -> list[str]:
        return [_.name for _ in self.original_graph_proto.output]


class Function:
    def __init__(self, function_proto: onnx.FunctionProto):
        self.original_function_proto = function_proto
        self.nodes = deque()  # type: ignore[var-annotated]
        self.values = {}  # type: ignore[var-annotated]

    @property
    def id(self) -> FunctionId:
        return (self.domain, self.name, self.overload)

    @property
    def domain(self) -> str:
        return self.original_function_proto.domain

    @property
    def name(self) -> str:
        return self.original_function_proto.name

    @property
    def overload(self) -> str:
        return getattr(self.original_function_proto, "overload", "")

    def __str__(self) -> str:
        return "\n".join(
            [
                "Function",
                f"Nodes: {[str(n) for n in self.nodes]}",
                f"Values: {[str(v) for v in self.values]}",
            ]
        )


class RefAttr:
    def __init__(self, name: str, ref_attr_name: str, type) -> None:
        self.name = name
        self.ref_attr_name = ref_attr_name
        self.type = type

    def to_proto(self) -> onnx.AttributeProto:
        attr_proto = onnx.AttributeProto()
        attr_proto.name = self.name
        attr_proto.ref_attr_name = self.ref_attr_name
        attr_proto.type = self.type
        return attr_proto


class Node:
    def __init__(
        self,
        node_proto: onnx.NodeProto,
        populate_io: bool = False,
    ) -> None:
        self.original_node_proto = node_proto
        self.domain: str = node_proto.domain
        self.version: int | None = None
        self.op_type: str = node_proto.op_type
        if populate_io:
            self.inputs: list[Value | None] = [Value(i) for i in node_proto.input]
            self.outputs: list[Value | None] = [Value(i) for i in node_proto.output]
        else:
            self.inputs: list[Value | None] = []  # type: ignore[no-redef]
            self.outputs: list[Value | None] = []  # type: ignore[no-redef]
        # TODO: attributes are never populated.
        self.attributes: dict[str, int | float | RefAttr | Graph | list[Graph]] = {}

    def __repr__(self) -> str:
        return (
            f"{self.op_type}({','.join(self.original_node_proto.input)})"
            f"->{','.join(self.original_node_proto.output)}"
        )

    @property
    def name(self) -> str:
        return self.original_node_proto.name

    @property
    def input_names(self):
        return self.original_node_proto.input

    @property
    def output_names(self):
        return self.original_node_proto.output

    @property
    def attribute(self):
        return self.original_node_proto.attribute

    def set_version_if_custom_op(self, version_map: dict[str, int]) -> None:
        if self.domain != "" and self.domain in version_map:
            self.version = version_map[self.domain]

    def get_attribute(self, name: str) -> int | float | None:
        return self.attributes.get(name, None)  # type: ignore[return-value]

    def __str__(self) -> str:
        return "\n".join(
            [
                "Node",
                f"OpType: {self.op_type}",
                f"Inputs: {self.inputs}",
                f"Outputs: {self.outputs}",
                f"Attributes: {self.attributes}",
            ]
        )
