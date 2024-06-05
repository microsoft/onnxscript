# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Dict, Mapping, Optional, Sequence, Set

import onnx
import onnx.defs

import onnxscript

logger = logging.getLogger(__name__)

_ONNX_DTYPE_TO_ONNX_TENSOR_TYPE_STR: Mapping[int, str] = {
    onnx.TensorProto.FLOAT: "tensor(float)",
    onnx.TensorProto.UINT8: "tensor(uint8)",
    onnx.TensorProto.INT8: "tensor(int8)",
    onnx.TensorProto.UINT16: "tensor(uint16)",
    onnx.TensorProto.INT16: "tensor(int16)",
    onnx.TensorProto.INT32: "tensor(int32)",
    onnx.TensorProto.INT64: "tensor(int64)",
    onnx.TensorProto.STRING: "tensor(string)",
    onnx.TensorProto.BOOL: "tensor(bool)",
    onnx.TensorProto.FLOAT16: "tensor(float16)",
    onnx.TensorProto.DOUBLE: "tensor(double)",
    onnx.TensorProto.COMPLEX64: "tensor(complex64)",
    onnx.TensorProto.COMPLEX128: "tensor(complex128)",
    onnx.TensorProto.UINT32: "tensor(uint32)",
    onnx.TensorProto.UINT64: "tensor(uint64)",
    onnx.TensorProto.BFLOAT16: "tensor(bfloat16)",
}

_ONNX_ATTR_TYPE_TO_ONNX_TENSOR_TYPE_STR: Mapping[int, str] = {
    onnx.AttributeProto.UNDEFINED: "undefined",
    onnx.AttributeProto.FLOAT: "tensor(float)",
    onnx.AttributeProto.INT: "tensor(int64)",
    onnx.AttributeProto.STRING: "tensor(string)",
    onnx.AttributeProto.FLOATS: "tensor(float)",
    onnx.AttributeProto.INTS: "tensor(int64)",
}


class TypeConstraint:
    """Type constraint shared by multiple values."""

    name: str
    type_strs: Set[str]
    values: Set[Value]

    def __init__(self, name: str, type_strs: Set[str]):
        self.name = name
        self.type_strs = type_strs
        self.values = set()

    def merge_type_constraint(self, other: TypeConstraint):
        """Merge two type constraints.

        Typically this process tightens the type constraints.

        For example, consider two values with different existing type constraints:

            input1: TypeConstraint(name="input1", type_strs={"tensor(float)", "tensor(int64)"})
            input2: TypeConstraint(name="input2", type_strs={"tensor(float)", "tensor(int32)"})

        Now consider they are two inputs to `Add` node, where the op schema is:

            inputs: [T, T]
            outputs: [T]
            T: ['tensor(uint8)', ..., 'tensor(float)', 'tensor(int64)', 'tensor(int32)']

        `input1` and `input2` must now bound to the same type constraint, hence the result is
        the intersection of the two existing type constraints with the op schema type constraint:

            input1: TypeConstraint(name="input1.input2", type_strs={"tensor(float)"})
            input2: TypeConstraint(name="input1.input2", type_strs={"tensor(float)"})
        """
        if self is other:
            return
        self.type_strs.intersection_update(other.type_strs)
        self.values = self.values.union(other.values)
        for meta in self.values:
            meta.type_constraint = self
        self.name = f"{self.name}.{other.name}"

    def bind_value(self, value: Value):
        """Bind a value to this type constraint.

        If the value does not have a type constraint, set it to this type constraint.
        Otherwise, merge the two type constraints.
        """
        if value.type_constraint is not None:
            self.merge_type_constraint(value.type_constraint)
        else:
            self.values.add(value)
            value.type_constraint = self

    def __repr__(self):
        value_names = [value.name for value in self.values]
        return f"TypeConstraint(name={self.name}, type_strs={self.type_strs}, values={value_names})"


class Value:
    """Represents a value in the graph. Associated with a TypeConstraint."""

    def __init__(self, name: str):
        self.type_constraint: Optional[TypeConstraint] = None
        self.name: str = name

    def merge_type_constraint(self, other: Value):
        if other.type_constraint is not None:
            other.type_constraint.bind_value(self)
        elif self.type_constraint is not None:
            self.type_constraint.bind_value(other)
        else:
            raise ValueError(
                f"Cannot merge two values without type constraints. {self} and {other}"
            )

    def __repr__(self) -> str:
        return f"Value(name={self.name}, type_constraint={self.type_constraint})"


@dataclasses.dataclass
class OnnxFunctionTypeConstraints:
    input_type_constraints: Dict[str, Optional[TypeConstraint]]
    output_type_constraints: Dict[str, Optional[TypeConstraint]]
    intermediate_type_constraints: Dict[str, Optional[TypeConstraint]]

    def __repr__(self):
        repr_strs = [
            "Type Constraints:",
            "  Inputs: ",
        ]
        repr_strs += [
            f"    {name}: {type_constraint.name}"
            if type_constraint is not None
            else f"    {name}: None"
            for name, type_constraint in self.input_type_constraints.items()
        ]
        repr_strs += [
            "  Outputs: ",
        ]
        repr_strs += [
            f"    {name}: {type_constraint.name}"
            if type_constraint is not None
            else f"    {name}: None"
            for name, type_constraint in self.output_type_constraints.items()
        ]
        repr_strs += [
            "  Type Constraints: ",
        ]
        # Trick to get unique type constraints but maintain the order.
        ordered_unique_type_constraints = dict.fromkeys(self.input_type_constraints.values())
        ordered_unique_type_constraints.update(
            dict.fromkeys(self.output_type_constraints.values())
        )
        repr_strs += [
            f"    {type_constraint.name}: {type_constraint.type_strs}"
            for type_constraint in ordered_unique_type_constraints
            if type_constraint is not None
        ]

        if self.intermediate_type_constraints:
            repr_strs += ["  Intermediate Values: "]
            repr_strs += [
                f"    {name}: {type_constraint.name}"
                if type_constraint is not None
                else f"    {name}: None"
                for name, type_constraint in self.intermediate_type_constraints.items()
            ]

            repr_strs += [
                "  Intermediate Type Constraints: ",
            ]
            ordered_unique_type_constraints = dict.fromkeys(
                self.intermediate_type_constraints.values()
            )
            repr_strs += [
                f"    {type_constraint.name}: {type_constraint.type_strs}"
                for type_constraint in ordered_unique_type_constraints
                if type_constraint is not None
            ]

        return "\n".join(repr_strs)


class TypeConstraintDeducer:
    def __init__(self, onnx_function: onnxscript.OnnxFunction):
        self.onnx_function = onnx_function
        self.values: Dict[str, Value] = {}

    def type_constraints(self, signature_only: bool = True) -> OnnxFunctionTypeConstraints:
        """Retrieve deduced type constraints for the ONNX function."""
        if not self.values:
            raise ValueError("Must call deduce() first")

        input_type_constraints = {
            name: self.values[name].type_constraint for name in self.function_proto.input
        }
        output_type_constraints = {
            name: self.values[name].type_constraint for name in self.function_proto.output
        }
        intermediate_type_constraints = (
            {}
            if signature_only
            else {name: value.type_constraint for name, value in self.values.items()}
        )

        # Rename type constraints to T0, T1, T2, ...
        _seen_type_constraints: Set[TypeConstraint] = set()
        for type_constraint in (
            *input_type_constraints.values(),
            *output_type_constraints.values(),
            *intermediate_type_constraints.values(),
        ):
            if type_constraint is not None and type_constraint not in _seen_type_constraints:
                type_constraint.name = f"T{len(_seen_type_constraints)}"
                _seen_type_constraints.add(type_constraint)

        return OnnxFunctionTypeConstraints(
            input_type_constraints, output_type_constraints, intermediate_type_constraints
        )

    def deduce(self, signature_only: bool = True) -> OnnxFunctionTypeConstraints:
        """Deduce type constraints for all values in the graph.

        Args:
            signature_only: If True, return deduce type constraints only for function signature.
                Otherwise, return deduced type constraints for all values in the graph.
        """
        self.values = {}
        self.function_proto = self.onnx_function.to_function_proto()
        self.opset_version = self.function_proto.opset_import[0].version

        param_schemas = self.onnx_function.param_schemas()
        for param_schema in param_schemas:
            if param_schema.is_input:
                self.values[param_schema.name] = Value(param_schema.name)

        for node in self.function_proto.node:
            self._process_node(node)

        return self.type_constraints(signature_only)

    def _bind_signature(
        self,
        node: onnx.NodeProto,
        param_names: Sequence[str],
        param_schemas: Sequence[onnx.defs.OpSchema.FormalParameter],
        op_type_constraints: Dict[str, TypeConstraint],
        is_output: bool = False,
    ):
        param_schemas = list(param_schemas)
        # If the last parameter is variadic, duplicate it to match the number of parameters.
        if (
            len(param_schemas) < len(param_names)
            and param_schemas[-1].option == onnx.defs.OpSchema.FormalParameterOption.Variadic
        ):
            param_schemas += [param_schemas[-1]] * (len(param_names) - len(param_schemas))
        for name, schema in zip(param_names, param_schemas):
            if is_output:
                if name in self.values:
                    raise ValueError(f"Output {name} already exists.")
                value = Value(name)
                self.values[name] = value
            else:
                if not name:
                    # Skip optional inputs
                    continue
                value = self.values[name]
            if (new_type_constraint := op_type_constraints.get(schema.type_str)) is None:
                # parameter is annotated with type string instead of type constraint.
                # Create individual type constraint.
                new_type_constraint = TypeConstraint(
                    name=f"T_{name}",
                    type_strs={schema.type_str},
                )
            if not schema.is_homogeneous:
                # Parameter is not homogeneous, this appears with variadic parameters.
                # Meaning the type constraint is not shared amongst them.
                # Creating a type constraint copy.
                new_type_constraint = TypeConstraint(
                    name=f"{schema.name}_{name}",
                    type_strs=new_type_constraint.type_strs,
                )
            prev_value_constraint = copy.deepcopy(value.type_constraint)
            new_type_constraint.bind_value(value)
            if prev_value_constraint is not None and len(
                prev_value_constraint.type_strs
            ) > len(new_type_constraint.type_strs):
                logger.info(
                    "Type constraint is tightened due to binding %s with parameter %s in node %s(%s)",
                    value.name,
                    schema.name,
                    node.op_type,
                    node.name,
                )
                logger.info("  %s", prev_value_constraint.type_strs)
                logger.info("->")
                logger.info("  %s", new_type_constraint.type_strs)

    def _perform_extra_type_constraint_tightening(self, node: onnx.NodeProto):
        if node.op_type == "If":
            # Step into subgraph for more type constraint deduction.
            subgraph_attr_names = ("then_branch", "else_branch")
            for attr in node.attribute:
                if attr.name not in subgraph_attr_names:
                    continue
                subgraph = attr.g
                for subgraph_node in subgraph.node:
                    self._process_node(subgraph_node)
                for subgraph_output, node_output_name in zip(subgraph.output, node.output):
                    # Type constraint must agree between "then" and "else" branch.
                    self.values[node_output_name].merge_type_constraint(
                        self.values[subgraph_output.name]
                    )
        elif node.op_type in ("Loop", "Scan"):
            # Step into subgraph for more type constraint deduction.
            raise NotImplementedError("Loop/Scan is not supported yet!")
        elif node.op_type == "Constant":
            # Constant type is static and can be inferred from attribute.
            # Tighten the type constraint.
            tensor_attr = node.attribute[0]
            type_constraint_name = f"Constant_{node.name}"
            if tensor_attr.type == onnx.AttributeProto.TENSOR:
                type_constraint = TypeConstraint(
                    type_constraint_name,
                    {_ONNX_DTYPE_TO_ONNX_TENSOR_TYPE_STR[tensor_attr.t.data_type]},
                )
            else:
                type_constraint = TypeConstraint(
                    type_constraint_name,
                    {_ONNX_ATTR_TYPE_TO_ONNX_TENSOR_TYPE_STR[tensor_attr.type]},
                )
            type_constraint.bind_value(self.values[node.output[0]])
        elif node.op_type == "Cast":
            to_attr = node.attribute[0]
            if not to_attr.ref_attr_name:
                # Cast to a static type.
                # Tighten the type constraint
                type_constraint = TypeConstraint(
                    f"Cast_{node.name}",
                    {_ONNX_DTYPE_TO_ONNX_TENSOR_TYPE_STR[node.attribute[0].i]},
                )
                type_constraint.bind_value(self.values[node.output[0]])

    def _process_node(self, node: onnx.NodeProto):
        if node.domain and node.domain != "onnx":
            raise NotImplementedError("Nested function is not supported yet.")

        if node.op_type in ("Loop", "Scan"):
            # Step into subgraph for more type constraint deduction.
            raise NotImplementedError("Loop/Scan is not supported yet!")

        # Creating new type constraints from op schema
        op_schema = onnx.defs.get_schema(
            node.op_type, max_inclusive_version=self.opset_version, domain=node.domain
        )
        op_type_constraints = {
            ts.type_param_str: TypeConstraint(
                name=f"{ts.type_param_str}_{node.name}", type_strs=set(ts.allowed_type_strs)
            )
            for ts in op_schema.type_constraints
        }

        # Binding new type constraints to input values.
        self._bind_signature(node, node.input, op_schema.inputs, op_type_constraints)
        # Creating new values for outputs, and bind with type constraints.
        self._bind_signature(
            node,
            node.output,
            op_schema.outputs,
            op_type_constraints,
            is_output=True,
        )

        # Postprocess a few special cases
        self._perform_extra_type_constraint_tightening(node)


def deduce_type_constraints(
    onnx_function: onnxscript.OnnxFunction, signature_only: bool = True
) -> OnnxFunctionTypeConstraints:
    """Deduce type constraints for an ONNX function.

    * Expects Tensors to be pre-annotated as `TensorType`.
    * Expects Attributes to be pre-annotated correctly.
    * Produces the least strict type constraints allowed by deducing from OpSchema in FunctionProto.
    * Produces explanation for type constraint tightening.
    * `TracedOnnxFunction` is not planned.

    Args:
        onnx_function: ONNX function to generate type constraints for.
        signature_only: Whether to only generate type constraints for the function signature.

    Returns:
        Type constraints for the ONNX function.
    """
    logger.info("Deducing type constraints for %s", onnx_function.name)
    return TypeConstraintDeducer(onnx_function).deduce(signature_only=signature_only)
