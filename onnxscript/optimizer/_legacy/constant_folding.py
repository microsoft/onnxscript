# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import onnx
import onnx.reference.ops

import onnxscript._legacy_ir as ir
import onnxscript.optimizer._constant_folding as _constant_folding
from onnxscript._legacy_ir import visitor
from onnxscript.optimizer._legacy import evaluator
from onnxscript.utils.utils import (
    is_control_flow_op,
    is_onnx_domain,
)

logger = logging.getLogger(__name__)

# Ops excluded from constant-propagation:
# * Random ops, which are not deterministic (checked below)
# * Control flow ops (checked by presence of graph-attribute)

onnx_domain = frozenset({"", "onnx.ai"})


def is_non_deterministic_op(node: onnx.NodeProto) -> bool:
    non_deterministic_ops = _constant_folding.non_deterministic_ops
    return node.op_type in non_deterministic_ops and is_onnx_domain(node.domain)


def is_constant_op(node: onnx.NodeProto) -> bool:
    return node.op_type in {"Constant", "ConstantOfShape"} and is_onnx_domain(node.domain)


class ConstantFolder(visitor.FunctionCallsiteProtoTransformer):
    def __init__(
        self,
        registry: evaluator.PartialEvaluatorRegistry,
        external_data_folder: str,
        *,
        do_shape_inference: bool,
    ) -> None:
        self.registry = registry
        # TODO: make evaluator a parameter
        self.evaluate = evaluator.reference_evaluator.evaluate
        self._do_shape_inference = do_shape_inference
        self._init()
        super().__init__(external_data_folder, do_shape_inference=do_shape_inference)

    def _init(self) -> None:
        self.counts = {}
        self.sizes = {}

    def add_count(self, op: str, size: int = 1):
        self.counts[op] = self.counts.get(op, 0) + 1
        self.sizes[op] = self.sizes.get(op, 0) + size

    def foldable_value(self, name: str, value):
        """Checks if a runtime-constant can and should be folded into the graph.

        We fold constants only if they are tensors (not lists of tensors, for example)
        and have size below desired limit.
        """
        if value is ir.NotConstant:
            return None

        if not isinstance(value, np.ndarray):
            # ONNX does not have a way to represent non-tensor constants, eg. a sequence.
            # So, a constant-value of type sequence is not folded, but it can be used
            # to optimize subsequent operations when possible.
            logger.info(
                "Skip storing constant folded value %s due to unsupported type %s.",
                name,
                type(value),
            )
            return None

        if value.nbytes > _constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT:
            logger.info(
                "Skip storing constant folded nvalue %s due to large size %s.",
                name,
                value.nbytes,
            )
            return None

        return onnx.numpy_helper.from_array(value, name)

    def new_constant(self, name, value):
        if isinstance(value, (int, float, np.ScalarType)):
            value = np.array(value)

        info = self.lookup_or_create(name)
        info.value = value

        tensor = self.foldable_value(name, value)
        if tensor is None:
            return None

        logger.debug(
            "New constant for value %s dtype: %s shape: %s",
            name,
            value.dtype,
            value.shape,
        )
        info.type = onnx.helper.make_tensor_type_proto(
            onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        node = onnx.helper.make_node("Constant", inputs=[], outputs=[name], value=tensor)
        return [node]

    def convert_attributes(self, attributes: Sequence[onnx.AttributeProto]) -> dict[str, Any]:
        if self.scopes.current_scope().current_function_scope():
            # Need to resolve ref_attr_name if inside a function.
            attr_dict = {}
            for attribute in attributes:
                concrete_attribute = (
                    self.lookup_ref_attribute(attribute.ref_attr_name)
                    if attribute.ref_attr_name
                    else attribute
                )
                if concrete_attribute is None:
                    continue
                attr_dict[attribute.name] = onnx.helper.get_attribute_value(concrete_attribute)
            return attr_dict
        return {attr.name: onnx.helper.get_attribute_value(attr) for attr in attributes}

    def replace_copy(self, node: onnx.NodeProto) -> None:
        for i in range(len(node.input)):
            input = self.get_input(node, i)
            if input is not None and input.is_copy():
                old_value = self.lookup_or_create(input.name)
                assert isinstance(input.symbolic_value, str)
                new_value = self.lookup_or_create(input.symbolic_value)
                # Merge meta info. It is important to do if the new value
                # is created by evaluator, and thus carries zero meta info.
                # Since this is a copy, the meta info should be the same.
                new_value.identity_merge_from(old_value)
                node.input[i] = input.symbolic_value

    def process_function_outputs(self, function: onnx.FunctionProto) -> bool:
        # Resolve copy for function subgraph output.
        # Avoid copy of function subgraph input, because it is illegal for a direct edge
        # from function input to function output.
        prohibited_value_set = set(function.input)
        updated = False
        for i, output_name in enumerate(function.output):
            output = self.lookup(output_name)
            if (
                output is not None
                and output.is_copy()
                and output.symbolic_value not in prohibited_value_set
            ):
                old_value = self.lookup_or_create(output.name)
                assert isinstance(output.symbolic_value, str)
                new_value = self.lookup_or_create(output.symbolic_value)
                new_value.identity_merge_from(old_value)
                function.output[i] = output.symbolic_value
                updated = True
        return updated

    def process_node(self, node: onnx.NodeProto) -> Sequence[onnx.NodeProto] | None:
        self.replace_copy(node)

        super().process_node(node)

        inputs = [self.lookup(x) for x in node.input]
        attrs = self.convert_attributes(node.attribute)

        domain = node.domain
        op = node.op_type
        version = self.lookup_version(domain)

        # if any(x is Undefined for x in inputs):
        #     return None
        # Above check ensures that none of the optimizations below need to handle
        # undefined inputs

        op_optimizers = self.registry.lookup_evaluators(domain, op, version)
        for optimizer in op_optimizers:
            assert optimizer
            output = optimizer(self, node)
            if output is None:
                continue
            if isinstance(output, list):
                return output
            else:
                # Currently handles single output only
                self.add_count(node.op_type, output.size)
                return self.new_constant(node.output[0], output)

        if is_control_flow_op(node) or is_non_deterministic_op(node):
            return None

        input_values = [x.value if x is not None else None for x in inputs]
        if any(x is ir.NotConstant for x in input_values):
            return None

        input_types = [x.type for x in inputs if x is not None]

        def is_excluded_type(type_proto: onnx.TypeProto | None) -> bool:
            if type_proto is None:
                return True
            if type_proto.HasField("tensor_type"):
                return type_proto.tensor_type.elem_type in {
                    onnx.TensorProto.BFLOAT16,
                    onnx.TensorProto.FLOAT8E4M3FN,
                    onnx.TensorProto.FLOAT8E4M3FNUZ,
                    onnx.TensorProto.FLOAT8E5M2,
                    onnx.TensorProto.FLOAT8E5M2FNUZ,
                }
            return False

        if any(is_excluded_type(x) for x in input_types):
            return None

        outputs = self.evaluate(domain, op, version, *input_values, **attrs)
        # TODO: what if evaluated value is None?
        if outputs is None:
            return None
        if len(node.output) == 1 and not isinstance(outputs, (tuple, list)):
            replacement = self.new_constant(node.output[0], outputs)
            if is_constant_op(node):
                return None
            self.add_count(op, outputs.size)
            return replacement
        else:
            logger.warning("Skipping constant folding for op %s with multiple outputs.", op)
        return None

    def process_function_node(
        self, node: onnx.NodeProto
    ) -> tuple[list[onnx.NodeProto] | None, onnx.FunctionProto | None]:
        self.replace_copy(node)

        _, new_function = super().process_function_node(node)

        # Replace function node with Constant if all outputs are constants
        ir_values = [self.lookup(output_name) for output_name in node.output]
        tensors = [
            self.foldable_value(output_name, ir_value.value if ir_value is not None else None)
            for output_name, ir_value in zip(node.output, ir_values)
        ]
        if all(tensor is not None for tensor in tensors):
            replacements = []
            for output_name, tensor in zip(node.output, tensors):
                newnode = onnx.helper.make_node(
                    "Constant", inputs=[], outputs=[output_name], value=tensor
                )
                replacements.append(newnode)
            logger.debug(
                "Function node replacements: node %s %s (%s/%s)",
                node.name,
                [replacement.output for replacement in replacements],
                len(replacements),
                len(node.output),
            )
            return replacements, new_function
        return None, new_function

    def visit_model(self, model: onnx.ModelProto) -> None:
        self._init()

        super().visit_model(model)


def fold_constants(
    model: onnx.ModelProto,
    external_data_folder: str = "",
    *,
    onnx_shape_inference: bool = False,
) -> bool:
    """
    Applies constant folding optimization to the model.
    Returns true iff the model was modified.
    """
    folder = ConstantFolder(
        evaluator.registry,
        external_data_folder,
        do_shape_inference=onnx_shape_inference,
    )
    folder.visit_model(model)
    for op in folder.counts:
        logger.info(
            "Constant-folded '%s' %s times, with %s size.",
            op,
            folder.counts[op],
            folder.sizes[op],
        )
    return folder.modified
