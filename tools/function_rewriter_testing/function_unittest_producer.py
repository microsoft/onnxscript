"""Fuction fusion unittest producer.

Takes in a full model, function keyword, and example inputs, produces unit model protos
that contains only a single node calling the target function proto.

- All initializers are lifted as model inputs.
- Example inputs and outputs are saved as test data for each unit model proto.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.inliner
import onnxruntime
from onnx import helper as onnx_helper
from onnx import numpy_helper

from onnxscript import _legacy_ir as ir
from onnxscript._legacy_ir import visitor
from onnxscript.utils import evaluation_utils, utils

logger = logging.getLogger(__name__)


# Copied from common.py from pytorch torchbench
def save_tensor_data(numpy_tensor, output_path: str):
    proto_tensor = numpy_helper.from_array(numpy_tensor)
    with open(output_path, "wb") as f:
        f.write(proto_tensor.SerializeToString())


class FunctionToKeepVisitor(visitor.ProtoVisitorCore):
    def __init__(self, function_keyword):
        self.function_keyword = function_keyword
        self.functions_to_keep = []
        self.in_target_function = False
        self._functions = {}
        super().__init__()

    def visit_function_node(self, node: onnx.NodeProto):
        prev_in_target_function = self.in_target_function
        function_id = ir.get_function_id_from_node(node)
        function = self._functions[function_id]
        if node.op_type.find(self.function_keyword) != -1:
            self.functions_to_keep.append(function_id)
            self.in_target_function = True
        elif prev_in_target_function:
            self.functions_to_keep.append(function_id)

        for subnode in function.node:
            self.visit_node(subnode)

        self.in_target_function = prev_in_target_function

    def process_node(self, node: onnx.NodeProto):
        if visitor.is_local_function_node(node, self._functions):
            return self.visit_function_node(node)
        return None

    def visit_model(self, model: onnx.ModelProto) -> None:
        for function in model.functions:
            self._functions[ir.get_function_id(function)] = function
        super().visit_model(model)


FunctionMetaDict = Dict[Tuple[str, str], Tuple[List[str], List[str]]]


class TargetFunctionMetaVisitor(visitor.ProtoVisitorCore):
    def __init__(self, function_keyword):
        self.function_keyword = function_keyword
        # Map from (domain, name) to (actual_input_names, actual_output_names)
        self.function_meta: FunctionMetaDict = {}
        self._functions = {}
        super().__init__()

    def visit_function_node(self, node: onnx.NodeProto):
        function = self._functions[ir.get_function_id_from_node(node)]
        if node.op_type.find(self.function_keyword) != -1:
            self.function_meta[(function.domain, function.name)] = (
                node.input,
                node.output,
            )
        for subnode in function.node:
            self.visit_node(subnode)

    def process_node(self, node: onnx.NodeProto):
        if visitor.is_local_function_node(node, self._functions):
            return self.visit_function_node(node)
        return None

    def visit_model(self, model: onnx.ModelProto) -> None:
        for function in model.functions:
            self._functions[ir.get_function_id(function)] = function
        super().visit_model(model)


class FunctionProtoProducerWithData(visitor.ProtoVisitor):
    """Fuction fusion unittest producer.

    Creates unit model proto for selected function, as well as example inputs and outputs.

    Utilizes ORT fetch feature.

    Steps as follows:

    - Identify the target function, and all functions called within.
    - Call onnx.inliner to inline all other functions.
    - Identity inputs and outputs to target function calls, construct ort fetch.
    - Run the model with ort fetch to receive example inputs and outputs.
    - For each target function call, construct a unit model proto with example inputs and outputs from previous step.
    """

    def __init__(self, function_keyword: str, model_path: str, output_dir: str):
        self.function_keyword = function_keyword
        self.model_path = model_path
        self.output_dir = output_dir
        self.output_model_basename = function_keyword
        self._functions: dict[ir.FunctionId, onnx.FunctionProto] = {}
        self._unit_model_protos: list[onnx.ModelProto] = []
        self._unit_model_inputs = []  # type: ignore[var-annotated]
        self._unit_model_outputs = []  # type: ignore[var-annotated]
        # Example intermediate data values
        self._named_values: dict[str, np.ndarray] = {}
        super().__init__()

    @property
    def unit_model_protos(self) -> list[onnx.ModelProto]:
        return self._unit_model_protos

    @property
    def unit_model_inputs(self):
        return self._unit_model_inputs

    @property
    def unit_model_outputs(self):
        return self._unit_model_outputs

    def find_all_called_function_protos(
        self, function: onnx.FunctionProto
    ) -> list[onnx.FunctionProto]:
        result: dict[ir.FunctionId, onnx.FunctionProto] = {
            ir.get_function_id(function): function
        }
        for node in function.node:
            if visitor.is_local_function_node(node, self._functions):
                sub_function = self._functions[ir.get_function_id_from_node(node)]
                result.update(
                    {
                        ir.get_function_id(func): func
                        for func in self.find_all_called_function_protos(sub_function)
                    }
                )
        return result.values()  # type: ignore[return-value]

    def _generate_value_info_for_function_value(
        self, value: str, function: onnx.FunctionProto
    ) -> onnx.ValueInfoProto | None:
        value_ir = self.function_shape_env.lookup(function, value)
        if value_ir is None:
            return None
        return self.function_shape_env.save_to_value_info(
            value_ir, *ir.get_function_id(function)
        )

    def _generate_value_info_for_function_values(
        self, function: onnx.FunctionProto
    ) -> list[onnx.ValueInfoProto]:
        value_infos = []
        values = {
            *function.input,
            *function.output,
            *itertools.chain((*node.input, *node.output) for node in function.node),
        }

        for value in values:
            value_info = self._generate_value_info_for_function_value(value, function)
            if value_info is not None:
                value_infos.append(value_info)
        return value_infos

    def create_unit_model_proto(
        self,
        function_proto: onnx.FunctionProto,
        actual_input_value_infos: list[ir.Value | None],
        actual_output_value_infos: list[ir.Value | None],
    ) -> onnx.ModelProto | None:
        unit_model_proto = onnx.ModelProto()
        unit_model_proto.ir_version = self._model_proto.ir_version
        unit_model_proto.producer_name = self._model_proto.producer_name
        unit_model_proto.producer_version = self._model_proto.producer_version
        unit_model_proto.domain = self._model_proto.domain
        unit_model_proto.model_version = self._model_proto.model_version
        unit_model_proto.opset_import.extend(self._model_proto.opset_import)
        graph_proto = unit_model_proto.graph

        for actual_input_value_info, formal_input in zip(
            actual_input_value_infos, function_proto.input
        ):
            if actual_input_value_info is None:
                logger.error(
                    "Value info for input %s is not found. Skip model proto creation for function %s::%s",
                    formal_input,
                    function_proto.domain,
                    function_proto.name,
                )
                return None
            if actual_input_value_info.type is None:
                logger.error(
                    "Value info for input %s has no type. Skip model proto creation for function %s::%s",
                    formal_input,
                    function_proto.domain,
                    function_proto.name,
                )

            value_info = onnx.ValueInfoProto()
            value_info.name = actual_input_value_info.name
            value_info.type.CopyFrom(actual_input_value_info.type)
            graph_proto.input.append(value_info)

        for actual_output_value_info, formal_output in zip(
            actual_output_value_infos, function_proto.output
        ):
            if actual_output_value_info is None:
                logger.error(
                    "Value info for output %s is not found. Skip model proto creation for function %s::%s",
                    formal_output,
                    function_proto.domain,
                    function_proto.name,
                )
                return None
            if actual_output_value_info.type is None:
                logger.error(
                    "Value info for output %s has no type. Skip model proto creation for function %s::%s",
                    formal_output,
                    function_proto.domain,
                    function_proto.name,
                )

            value_info = onnx.ValueInfoProto()
            value_info.name = actual_output_value_info.name
            value_info.type.CopyFrom(actual_output_value_info.type)
            graph_proto.output.append(value_info)

        new_function_node = onnx.NodeProto()
        new_function_node.op_type = function_proto.name
        new_function_node.domain = function_proto.domain
        new_function_node.input.extend([input.name for input in actual_input_value_infos])  # type: ignore[union-attr]
        new_function_node.output.extend([output.name for output in actual_output_value_infos])  # type: ignore[union-attr]
        # TODO: Producing function node attribute is not supported yet.

        graph_proto.node.append(new_function_node)
        called_function_protos = self.find_all_called_function_protos(function_proto)
        for called_function_proto in called_function_protos:
            graph_proto.value_info.extend(
                self._generate_value_info_for_function_values(called_function_proto)
            )
        unit_model_proto.functions.extend(called_function_protos)
        return unit_model_proto

    def process_initializer(self, init: onnx.TensorProto):
        self.bind(
            init.name,
            ir.Value(name=init.name, type=utils.get_initializer_type(init)),
        )

    def lookup(self, name: str) -> ir.Value | None:
        """Override unit model proto inputs & outputs value infos with value info derived from actual example data.

        This step is required because onnx FunctionProto does not contain value info.
        The experimental solution from exporter writes value infos under root GraphProto, and associate them with
        FunctionProto by name mangling. This is lost during onnx.inliner because of the structural and value name
        changes.

        This step is not necessary once value info is natively supported in FunctionProto.

        This step by design cannot support dynamic shape.
        """
        if name in self._named_values:
            return ir.Value(
                name=name,
                type=onnx_helper.make_tensor_type_proto(
                    onnx_helper.np_dtype_to_tensor_dtype(self._named_values[name].dtype),
                    self._named_values[name].shape,
                ),
            )
        return super().lookup(name)

    def visit_model(self, model: onnx.ModelProto):
        functions_to_keep_visitor = FunctionToKeepVisitor(self.function_keyword)
        functions_to_keep_visitor.visit_model(model)
        functions_to_keep = functions_to_keep_visitor.functions_to_keep
        # TODO: bug report: IsScalar function inside if subgraph is not part of functions_to_keep.
        # Yet it is also not inlined. But its function_proto is removed by inliner.
        # To unblock us, we manually add it to functions_to_keep.
        functions_to_keep.append(("pkg.onnxscript.torch_lib.common", "IsScalar"))
        # TODO: Post ONNX 1.16, overload will be introduced.
        functions_to_keep = [function_id[:2] for function_id in functions_to_keep]
        inlined_model_proto = onnx.inliner.inline_selected_functions(
            model, functions_to_keep, exclude=True
        )
        target_function_meta_visitor = TargetFunctionMetaVisitor(self.function_keyword)
        target_function_meta_visitor.visit_model(inlined_model_proto)
        target_function_meta = target_function_meta_visitor.function_meta

        fetch_outputs = []  # type: ignore[var-annotated]
        for inputs, outputs in target_function_meta.values():
            fetch_outputs.extend((*inputs, *outputs))

        fetch_output_value_infos = []
        for fetch_output in fetch_outputs:
            value_info = onnx.ValueInfoProto()
            value_info.name = fetch_output
            fetch_output_value_infos.append(value_info)

        inlined_model_proto.graph.output.extend(fetch_output_value_infos)
        inlined_model_proto = onnx.shape_inference.infer_shapes(inlined_model_proto)

        self._model_proto = inlined_model_proto

        model_path = self.model_path
        model_dir = os.path.dirname(model_path)
        inputs, _ = evaluation_utils.load_test_data(  # type: ignore[assignment]
            model_dir, [i.name for i in model.graph.input]
        )
        tmp_model_path = f"{model_dir}/tmp_model.onnx"
        onnx.save(inlined_model_proto, tmp_model_path)

        sess = onnxruntime.InferenceSession(
            tmp_model_path, providers=["CUDAExecutionProvider"]
        )
        outputs = sess.run(fetch_outputs, inputs)
        assert (
            len(outputs) == len(fetch_outputs)
        ), f"Number of outputs mismatch. outputs: {len(outputs)}, fetch_outputs: {len(fetch_outputs)}"

        self._named_values = dict(zip(fetch_outputs, outputs))  # type: ignore[arg-type]
        for inputs, outputs in target_function_meta.values():
            named_inputs = [(i, self._named_values[i]) for i in inputs]
            named_outputs = [(o, self._named_values[o]) for o in outputs]
            self._unit_model_inputs.append(named_inputs)
            self._unit_model_outputs.append(named_outputs)

        for function in inlined_model_proto.functions:
            self._functions[ir.get_function_id(function)] = function

        super().visit_model(inlined_model_proto)

    def process_function(self, function: onnx.FunctionProto):
        if function.name.find(self.function_keyword) == -1:
            return

        try:
            actual_input_value_infos = [self.lookup(input) for input in function.input]
            actual_output_value_infos = [self.lookup(output) for output in function.output]
        except ValueError as e:
            raise ValueError(
                "Cannot create ModelProto unittest for function. "
                f"Failed to find value info for function {function.domain}::{function.name}"
            ) from e
        unit_model_proto = self.create_unit_model_proto(
            function, actual_input_value_infos, actual_output_value_infos
        )
        if unit_model_proto is not None:
            self._unit_model_protos.append(unit_model_proto)


def produce_function_proto_unittest(
    model_path: str,
    function_keyword: str,
    output_dir: str,
) -> tuple[
    list[onnx.ModelProto],
    list[list[tuple[str, np.ndarray]]],
    list[list[tuple[str, np.ndarray]]],
]:
    model_proto = onnx.load(model_path, load_external_data=False)

    # model_proto = optimizer.optimize(model_proto, onnx_shape_inference=False)

    producer = FunctionProtoProducerWithData(
        function_keyword,
        model_path,
        output_dir,
    )

    producer.visit_model(model_proto)
    return (
        producer.unit_model_protos,
        producer.unit_model_inputs,
        producer.unit_model_outputs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str)
    parser.add_argument("--function", type=str)
    parser.add_argument("--output-dir", "--output_dir", type=str)
    parser.add_argument("--max-outputs", "--max_outputs", type=int, default=sys.maxsize)
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    model_path = args.model_path
    function = args.function
    output_dir = args.output_dir
    max_outputs = args.max_outputs
    name = args.name

    (
        unit_model_protos,
        named_inputs_list,
        named_outputs_list,
    ) = produce_function_proto_unittest(model_path, function, output_dir)

    for i, unit_model_proto in enumerate(unit_model_protos[:max_outputs]):
        if logger.level <= logging.DEBUG:
            logger.debug("unit model proto %d:", i)
            # logger.debug(onnx.printer.to_text(unit_model_proto))
        output_model_dir = f"{output_dir}/{name}_{i}/"
        os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(unit_model_proto, f"{output_model_dir}/{name}_{i}.onnx")
        # save test data
        test_data_dir = f"{output_model_dir}/test_data_set_0/"
        os.makedirs(test_data_dir, exist_ok=True)
        named_inputs = named_inputs_list[i]
        for j, (_, input) in enumerate(named_inputs):
            save_tensor_data(input, f"{test_data_dir}/input_{j}.pb")
        named_outputs = named_outputs_list[i]
        for j, (_, output) in enumerate(named_outputs):
            save_tensor_data(output, f"{test_data_dir}/output_{j}.pb")

    print(
        f"{len(unit_model_protos[:max_outputs])} unit model protos and test data are saved to {output_dir}."
    )


if __name__ == "__main__":
    # python tools/function_rewriter_testing/function_unittest_producer.py \
    #    --model_path tools/ort_rewriter_profiling/onnx_models/stable_diffusion_unet/dynamo/stable_diffusion_unet_dynamo.onnx \
    #    --function GEGLU --output-dir testdata/unittest_models/ --max_outputs 4 --name geglu_stable_diffusion_unet
    main()
