# SPDX-License-Identifier: Apache-2.0

import functools
import numbers
import numpy as np
import typing

import onnx
from onnx import numpy_helper, AttributeProto, TypeProto
from onnxruntime import InferenceSession

from .utils import convert_arrays_to_value_infos
from .values import Opset


class EagerModeEvaluator(Opset):
    def __init__(self, domain, version):
        super().__init__(domain, version)

    def convert_to_tensor(v, k):
        if isinstance(v, np.ndarray):
            return numpy_helper.from_array(v)
        elif isinstance(v, list):
            return numpy_helper.from_array(np.array(v))
        elif isinstance(v, numbers.Number):
            return numpy_helper.from_array(np.array([v]))
        elif isinstance(v, onnx.TensorProto):
            return v
        else:
            raise ValueError("attribute {attribute_name} must be convertable to TensorProto, got {type}", k, type(v)) # noqa E501

    def convert_attributes_to_tensors_with_schema(
            attribute_dict, schema_attribute_dict):
        # Constant and ConstantLike are the 2 ops in onnx
        # that take a tensor as attribute value.
        # onnx-script tends to use a literal number for attribute.
        # This methods is to make this scenario work.
        for k, v in attribute_dict.items():
            attribute_type = schema_attribute_dict[k].type
            if attribute_type == AttributeProto.TENSOR:
                attribute_dict[k] = EagerModeEvaluator.convert_to_tensor(v, k)

    def call(self, opname, *args, **kwargs):
        schema = self[opname]
        EagerModeEvaluator.convert_attributes_to_tensors_with_schema(
            kwargs, schema.attributes)

        num_inputs = len(args)
        num_outputs = len(schema.outputs)
        inputs = ["input" + str(i) for i in range(num_inputs)]
        outputs = ["output" + str(i) for i in range(num_outputs)]
        node = onnx.helper.make_node(opname, inputs, outputs, **kwargs)
        input_value_infos = convert_arrays_to_value_infos(inputs, list(args))

        output_value_infos = [
            onnx.helper.make_value_info(name, TypeProto()) for name in outputs]
        graph_temp = onnx.helper.make_graph(
            [node], "node_graph", input_value_infos, output_value_infos)
        opset_id = onnx.helper.make_opsetid(self.domain, self.version)
        model = onnx.helper.make_model(
            graph_temp, opset_imports=[opset_id])
        sess = InferenceSession(
            model.SerializeToString(), providers=['CPUExecutionProvider'])

        session_run_input = {
            input: arg if isinstance(arg, np.ndarray) else [arg]
            for input, arg in zip(inputs, args)}

        got = sess.run(None, session_run_input)
        return got[0] if len(got) == 1 else got

    def __getattr__(self, attr: str) -> typing.Any:
        return globals().get(
            attr,
            functools.partial(
                self.call,
                attr))
