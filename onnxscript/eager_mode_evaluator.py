import numpy as np
import onnx
from onnxruntime import InferenceSession
from onnx import ValueInfoProto, TensorProto
from onnx import numpy_helper
from onnx import AttributeProto
import onnx.shape_inference
from onnxscript.utils import convert_data_to_value_infos
import numbers
import functools
import typing

version = 15
# domain = "onnx.ai"
domain = ""


def convert_to_tensor(v):
    if isinstance(v, np.ndarray):
        return numpy_helper.from_array(v)
    elif isinstance(v, list):
        return numpy_helper.from_array(np.array(v))
    elif isinstance(v, numbers.Number):
        return numpy_helper.from_array(np.array([v]))
    elif not isinstance(v, onnx.TensorProto):
        raise ValueError("attribute {attribute_name} must be convertable to TensorProto, got {type}", k, type(v))


def convert_attributes_to_tensors_with_schema(attribute_dict, schema_attribute_dict):
    for k, v in attribute_dict.items():
        attribute_type = schema_attribute_dict[k].type
        if attribute_type == AttributeProto.TENSOR:
            attribute_dict[k] = convert_to_tensor(v)


def call(opname, domain, version, *args, **vargs):
    schema = onnx.defs.get_schema(opname, version, domain)
    convert_attributes_to_tensors_with_schema(vargs, schema.attributes)

    num_inputs = len(args)
    num_outputs = len(schema.outputs)
    inputs = ["input" + str(i) for i in range(num_inputs)]
    outputs = ["output" + str(i) for i in range(num_outputs)]
    node = onnx.helper.make_node(opname, inputs, outputs, **vargs)
    input_value_infos = convert_data_to_value_infos(inputs, list(args))

    def make_value_info(name):
        vi = ValueInfoProto()
        vi.name = name
        return vi

    output_value_infos = [make_value_info(name) for name in outputs]
    graph_temp = onnx.helper.make_graph([node], "node_graph", input_value_infos, output_value_infos)
    model_temp = onnx.helper.make_model(graph_temp)
    model = onnx.shape_inference.infer_shapes(model_temp, check_type=True, strict_mode=True)
    sess = InferenceSession(model.SerializeToString())

    session_run_input = {}
    for input, arg in zip(inputs, args):
        if isinstance(arg, np.ndarray):
            session_run_input[input] = arg
        else:
            session_run_input[input] = np.array([arg], dtype=np.float32)
    
    got = sess.run(None, session_run_input)
    return got[0] if len(got) == 1 else got


def __getattr__(attr: str) -> typing.Any:
    return globals().get(attr, functools.partial(call, attr, domain, version))
