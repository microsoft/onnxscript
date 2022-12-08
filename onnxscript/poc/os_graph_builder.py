from onnxscript import script
import numpy as np
import onnxruntime
from onnx import helper, checker
from onnx import TensorProto
import onnx

from onnxscript.function_libs.torch_aten import ops as aten_ops

# ATEN data type mapping to ONNX Script
data_type_mapping = {
    # Aten DataType -> Onnx Op DateType
    "Float": TensorProto.FLOAT,
    "Long": TensorProto.INT64,
}


def get_dtype(dt):
    return data_type_mapping[dt]


# Demo. when the mapping is properly built we will not have script() everywhere
# The function naming and namespace will also be simplified
op_type_mapping = {
    # aten op -> onnx-script function
    "aten::add": script()(aten_ops.core.aten_add),
    "aten::sub": script()(aten_ops.core.aten_sub),
    "aten::mul": script()(aten_ops.core.aten_mul),
}


def get_op_type(op):
    return op_type_mapping[op]


def get_unique_variable_name(var):
    return "v" + var[1:]


class GraphBuilder:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.functions = {}
        self.onnx_model = None

    def add_input(self, info):
        value_info = self._make_tensor_value(info)
        self.inputs.append(value_info)

    def add_output(self, info):
        value_info = self._make_tensor_value(info)
        self.outputs.append(value_info)

    def add_node(self, info):
        node, func = self._make_node(info)
        self.nodes.append(node)
        if func is not None:
            self.functions[func.name] = func

    def make_model(self, model_name):
        graph = helper.make_graph(self.nodes, model_name, self.inputs, self.outputs)

        kwargs = {
            "opset_imports": [
                helper.make_opsetid("this", 1),
                helper.make_opsetid("", 17),
            ]
        }

        func_proto = []
        for func in self.functions.values():
            func_proto.append(func.to_function_proto())

        self.onnx_model = helper.make_model(graph, functions=func_proto, **kwargs)
        checker.check_model(self.onnx_model)
        return self.onnx_model

    def _make_tensor_value(self, info):
        (raw_var, raw_op, raw_dt, shape) = info
        var_name = get_unique_variable_name(raw_var)
        dtype = get_dtype(raw_dt)
        assert raw_op == "Value"
        value_info = helper.make_tensor_value_info(var_name, dtype, shape)
        return value_info

    def _make_node(self, info):
        (raw_var, raw_op, raw_dt, shape, raw_params) = info
        var_name = get_unique_variable_name(raw_var)
        if raw_op == "Constant":
            node = helper.make_node(
                op_type=raw_op,
                inputs=[],
                outputs=[var_name],
                value=helper.make_tensor(
                    name=var_name, data_type=get_dtype(raw_dt), dims=shape, vals=[raw_params]
                ),
            )
            return node, None
        else:
            os_func = get_op_type(raw_op)
            inputs = []
            for input in raw_params:
                inputs.append(get_unique_variable_name(input))
            node = helper.make_node(
                op_type=os_func.name,
                inputs=inputs,
                outputs=[var_name],
                name=os_func.name + "_" + var_name,
                domain="this",
            )
            return node, os_func


if __name__ == "__main__":
    demo_data = {
        "input": [("%0", "Value", "Float", [2, 3, 4]), ("%1", "Value", "Float", [2, 3, 4])],
        "graph": [
            ("%3", "aten::add", "Float", [2, 3, 4], ["%0", "%1"]),
            ("%4", "Constant", "Float", [], 5),
            ("%5", "aten::add", "Float", [2, 3, 4], ["%3", "%4"]),
            ("%6", "aten::sub", "Float", [2, 3, 4], ["%0", "%1"]),
            ("%7", "aten::mul", "Float", [2, 3, 4], ["%5", "%6"]),
        ],
        "output": [("%7", "Value", "Float", [2, 3, 4])],
    }

    gb: GraphBuilder = GraphBuilder()

    for key, value in demo_data.items():
        if key == "input":
            for info in value:
                gb.add_input(info)
        elif key == "output":
            for info in value:
                gb.add_output(info)
        elif key == "graph":
            for info in value:
                gb.add_node(info)
        else:
            raise Exception("wrong key=" + key)

    model_name = "os_test_model.onnx"
    onnx_model = gb.make_model(model_name)
    onnx.save(onnx_model, model_name)
    print("onnx model saved !!!")

    sess = onnxruntime.InferenceSession(model_name)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)

    result = sess.run(None, {"v0": x1, "v1": x2})
    print(result)
