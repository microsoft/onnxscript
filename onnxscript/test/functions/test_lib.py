import dataclasses
import unittest
import numpy as np
import onnx
from onnx import TensorProto, ValueInfoProto
from onnx.backend.test.case.node import _extract_value_info
from onnxscript.converter import Converter
from onnxscript import eager_mode_evaluator
from onnxscript.irbuilder import Function
from onnxruntime import InferenceSession
import inspect
import onnx.backend.test.case.node as node_test
import importlib

@dataclasses.dataclass
class FunctionTestParams:
    function: Function
    input: list
    output: list
    attrs: dict = None

class OnnxScriptTestCase(unittest.TestCase):
    def _create_model(io):
        converter = Converter()
        module = importlib.import_module(io.function.__module__)
        setattr(module, 'oxs', eager_mode_evaluator)

        function_protos = converter.convert(inspect.getsource(module))
        function_protos = [x for x in function_protos if x.name == io.function.__name__]
        func_opset_imports = [onnx.helper.make_opsetid("", 15)]
        local_function_domain = "local"
        local_function_name = "local_function"
        local_function_proto = function_protos[0].to_function_proto(
            local_function_domain, local_function_name, func_opset_imports)

        inputs = ["input_" + str(i) for i in range(len(io.input))]
        outputs = ["output_" + str(i) for i in range(len(io.output))]

        input_value_infos = []
        for input, arr in zip(inputs, io.input):
            elem_type: TensorProto.DataType
            shape: tuple
            if isinstance(arr, np.ndarray):
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
                shape = arr.shape
            else:
                # FIXME(liqunfu): use schema to get the currect element type
                elem_type = TensorProto.FLOAT
                shape = (1,)

            value_info = onnx.helper.make_tensor_value_info(
                name=input,
                elem_type=elem_type,
                shape=shape)
            input_value_infos.append(value_info)
        
        output_value_infos = []
        for output, arr in zip(outputs, io.output):
            elem_type: TensorProto.DataType
            shape: tuple
            if isinstance(arr, np.ndarray):
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
                shape = arr.shape
            else:
                # FIXME(liqunfu): use schema to get the currect element type
                elem_type = TensorProto.FLOAT
                shape = (1,)

            value_info = onnx.helper.make_tensor_value_info(
                name=output,
                elem_type=elem_type,
                shape=shape)
            output_value_infos.append(value_info)

        node = None
        if io.attrs:
            node = onnx.helper.make_node(local_function_name, inputs, outputs, domain=local_function_domain, **io.attrs)
        else:
            node = onnx.helper.make_node(local_function_name, inputs, outputs, domain=local_function_domain)
        graph_temp = onnx.helper.make_graph([node], "node_graph", input_value_infos, output_value_infos)
        model_temp = onnx.helper.make_model(
            graph_temp,
            functions=[local_function_proto],
            producer_name='onnx-script',
            opset_imports=[onnx.helper.make_opsetid("", 15), onnx.helper.make_opsetid("local", 1)])
        # model = onnx.shape_inference.infer_shapes(model_temp, check_type=True, strict_mode=True)
        # onnx.checker.check_model(model)
        return model_temp

    def _run_converter_test(self, io):
        model = OnnxScriptTestCase._create_model(io)
        input = {vi.name: t for vi, t in zip(model.graph.input, io.input)}
        while len(model.graph.input) > 0:
            model.graph.input.remove(model.graph.input[0])
        for name, value, in input.items():
            vi = _extract_value_info(value, name)
            model.graph.input.append(vi)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

        # m = onnx.load("C:/Temp/FunctionVerification.TestModelLocalFunctions.before.onnx")
        # sess = InferenceSession(m.SerializeToString())

        sess = InferenceSession(model.SerializeToString())
        actual = sess.run(None, input)
        np.testing.assert_equal(actual, io.output)
    
    def _run_eager_test(self, io):
        module = importlib.import_module(io.function.__module__)
        setattr(module, 'oxs', eager_mode_evaluator)
        actual = []
        if io.attrs:
            actual = io.function(*io.input, **io.attrs)
        else:
            actual = io.function(*io.input)
        np.testing.assert_allclose(actual if isinstance(actual, list) else [actual], io.output, rtol=1e-05)

def generate_test(io):
    def test_convertor(self):
        self._run_converter_test(io)        

    def test_eager(self):
        self._run_eager_test(io)        

    return test_convertor, test_eager

def run_onnx_test(function, list_attrs):
    io = []
    cases = node_test.collect_testcases_by_operator(function.__name__)
    for i, case in enumerate(cases):
        if not list_attrs:
            io.extend([FunctionTestParams(function, ds[0], ds[1]) for ds in case.data_sets])
        else:
            io.extend([FunctionTestParams(function, ds[0], ds[1], attrs=list_attrs[i]) for ds in case.data_sets])

    test = OnnxScriptTestCase()
    for io_ in io:
        test._run_converter_test(io_)
        test._run_eager_test(io_)
