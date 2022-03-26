import dataclasses
import unittest
import numpy as np
import onnx
from onnx.backend.test.case.node import _extract_value_info
from onnxscript.converter import Converter
from onnxscript.utils import convert_arrays_to_value_infos
from onnxruntime import InferenceSession
import inspect
import onnx.backend.test.case.node as node_test
import importlib
from typing import Callable


@dataclasses.dataclass
class FunctionTestParams:
    function: Callable
    input: list
    output: list
    attrs: dict = None


class OnnxScriptTestCase(unittest.TestCase):
    onnx_opset_import = onnx.helper.make_opsetid("", 15)
    local_opset_import = onnx.helper.make_opsetid("local", 1)
    local_function_domain = "local"
    local_function_name = "local_function"

    def _create_model_from_param(param):
        converter = Converter()
        module = importlib.import_module(param.function.__module__)

        ir_functions = converter.convert(inspect.getsource(module))
        ir_functions = [
            x for x in ir_functions if x.name == param.function.__name__]
        if len(ir_functions) != 1:
            raise ValueError(f"Cannot find signle function of '{param.function.__name__}' from module '{module.__name__}.py'") # noqa E501

        local_function_proto = ir_functions[0].to_function_proto(
            OnnxScriptTestCase.local_function_domain,
            [OnnxScriptTestCase.onnx_opset_import])

        inputs = ["input_" + str(i) for i in range(len(param.input))]
        outputs = ["output_" + str(i) for i in range(len(param.output))]
        input_value_infos = convert_arrays_to_value_infos(inputs, param.input)
        output_value_infos = convert_arrays_to_value_infos(
            outputs, param.output)

        node = None
        if param.attrs:
            node = onnx.helper.make_node(
                local_function_proto.name, inputs, outputs,
                domain=OnnxScriptTestCase.local_function_domain,
                **param.attrs)
        else:
            node = onnx.helper.make_node(
                local_function_proto.name, inputs, outputs,
                domain=OnnxScriptTestCase.local_function_domain)
        graph = onnx.helper.make_graph(
            [node], "node_graph",
            input_value_infos, output_value_infos)
        model = onnx.helper.make_model(
            graph,
            functions=[local_function_proto],
            producer_name='onnx-script',
            opset_imports=[
                OnnxScriptTestCase.onnx_opset_import,
                OnnxScriptTestCase.local_opset_import])
        return model

    def run_converter_test(self, param):
        model = OnnxScriptTestCase._create_model_from_param(param)
        input = {vi.name: t for vi, t in zip(model.graph.input, param.input)}
        while len(model.graph.input) > 0:
            model.graph.input.remove(model.graph.input[0])
        for name, value, in input.items():
            vi = _extract_value_info(value, name)
            model.graph.input.append(vi)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        sess = InferenceSession(model.SerializeToString())
        actual = sess.run(None, input)
        np.testing.assert_equal(actual, param.output)

    def run_eager_test(self, param):
        actual = []
        if param.attrs:
            actual = param.function(*param.input, **param.attrs)
        else:
            actual = param.function(*param.input)
        np.testing.assert_allclose(
            actual if isinstance(actual, list)
            else [actual], param.output, rtol=1e-05)

    def run_onnx_test(self, function, list_attrs):
        params = self.get_onnx_test_cases(function, list_attrs)

        for param in params:
            self.run_converter_test(param)
            self.run_eager_test(param)

    def get_onnx_test_cases(self, function, list_attrs):
        params = []
        cases = node_test.collect_testcases_by_operator(function.__name__)
        for i, case in enumerate(cases):
            if not list_attrs:
                params.extend([
                    FunctionTestParams(function, ds[0], ds[1]) for ds in case.data_sets]) # noqa E501
            else:
                params.extend([
                    FunctionTestParams(function, ds[0], ds[1], attrs=list_attrs[i]) # noqa E501
                    for ds in case.data_sets])

        return params
