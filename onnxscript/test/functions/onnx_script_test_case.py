import dataclasses
import unittest
import numpy as np
import onnx
from onnx.backend.test.case.node import _extract_value_info
from onnxscript.converter import Converter
from onnxscript import utils
    
from onnxruntime import InferenceSession
import onnx.backend.test.case.node as node_test
from typing import Callable


@dataclasses.dataclass(repr=False, eq=False)
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
        local_function_proto = utils.convert_python_function_to_function_proto(
            param.function,
            OnnxScriptTestCase.local_function_domain,
            [OnnxScriptTestCase.onnx_opset_import])

        input_names = ["input_" + str(i) for i in range(len(param.input))]
        output_names = ["output_" + str(i) for i in range(len(param.output))]
        input_value_infos = utils.convert_arrays_to_value_infos(
            input_names, param.input)
        output_value_infos = utils.convert_arrays_to_value_infos(
            output_names, param.output)

        return utils.make_model_from_function_proto(
            local_function_proto, 
            input_value_infos,
            output_value_infos,
            OnnxScriptTestCase.local_function_domain,
            OnnxScriptTestCase.onnx_opset_import,
            OnnxScriptTestCase.local_opset_import,
            **(param.attrs or {}))

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
        actual = param.function(*param.input, **(param.attrs or {}))
        np.testing.assert_allclose(
            actual if isinstance(actual, list)
            else [actual], param.output, rtol=1e-05)

    def run_onnx_test(self, function, **attrs):
        params = self.get_onnx_test_cases(function, **attrs)

        for param in params:
            self.run_converter_test(param)
            self.run_eager_test(param)

    def get_onnx_test_cases(self, function, **attrs):
        params = []
        cases = node_test.collect_testcases_by_operator(function.__name__)
        for i, case in enumerate(cases):
            params.extend([
                FunctionTestParams(function, ds[0], ds[1], attrs=attrs) # noqa E501
                for ds in case.data_sets])

        return params
