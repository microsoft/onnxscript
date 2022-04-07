import dataclasses
import unittest
import numpy as np
import importlib
import onnx
from onnx import ModelProto, OperatorSetIdProto
from onnxscript import utils
from onnxruntime import InferenceSession
import onnx.backend.test.case.node as node_test
from typing import Any, Callable, Sequence


@dataclasses.dataclass(repr=False, eq=False)
class FunctionTestParams:
    function: Callable
    input: list
    output: list
    attrs: dict = None


class OnnxScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.default_opset_imports = [onnx.helper.make_opsetid("", 15)]
        self.local_opset_import = onnx.helper.make_opsetid("local", 1)
        self.local_function_domain = "local"
        self.rtol = 1e-7

    def _create_model_from_param(
            self,
            param: FunctionTestParams,
            opset_imports: Sequence[OperatorSetIdProto]
    ) -> ModelProto:
        opset_imports = opset_imports if opset_imports\
            else self.default_opset_imports
        local_function_proto = utils.convert_python_function_to_function_proto(
            param.function,
            self.local_function_domain,
            opset_imports)

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
            self.local_function_domain,
            opset_imports,
            self.local_opset_import,
            **(param.attrs or {}))

    def run_converter_test(
            self,
            param: FunctionTestParams,
            opset_import: OperatorSetIdProto = None):
        model = self._create_model_from_param(param, opset_import)
        input = {vi.name: t for vi, t in zip(model.graph.input, param.input)}
        sess = InferenceSession(
            model.SerializeToString(), providers=['CPUExecutionProvider'])
        actual = sess.run(None, input)
        np.testing.assert_equal(actual, param.output)

    def run_eager_test(
            self,
            param: FunctionTestParams,
            opset_imports: Sequence[OperatorSetIdProto] = None):
        # if opset_imports:
        #     module = importlib.import_module(param.function.__module__)
        #     if not module.op or\
        #         module.op.domain != opset_imports[0].domain or\
        #             module.op.version != opset_imports[0].version:
        #         # we want to run eager mode executor with the same domain
        #         # and version as requested.
        #         utils.assign_eager_mode_evaluator_to_module(
        #             module, opset_imports[0].domain, opset_imports[0].version)

        actual = param.function(*param.input, **(param.attrs or {}))
        np.testing.assert_allclose(
            actual if isinstance(actual, list)
            else [actual], param.output, rtol=self.rtol)

    def run_onnx_test(
            self,
            function: Callable,
            **attrs: Any):
        cases = node_test.collect_testcases_by_operator(function.__name__)
        for i, case in enumerate(cases):
            for ds in case.data_sets:
                param = FunctionTestParams(function, ds[0], ds[1], attrs=attrs)
                self.run_converter_test(param, case.model.opset_import)
                self.run_eager_test(param, case.model.opset_import)
