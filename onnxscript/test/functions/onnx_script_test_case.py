# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import dataclasses
import unittest
from typing import Any, List
import numpy as np
import onnx
from onnx import ModelProto
import onnx.backend.test.case.node as node_test
from onnxscript import utils
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument
from onnxscript.main import OnnxFunction


@dataclasses.dataclass(repr=False, eq=False)
class FunctionTestParams:
    function: OnnxFunction
    input: list
    output: list
    attrs: dict = None


class OnnxScriptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A function (and node) in a model tells its domain, not version.
        # When building a model that consumes functions and nodes, model opset_imports
        # indicate domains and versions of nodes and functions that are used.
        # Function version number is needed for a runtime to run it without inlining.
        # Before ONNX IR (or FunctionIR) being updated
        # for FunctionProto to have version number we
        # need to put a default version number here to workaround the problem.
        cls.local_function_opset_version = 1
        cls.atol = 1e-7
        cls.rtol = 1e-7
        try:
            # experimental version
            cls.all_test_cases = node_test.collect_testcases()
        except TypeError:
            # official version
            cls.all_test_cases = node_test.collect_testcases(None)

    def _create_model_from_param(
            self,
            param: FunctionTestParams
    ) -> ModelProto:
        ir = param.function.function_ir
        local_function_proto = ir.to_function_proto("")

        input_names = ["input_" + str(i) for i in range(len(param.input))]
        output_names = ["output_" + str(i) for i in range(len(param.output))]
        input_value_infos = utils.convert_arrays_to_value_infos(
            input_names, param.input)
        output_value_infos = utils.convert_arrays_to_value_infos(
            output_names, param.output)

        return utils.make_model_from_function_proto(
            local_function_proto,
            self.local_function_opset_version,
            input_value_infos,
            output_value_infos,
            **(param.attrs or {}))

    def _filter_test_case_by_op_type(self, op_type):
        test_cases = [
            case for case in self.all_test_cases
            if case.kind == 'node' and len(case.model.graph.node) == 1
            and case.model.graph.node[0].op_type == op_type]
        return test_cases

    def run_converter_test(self, param: FunctionTestParams):
        # we need the latest version in onnx.ai domain
        # to build a function
        model = self._create_model_from_param(param)
        onnx.checker.check_model(model)
        input = {
            vi.name: t
            for vi, t in zip(model.graph.input, param.input)}
        try:
            sess = InferenceSession(
                model.SerializeToString(), providers=['CPUExecutionProvider'])
        except (Fail, InvalidArgument) as e:
            raise AssertionError(
                "Unable to load model\n%s" % str(model)) from e
        actual = sess.run(None, input)
        np.testing.assert_allclose(actual, param.output, rtol=self.rtol)

    def run_eager_test(
            self,
            param: FunctionTestParams,
            rtol: float = None,
            atol: float = None):

        actual = param.function(*param.input, **(param.attrs or {}))
        np.testing.assert_allclose(
            actual if isinstance(actual, list)
            else [actual], param.output,
            rtol=rtol or self.rtol, atol=atol or self.atol)

    def run_onnx_test(
            self,
            function: OnnxFunction,
            rtol: float = None,
            atol: float = None,
            skip_eager_test: bool = False,
            skip_test_names: List[str] = [],
            **attrs: Any) -> None:
        '''
        Run ONNX test cases with an OnnxFunction.
        The function shall have test cases in ONNX repo.
        For example: in onnx/test/case/node.
        Test case models and data are used to do converter and eager mode test.

        Arguments:
            function (OnnxFunction): the function to be tested.
            skip_eager_test (bool): not to run eager test if Ture.
            skip_test_names (List[str]): to skip these tests.
            attrs (Any): default attributes of the function node.

        '''

        cases = self._filter_test_case_by_op_type(function.function_ir.name)
        for i, case in enumerate(cases):
            if len(case.model.graph.node) != 1:
                raise ValueError("run_onnx_test only \
                    tests models with one operator node.")

            if case.name not in skip_test_names:
                test_case_attrs = {
                    a.name: onnx.helper.get_attribute_value(a)
                    for a in case.model.graph.node[0].attribute}
                test_case_attrs = {**attrs, **test_case_attrs}

                for ds in case.data_sets:
                    param = FunctionTestParams(
                        function, ds[0], ds[1], attrs=test_case_attrs)
                    self.run_converter_test(param)
                    if not skip_eager_test:
                        self.run_eager_test(param, rtol=rtol, atol=atol)
