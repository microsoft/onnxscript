# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import dataclasses
import numbers
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

    def _map_op_input_to_model(self, onnx_case_model: ModelProto):
        # op("x", "", "max") model("x", "max") => map_op_input_to_model[0, -1, 1]
        op_input = onnx_case_model.graph.node[0].input
        model_input = [input.name for input in onnx_case_model.graph.input]
        return [-1 if input == "" else model_input.index(input) for input in op_input]

    def _create_model_from_param(
            self,
            param: FunctionTestParams,
            onnx_case_model: ModelProto
    ) -> ModelProto:
        ir = param.function.function_ir
        local_function_proto = ir.to_function_proto("")

        if onnx_case_model:
            # we want to create a model that onnx_test_runner can run with onnx test case data
            input_names = [i.name for i in onnx_case_model.graph.input]
            output_names = [o.name for o in onnx_case_model.graph.output]
        else:
            input_names = ["input_" + str(i) for i in range(len(param.input))]
            output_names = ["output_" + str(i) for i in range(len(param.output))]

        # models from onnx test case do not have optional input if test data
        # are not provided for the input.
        # models from script keep all optional inputs, notwithstanding test data availability.
        # to run script model with onnx test data, we need to map input test data
        # to the corresponding script model input.
        if onnx_case_model:
            input_index_map_op_to_model = self._map_op_input_to_model(onnx_case_model)

            test_case_input_value_infos = utils.convert_arrays_to_value_infos(input_names, param.input)
            script_model_input_type_infos = [input.typeinfo.to_type_proto() if input.typeinfo else None for input in param.function.function_ir.inputs]
            input_value_infos = [None] * len(script_model_input_type_infos)

            for i in range(len(script_model_input_type_infos)):

                if input_index_map_op_to_model and i < len(input_index_map_op_to_model):
                    index_test_case_input = input_index_map_op_to_model[i]
                else:
                    index_test_case_input = -1
                if index_test_case_input >= 0:
                    if script_model_input_type_infos[i]:
                        input_value_infos[i] = onnx.helper.make_value_info(
                            test_case_input_value_infos[index_test_case_input].name, script_model_input_type_infos[i])
                    else:
                        input_value_infos[i] = test_case_input_value_infos[index_test_case_input]
                else:
                    # test data is missing for an optional input
                    assert script_model_input_type_infos[i], "type info is not provided for optional input"
                    input_value_infos[i] = onnx.helper.make_value_info(
                        "input_" + str(i), script_model_input_type_infos[i])
        else:
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

    def run_converter_test(
        self,
        param: FunctionTestParams,
        onnx_case_model: ModelProto=None):
        # we need the latest version in onnx.ai domain
        # to build a function
        model = self._create_model_from_param(param, onnx_case_model)
        onnx.checker.check_model(model)
        if onnx_case_model:
            input = {
                vi.name: np.array(t) if isinstance(t, numbers.Number) else t
                for vi, t in zip(onnx_case_model.graph.input, param.input)}
        else:
            input = {
                vi.name: np.array(t) if isinstance(t, numbers.Number) else t
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
            atol: float = None,
            onnx_case_model: ModelProto=None):
        input_index_map_op_to_model = None
        if onnx_case_model:
            input_index_map_op_to_model = self._map_op_input_to_model(onnx_case_model)
        if input_index_map_op_to_model and any([idx == -1 for idx in input_index_map_op_to_model]):
            # there are missing optional input.
            # shall call script function with named args for optional inputs
            function_ir_input = param.function.function_ir.inputs
            first_optional_input_index = -1
            optional_input_dict = {}
            for i, input in enumerate(function_ir_input):
                if input.typeinfo and input.typeinfo.optional:
                    if input_index_map_op_to_model[i] != -1:
                        optional_input_dict[input.name] = param.input[input_index_map_op_to_model[i]]
                    if first_optional_input_index == -1:
                        first_optional_input_index = i

            actual = param.function(
                *param.input[:first_optional_input_index],
                **{**optional_input_dict, **param.attrs})
        else:
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
                print(case.name)
                if case.name == "test_clip_default_max":
                    print(case.name)
                test_case_attrs = {
                    a.name: onnx.helper.get_attribute_value(a)
                    for a in case.model.graph.node[0].attribute}
                test_case_attrs = {**attrs, **test_case_attrs}

                for ds in case.data_sets:
                    param = FunctionTestParams(
                        function, ds[0], ds[1], attrs=test_case_attrs)
                    self.run_converter_test(param, case.model)
                    if not skip_eager_test:
                        self.run_eager_test(param, rtol=rtol, atol=atol, onnx_case_model=case.model)
