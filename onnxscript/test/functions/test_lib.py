import dataclasses
import unittest
import numpy as np
import onnx
from onnx.backend.test.case.node import _extract_value_info
from onnxscript.converter import Converter
from onnxscript import eager_mode_evaluator
from onnxruntime import InferenceSession
import inspect
import functools
import collections
import importlib

@dataclasses.dataclass
class Params:
    input: list
    output: list
    attrs: dict = None

class OnnxScriptTestCase(unittest.TestCase):
    function = None
    @classmethod
    def setUpClass(cls):
        converter = Converter()
        module = importlib.import_module(cls.function.__module__)
        setattr(module, 'oxs', eager_mode_evaluator)

        function_protos = converter.convert(inspect.getsource(module))
        function_protos = [x for x in function_protos if x.name == cls.function.__name__]        
        graph = function_protos[0].to_graph_proto()
        cls.model = onnx.helper.make_model(
            graph, producer_name='onnx-script',
            opset_imports=[onnx.helper.make_opsetid("", 15)])

    def _run_converter_test(self, io):
        input = {vi.name: t for vi, t in zip(self.model.graph.input, io.input)}
        self.model.graph.input.clear()
        [self.model.graph.add_input(_extract_value_info(value, name, None))
         for name, value, in input.items()]
        model = onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(model)
        sess = InferenceSession(model.SerializeToString())
        actual = sess.run(None, input)
        self.assertEqual(actual, io.output)
    
    def _run_eager_test(self, io):
        # setattr(self.function, 'oxs', eager_mode_evaluator)
        actual = self.__class__.function(*io.input)
        self.assertEqual(actual, io.output)

def generate_test(io):
    def test_convertor(self):
        self._run_converter_test(io)        

    def test_eager(self):
        self._run_eager_test(io)        

    return test_convertor, test_eager

def create_test_case(function, cases):
    test_case_dict = {"function": function}
    for i, io in enumerate(cases):
        test_name = 'test_%d' % i
        test = generate_test(io)
        test_case_dict[test_name + "_converter"] = test[0]
        test_case_dict[test_name + "_eager"] = test[1]

    return type('%sTestCase' % function.__name__, (OnnxScriptTestCase, ), test_case_dict)
