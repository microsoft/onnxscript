# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx.parser
import onnx.shape_inference

from onnxscript import ir
from onnxscript.rewriter import collapse_slices, testing

_INT64_MAX = 9223372036854775807


class TwoReshapesMatMulReshapeTest(unittest.TestCase):
    def test_slice_is_redundant_when_ends_is_greater_than_input_shape(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112] data) => (float[512, 16, 112] output)
            {
                starts = Constant<value: tensor = int64[1] {0}>()
                ends = Constant<value: tensor = int64[1] {9999}>()
                axes = Constant<value: tensor = int64[1] {0}>()
                steps = Constant<value: tensor = int64[1] {1}>()
                output = Slice (data, starts, ends, axes, steps)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)
        self.assertIn("Identity", [node.op_type for node in model.graph])
        testing.assert_numerically_equal(
            model_proto,
            model,
            (np.random.rand(512, 16, 112).astype(np.float32),),
        )

    def test_slice_is_redundant_when_ends_reaches_int64_max(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112] data) => (float[512, 16, 112] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{_INT64_MAX}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)
        self.assertIn("Identity", [node.op_type for node in model.graph])
        testing.assert_numerically_equal(
            model_proto,
            model,
            (np.random.rand(512, 16, 112).astype(np.float32),),
        )

    def test_slice_pattern_is_not_matched_when_input_is_dynamic(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[L, M, N] data) => (float[L, M, N] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{9}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_scatternd_is_redundant_when_it_is_updating_the_whole_input_in_order(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[112, 16, 512] data, float[112, 16, 512] updates) => (float[112, 16, 512] output)
            {
                output = ScatterND (data, indices, updates)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        indices = np.arange(112).reshape(112, 1).astype(np.int64)
        model = ir.serde.deserialize_model(model_proto)
        # from numpy to ir.Tensor
        indices_ir_tensor = ir.Tensor(
            name="indices",
            value=indices,
        )
        # assign the tensor to a value
        indices = model.graph[0].inputs[1]
        indices.const_value = indices_ir_tensor
        model.graph.initializers["indices"] = indices
        original_model_proto = ir.serde.serialize_model(model)

        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertIn("Identity", [node.op_type for node in model.graph])

        input = np.random.rand(112, 16, 512).astype(np.float32)
        testing.assert_numerically_equal(original_model_proto, model, (input, input))

    def test_check_if_redundant_scatternd_logs_when_indices_not_referring_to_whole_data(self):
        data = ir.Value(shape=ir.Shape([112, 16, 512]))
        updates = ir.Value(shape=ir.Shape([112, 16, 512]))
        indices = ir.Value(const_value=ir.Tensor(np.array([[0], [1], [2]])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_scatternd(None, data, indices, updates)
            self.assertFalse(result)
            self.assertIn("The 'indices' is not referring to the whole data.", log.output[0])


    def test_check_if_redundant_scatternd_logs_when_updates_shape_not_statically_known(self):
        data = ir.Value(shape=ir.Shape([112, 16, 512]))
        updates = ir.Value(shape=None)
        indices = ir.Value(const_value=ir.Tensor(np.arange(112).reshape(112, 1).astype(np.int64)))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_scatternd(None, data, indices, updates)
            self.assertFalse(result)
            self.assertIn("The value 'updates' shape is not statically known.", log.output[0])


    def test_check_if_redundant_scatternd_logs_when_data_shape_not_statically_known(self):
        data = ir.Value(shape=None)
        updates = ir.Value(shape=ir.Shape([112, 16, 512]))
        indices = ir.Value(const_value=ir.Tensor(np.arange(112).reshape(112, 1).astype(np.int64)))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_scatternd(None, data, indices, updates)
            self.assertFalse(result)
            self.assertIn("The value 'data' shape is not statically known.", log.output[0])


    def test_check_if_redundant_slice_logs_when_start_not_zero(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([1])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'start' is not 0.", log.output[0])


    def test_check_if_redundant_slice_logs_when_step_not_one(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([2])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'step' is not 1.", log.output[0])


    def test_check_if_redundant_slice_logs_when_step_not_scalar(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1, 2])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'step' is not a scalar.", log.output[0])


    def test_check_if_redundant_slice_logs_when_axis_not_scalar(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0, 1])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'axis' is not a scalar.", log.output[0])


    def test_check_if_redundant_slice_logs_when_end_not_scalar(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999, 10000])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'end' is not a scalar.", log.output[0])


    def test_check_if_redundant_scatternd_logs_when_shapes_different(self):
        data = ir.Value(shape=ir.Shape([112, 16, 512]))
        updates = ir.Value(shape=ir.Shape([112, 16, 256]))
        indices = ir.Value(const_value=ir.Tensor(np.arange(112).reshape(112, 1).astype(np.int64)))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_scatternd(None, data, indices, updates)
            self.assertFalse(result)
            self.assertIn("The shape of 'data' and 'updates' are different.", log.output[0])


    def test_check_if_redundant_slice_logs_when_end_less_than_shape(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0])))
        ends = ir.Value(const_value=ir.Tensor(np.array([100])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'end' is less than the shape of the specified axis.", log.output[0])


    def test_check_if_redundant_slice_logs_when_start_not_scalar(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value(const_value=ir.Tensor(np.array([0, 1])))
        ends = ir.Value(const_value=ir.Tensor(np.array([9999])))
        axes = ir.Value(const_value=ir.Tensor(np.array([0])))
        steps = ir.Value(const_value=ir.Tensor(np.array([1])))
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'start' is not a scalar.", log.output[0])


    def test_check_if_redundant_slice_logs_when_values_not_statically_known(self):
        data = ir.Value(shape=ir.Shape([512, 16, 112]))
        starts = ir.Value()
        ends = ir.Value()
        axes = ir.Value()
        steps = ir.Value()
        
        with self.assertLogs(collapse_slices.logger, level='INFO') as log:
            result = collapse_slices._check_if_redundant_slice(None, data, starts, ends, axes, steps)
            self.assertFalse(result)
            self.assertIn("The value 'start', 'end', 'axis', 'step' is not statically known.", log.output[0])

