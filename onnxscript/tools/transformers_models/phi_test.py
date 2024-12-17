# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=not-callable

import copy
import sys
import unittest

import numpy as np
import onnxruntime
import torch

import onnxscript.optimizer
import onnxscript.rewriter
import onnxscript.tools.training_helper
import onnxscript.tools.transformers_models
import onnxscript.tools.transformers_models.phi
from onnxscript._internal.version_utils import (
    has_transformers,
    ignore_warnings,
    torch_older_than,
)


class TestExportPhi(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.6"), reason="fails to export")
    @ignore_warnings(UserWarning)
    def test_phi_export_cpu(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.6"), reason="fails to export")
    @ignore_warnings(UserWarning)
    def test_phi_export_cpu_export_api(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(
            model, *input_tensors, export_api=True
        )
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @ignore_warnings(UserWarning)
    def test_phi_export_cuda(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors_cpu = input_tensors_many[0]
        model = model.to("cuda")
        input_tensors = [i.to("cuda") for i in input_tensors_cpu]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.detach().cpu().numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @ignore_warnings(UserWarning)
    def test_phi_dort_static(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)

        local_aot_ort = onnxscript.tools.training_helper.make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        results = compiled_model(*input_tensors)
        torch.testing.assert_close(expected[0], results[0], atol=1e-5, rtol=1e-5)

        expected_gradients = onnxscript.tools.training_helper.train_loop(model, *input_tensors)
        gradients = onnxscript.tools.training_helper.train_loop(compiled_model, *input_tensors)
        torch.testing.assert_close(expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5)


    def test_get_phi_model_from_config_unexpected_config(self):
        with self.assertRaises(ValueError) as context:
            onnxscript.tools.transformers_models.phi.get_phi_model_from_config(config="unexpected")
        self.assertIn("Unexpected configuration", str(context.exception))


    def test_get_phi_model_no_mask(self):
        model, input_tensors_many, dynamic_shapes = onnxscript.tools.transformers_models.phi.get_phi_model(with_mask=False)
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        self.assertIsNotNone(expected)
        self.assertEqual(len(input_tensors), 1)


    def test_prepare_config_and_inputs_with_token_type_ids_and_labels(self):
        batch_size = 2
        seq_length = 3
        vocab_size = 10
        type_sequence_label_size = 2
        type_vocab_size = 5
        num_labels = 3
        num_choices = 4
        use_input_mask = True
        use_token_type_ids = True
        use_labels = True
    
        result = onnxscript.tools.transformers_models.phi._prepare_config_and_inputs(
            batch_size,
            seq_length,
            vocab_size,
            type_sequence_label_size,
            type_vocab_size,
            num_labels,
            num_choices,
            use_input_mask,
            use_token_type_ids,
            use_labels,
        )
    
        input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = result
    
        self.assertIsNotNone(token_type_ids)
        self.assertIsNotNone(sequence_labels)
        self.assertIsNotNone(token_labels)
        self.assertIsNotNone(choice_labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
