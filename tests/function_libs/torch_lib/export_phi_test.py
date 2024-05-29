import copy
import random
import sys
import unittest
from typing import Any, Sequence, Tuple

import numpy as np
import onnx.inliner
import onnxruntime
import torch
from torch.onnx import ExportOptions
from torch.onnx import _OrtBackend as OrtBackend
from torch.onnx import _OrtBackendOptions as OrtBackendOptions

import onnxscript.optimizer
import onnxscript.rewriter

# Common functions


def has_transformers():
    try:
        import transformers

        assert transformers
        return True
    except ImportError:
        return False


def export_to_onnx(model, *input_tensors, optimize=True):
    prog = torch.onnx.dynamo_export(model, *input_tensors)
    model_proto = prog.model_proto
    if optimize:
        model_proto = onnxscript.optimizer.optimize(
            model_proto,
            num_iterations=2,
            onnx_shape_inference=True,
        )
        model_proto = onnxscript.rewriter.rewrite(model_proto)
        model_proto = onnx.inliner.inline_local_functions(model_proto)
    return model_proto


def make_aot_ort(dynamic: bool = False):
    export_options = ExportOptions(dynamic_shapes=dynamic)
    options = OrtBackendOptions(export_options=export_options)
    ort_backend = OrtBackend(options=options)
    return ort_backend


def train_loop(model, *args, loss_fn=None, optimizer=None):

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # Compute prediction and loss
    pred = model(*args)
    if isinstance(pred, tuple):
        v = pred[0]
    elif hasattr(pred, "last_hidden_state"):
        v = pred.last_hidden_state
    else:
        v = pred
    loss = loss_fn(v, torch.ones_like(v))

    # Backpropagation
    loss.backward()
    optimizer.step()
    # skip that part to retrieve the gradients
    # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"
    return res


# Specific functions


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    import torch

    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def _prepare_config_and_inputs(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    type_sequence_label_size: int = 2,
    type_vocab_size: int = 16,
    num_labels: int = 3,
    num_choices: int = 4,
    use_input_mask: bool = False,
    use_token_type_ids: bool = False,
    use_labels: bool = False,
) -> Tuple[Any]:
    import torch

    input_ids = ids_tensor([batch_size, seq_length], vocab_size)

    input_mask = None
    if use_input_mask:
        input_mask = torch.tril(torch.ones(batch_size, seq_length))

    token_type_ids = None
    if use_token_type_ids:
        assert type_vocab_size > 0, "type_vocab_size is null"
        token_type_ids = ids_tensor([batch_size, seq_length], type_vocab_size)

    sequence_labels = None
    token_labels = None
    choice_labels = None
    if use_labels:
        assert type_sequence_label_size > 0, "type_sequence_label_size is null"
        assert num_labels > 0, "num_labels is null"
        assert num_choices > 0, "num_choices is null"
        sequence_labels = ids_tensor([batch_size], type_sequence_label_size)
        token_labels = ids_tensor([batch_size, seq_length], num_labels)
        choice_labels = ids_tensor([batch_size], num_choices)

    return (
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    )


def get_phi_model(
    input_dims: Sequence[Tuple[int, int]] = ((13, 7), (14, 7), (15, 8)),
    hidden_size=32,
    num_hidden_layers=2,
    vocab_size=99,
    intermediate_size=16,
    max_position_embeddings=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    _attn_implementation="eager",  # needed value to remove graph breaks
    with_mask: bool = True,
):
    """
    Returns a model.
    See `PhiConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/phi#transformers.PhiConfig>`_.
    The parameters are chosen for a unit test configuration from `test_modeling_phi.py
    <https://github.com/huggingface/transformers/blob/main/tests/models/phi/test_modeling_phi.py>`_.
    """
    from transformers import PhiConfig
    from transformers.models.phi.modeling_phi import PhiModel

    config = PhiConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    if with_mask:

        class PhiModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = PhiModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(input_ids, attention_mask=attention_mask)
                return model_output.to_tuple()

        def generate_example_inputs(batch: int, seq: int, vocab_size: int):
            (
                input_ids,
                token_type_ids,
                input_mask,
                sequence_labels,
                token_labels,
                choice_labels,
            ) = _prepare_config_and_inputs(
                batch_size=batch,
                seq_length=seq,
                vocab_size=vocab_size,
                use_input_mask=True,
            )
            return input_ids, input_mask

        example_args_collection = []
        for b, s in input_dims:
            example_args_collection.append(generate_example_inputs(b, s, vocab_size))

        return PhiModelWrapper(config), example_args_collection

    # no mask

    class PhiModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = PhiModel(config)

        def forward(self, input_ids):
            model_output = self.model(input_ids)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        (
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = _prepare_config_and_inputs(
            batch_size=batch,
            seq_length=seq,
            vocab_size=vocab_size,
            use_input_mask=True,
        )
        return (input_ids,)

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    return PhiModelWrapper(config), example_args_collection


class TestExportPhi(unittest.TestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    def test_phi_export_cpu(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        proto = export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        xp = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, xp))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    def test_phi_export_cuda(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        model = model.to("cuda")
        input_tensors = [i.to("cuda") for i in input_tensors]
        expected = model(*input_tensors)
        proto = export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        xp = [x.detach().cpu().numpy() for x in input_tensors]
        feeds = dict(zip(names, xp))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    def test_phi_dort_static(self):
        model, input_tensors = get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        local_aot_ort = make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        results = compiled_model(*input_tensors)
        torch.testing.assert_allclose(expected[0], results[0], atol=1e-5, rtol=1e-5)

        expected_gradients = train_loop(model, *input_tensors)
        gradients = train_loop(compiled_model, *input_tensors)
        torch.testing.assert_allclose(
            expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
