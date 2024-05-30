# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any, Sequence, Tuple

import torch

import onnxscript.testing.transformers_models


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
    input_ids = onnxscript.testing.transformers_models.ids_tensor(
        [batch_size, seq_length], vocab_size
    )

    input_mask = None
    if use_input_mask:
        input_mask = torch.tril(torch.ones(batch_size, seq_length))

    token_type_ids = None
    if use_token_type_ids:
        assert type_vocab_size > 0, "type_vocab_size is null"
        token_type_ids = onnxscript.testing.transformers_models.ids_tensor(
            [batch_size, seq_length], type_vocab_size
        )

    sequence_labels = None
    token_labels = None
    choice_labels = None
    if use_labels:
        assert type_sequence_label_size > 0, "type_sequence_label_size is null"
        assert num_labels > 0, "num_labels is null"
        assert num_choices > 0, "num_choices is null"
        sequence_labels = onnxscript.testing.transformers_models.ids_tensor(
            [batch_size], type_sequence_label_size
        )
        token_labels = onnxscript.testing.transformers_models.ids_tensor(
            [batch_size, seq_length], num_labels
        )
        choice_labels = onnxscript.testing.transformers_models.ids_tensor(
            [batch_size], num_choices
        )

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
