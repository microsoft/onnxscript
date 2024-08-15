# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=import-outside-toplevel
from __future__ import annotations

from typing import Any, Sequence

import torch

import onnxscript.tools.transformers_models


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
) -> tuple[Any, ...]:
    input_ids = onnxscript.tools.transformers_models.ids_tensor(
        [batch_size, seq_length], vocab_size
    )

    input_mask = None
    if use_input_mask:
        input_mask = torch.tril(torch.ones(batch_size, seq_length))

    token_type_ids = None
    if use_token_type_ids:
        assert type_vocab_size > 0, "type_vocab_size is null"
        token_type_ids = onnxscript.tools.transformers_models.ids_tensor(
            [batch_size, seq_length], type_vocab_size
        )

    sequence_labels = None
    token_labels = None
    choice_labels = None
    if use_labels:
        assert type_sequence_label_size > 0, "type_sequence_label_size is null"
        assert num_labels > 0, "num_labels is null"
        assert num_choices > 0, "num_choices is null"
        sequence_labels = onnxscript.tools.transformers_models.ids_tensor(
            [batch_size], type_sequence_label_size
        )
        token_labels = onnxscript.tools.transformers_models.ids_tensor(
            [batch_size, seq_length], num_labels
        )
        choice_labels = onnxscript.tools.transformers_models.ids_tensor(
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


def get_mistral_model(
    input_dims: Sequence[tuple[int, int]] = ((13, 7), (14, 7), (15, 8)),
    hidden_size=32,
    num_hidden_layers=2,
    vocab_size=99,
    intermediate_size=16,
    max_position_embeddings=512,
    num_attention_heads=2,
    num_key_value_heads=2,
    sliding_window=4096,
    _attn_implementation="eager",  # needed value to remove graph breaks
    with_mask: bool = True,
) -> tuple[Any, list[tuple[torch.Tensor, ...]], dict]:
    """
    Returns a model.
    See `MistralConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralConfig>`_.
    The parameters are chosen for a unit test configuration.
    """
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    config = MistralConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        sliding_window=sliding_window,
    )

    dynamic_shapes = {0: {0: "batch", 1: "length"}}
    if with_mask:
        dynamic_shapes.update({1: {0: "batch", 1: "length"}})

    if _attn_implementation:
        config._attn_implementation = _attn_implementation  # pylint: disable=protected-access

    def generate_example_inputs(batch: int, seq: int, vocab_size: int, with_mask: bool):
        (
            input_ids,
            _,  # token_type_ids,
            input_mask,
            _,  # sequence_labels,
            _,  # token_labels,
            _,  # choice_labels,
        ) = _prepare_config_and_inputs(
            batch_size=batch,
            seq_length=seq,
            vocab_size=vocab_size,
            use_input_mask=with_mask,
        )
        if with_mask:
            return input_ids, input_mask
        return (input_ids,)

    if with_mask:

        class MistralModelWrapperWithMask(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = MistralModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(
                    input_ids, attention_mask=attention_mask, use_cache=False
                )
                return model_output.to_tuple()

        example_args_collection = []
        for b, s in input_dims:
            example_args_collection.append(
                generate_example_inputs(b, s, vocab_size, with_mask)
            )

        return MistralModelWrapperWithMask(config), example_args_collection, dynamic_shapes

    class MistralModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = MistralModel(config)

        def forward(self, input_ids):
            model_output = self.model(input_ids, use_cache=False)
            return model_output.to_tuple()

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size, with_mask))

    return MistralModelWrapper(config), example_args_collection, dynamic_shapes


def get_mistral_model_from_config(
    warmup: int = 5,
    repeat: int = 10,
    config: str = "small",
    num_hidden_layers: int = 1,
    implementation: str = "eager",
    dynamic_shapes: bool = False,
    with_mask: bool = True,
) -> tuple[Any, list[tuple[torch.Tensor, ...]], dict]:
    """
    Returns a model Phi to test or benchmark.

    Args:
        warmup: Number of inputs to generate.
        repeat: Number of inputs to generate for repeat.
        config: small, medium or large
        num_hidden_layers: number of hidden layers
        implementation: eager or sdpa
        with_mask: One or two inputs.
        dynamic_shapes: dynamic shapes or not

    Returns:
        Model and list of inputs.
    """
    if config == "small":
        conf_dict = dict(
            input_dims=onnxscript.tools.transformers_models.get_input_dims_for_llm(
                dynamic_shapes, warmup, repeat
            ),
            hidden_size=32,
            num_hidden_layers=num_hidden_layers,
            vocab_size=99,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            _attn_implementation=implementation,
            with_mask=with_mask,
        )
    elif config == "medium":
        conf_dict = dict(
            input_dims=onnxscript.tools.transformers_models.get_input_dims_for_llm(
                dynamic_shapes, warmup, repeat
            ),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            sliding_window=4096,
            _attn_implementation=implementation,
            with_mask=with_mask,
        )
    elif config in ("large", "default"):
        conf_dict = dict(
            input_dims=onnxscript.tools.transformers_models.get_input_dims_for_llm(
                dynamic_shapes, warmup, repeat
            ),
            hidden_size=4096,
            num_hidden_layers=num_hidden_layers,
            vocab_size=32000,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            sliding_window=4096,
            _attn_implementation=implementation,
            with_mask=with_mask,
        )
    else:
        raise ValueError(f"Unexpected configuration {config!r}.")

    return get_mistral_model(**conf_dict)  # type: ignore[arg-type]
