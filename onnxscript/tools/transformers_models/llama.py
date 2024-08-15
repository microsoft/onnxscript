# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=import-outside-toplevel
from __future__ import annotations

from typing import Any, Sequence

import torch

import onnxscript.tools.transformers_models


def get_llama_model(
    input_dims: Sequence[tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size: int = 16,
    num_hidden_layers: int = 1,
    vocab_size: int = 1024,
    intermediate_size: int = 16,
    max_position_embeddings: int = 1024,
    num_attention_heads: int = 2,
    _attn_implementation: str = "eager",  # needed value to remove graph breaks
    with_mask: bool = True,
) -> tuple[Any, list[tuple[torch.Tensor, ...]], dict]:
    """
    Returns a model.
    See `LlamaConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig>`_.
    The parameters are chosen for a unit test configuration.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    dynamic_shapes = {0: {0: "batch", 1: "length"}}
    if with_mask:
        dynamic_shapes.update({1: {0: "batch", 1: "length"}})

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation  # pylint: disable=protected-access

    if with_mask:

        class LlamaModelWrapperMask(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = LlamaModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(
                    input_ids, attention_mask=attention_mask, use_cache=False
                )
                return model_output.to_tuple()

        def generate_example_inputs_mask(batch: int, seq: int, vocab_size: int):
            input_ids = onnxscript.tools.transformers_models.ids_tensor(
                [batch, seq], vocab_size
            )
            input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
            assert input_mask.dtype == torch.float32
            return input_ids, input_mask

        example_args_collection = []
        for b, s in input_dims:
            example_args_collection.append(generate_example_inputs_mask(b, s, vocab_size))

        return LlamaModelWrapperMask(config), example_args_collection, dynamic_shapes

    # no mask

    class LlamaModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = LlamaModel(config)

        def forward(self, input_ids):
            model_output = self.model(input_ids, use_cache=False)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        input_ids = onnxscript.tools.transformers_models.ids_tensor([batch, seq], vocab_size)
        return (input_ids,)

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    return LlamaModelWrapper(config), example_args_collection, dynamic_shapes


def get_llama_model_from_config(
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
        num_hidden_layers: Number of hidden layers.
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
            hidden_size=16,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
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
            max_position_embeddings=1024,
            num_attention_heads=2,
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
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
            _attn_implementation=implementation,
            with_mask=with_mask,
        )
    else:
        raise ValueError(f"Unexpected configuration {config!r}.")

    return get_llama_model(**conf_dict)  # type: ignore[arg-type]
