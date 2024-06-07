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
        config._attn_implementation = _attn_implementation  # type: ignore[attr-defined]

    if with_mask:

        class LlamaModelWrapperMask(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = LlamaModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(input_ids, attention_mask=attention_mask)
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
            model_output = self.model(input_ids)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        input_ids = onnxscript.tools.transformers_models.ids_tensor([batch, seq], vocab_size)
        return (input_ids,)

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    return LlamaModelWrapper(config), example_args_collection, dynamic_shapes
