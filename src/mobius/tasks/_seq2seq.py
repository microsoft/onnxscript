# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Seq2Seq task for encoder-decoder models (T5, BART, etc.)."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
)


class Seq2SeqTask(ModelTask):
    """Encoder-decoder model for seq2seq generation.

    Produces a ModelPackage with two components:
    - "encoder": input_ids, attention_mask → last_hidden_state
    - "decoder": input_ids, encoder_hidden_states, attention_mask,
                 past_key_values → logits, present_key_values

    The module must have ``encoder`` and ``decoder`` attributes.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        encoder_model = self._build_encoder_graph(module, config)
        decoder_model = self._build_decoder_graph(module, config)
        return ModelPackage(
            {"encoder": encoder_model, "decoder": decoder_model},
            config=config,
        )

    def _build_encoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([input_ids, attention_mask], name="encoder")
        op = builder.op

        encoder_hidden_states = module.encoder(
            op, input_ids=input_ids, attention_mask=attention_mask
        )

        encoder_hidden_states.name = "last_hidden_state"
        graph.outputs.append(encoder_hidden_states)

        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        dec_seq_len = ir.SymbolicDim("decoder_sequence_len")
        enc_seq_len = ir.SymbolicDim("encoder_sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, dec_seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        encoder_hidden_states = ir.Value(
            name="encoder_hidden_states",
            shape=ir.Shape([batch, enc_seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + dec_seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids, encoder_hidden_states, attention_mask]

        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        num_decoder_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)

        # Self-attention KV cache
        self_kv_inputs, past_self_kvs = _make_kv_cache_inputs(
            num_decoder_layers,
            num_heads,
            head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )
        # Use .self. naming for seq2seq self-attention KVs
        for v in self_kv_inputs:
            idx = v.name.split(".")[1]
            kv_type = v.name.rsplit(".", 1)[-1]
            v.name = f"past_key_values.{idx}.self.{kv_type}"
        graph_inputs.extend(self_kv_inputs)

        # Cross-attention KV cache
        cross_kv_inputs, cross_past_kvs = _make_kv_cache_inputs(
            num_decoder_layers,
            num_heads,
            head_dim,
            config.dtype,
            batch,
            enc_seq_len,
            prefix="past_key_values",
        )
        for v in cross_kv_inputs:
            idx = v.name.split(".")[1]
            kv_type = v.name.rsplit(".", 1)[-1]
            v.name = f"past_key_values.{idx}.cross.{kv_type}"
        graph_inputs.extend(cross_kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        op = builder.op

        logits, present_self_kvs, present_cross_kvs = module.decoder(
            op,
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_self_kvs,
            cross_past_key_values=cross_past_kvs,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        for i, (k, v) in enumerate(present_self_kvs):
            k.name = f"present.{i}.self.key"
            v.name = f"present.{i}.self.value"
            graph.outputs.extend([k, v])

        for i, (k, v) in enumerate(present_cross_kvs):
            k.name = f"present.{i}.cross.key"
            v.name = f"present.{i}.cross.value"
            graph.outputs.extend([k, v])

        return _make_model(graph)
