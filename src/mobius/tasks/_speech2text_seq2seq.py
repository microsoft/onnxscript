# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Speech2Text seq2seq task for Facebook S2T ASR models.

Unlike the standard ``Seq2SeqTask`` (which takes token IDs as encoder input),
this task builds an encoder graph that accepts raw mel spectrogram features.
The decoder follows the standard seq2seq pattern with separate self-attention
and cross-attention KV caches.

Encoder inputs:
    input_features: (batch, seq_len, feat_dim) FLOAT  — mel spectrogram

Encoder outputs:
    encoder_hidden_states: (batch, seq_len // 4, hidden_size)

Decoder inputs:
    input_ids: (batch, dec_seq_len) INT64
    encoder_hidden_states: (batch, enc_seq_len, hidden_size)
    attention_mask: (batch, past_seq_len + dec_seq_len) INT64
    past_key_values.{i}.self.{key,value}: (batch, heads, past_seq_len, head_dim)
    past_key_values.{i}.cross.{key,value}: (batch, heads, enc_seq_len, head_dim)

Decoder outputs:
    logits: (batch, dec_seq_len, vocab_size)
    present.{i}.self.{key,value}: updated self-attention cache
    present.{i}.cross.{key,value}: updated cross-attention cache
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Speech2TextConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
)


class Speech2TextSeq2SeqTask(ModelTask):
    """Encoder-decoder task for Speech2Text ASR models.

    Produces a ``ModelPackage`` with two ONNX models:
    - ``"encoder"``: input_features → encoder_hidden_states
    - ``"decoder"``: input_ids, encoder_hidden_states, attention_mask,
      past self+cross KV → logits, present self+cross KV

    The module must expose ``encoder`` and ``decoder`` attributes
    (matching ``Speech2TextForConditionalGeneration`` layout).
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, Speech2TextConfig), (
            f"Speech2TextSeq2SeqTask requires Speech2TextConfig, got {type(config).__name__}"
        )
        encoder_model = self._build_encoder(module.encoder, config)
        decoder_model = self._build_decoder(module.decoder, config)
        return ModelPackage(
            {"encoder": encoder_model, "decoder": decoder_model},
            config=config,
        )

    def _build_encoder(
        self,
        encoder: nn.Module,
        config: Speech2TextConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        audio_seq_len = ir.SymbolicDim("audio_seq_len")
        feat_dim = config.input_feat_per_channel * config.input_channels

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, audio_seq_len, feat_dim]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, builder = _make_graph([input_features], name="encoder")
        op = builder.op

        encoder_hidden_states = encoder(op, input_features=input_features)

        encoder_hidden_states.name = "encoder_hidden_states"
        graph.outputs.append(encoder_hidden_states)

        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: Speech2TextConfig,
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
        num_decoder_layers = config.num_decoder_layers or config.num_hidden_layers

        # Self-attention KV cache (grows with sequence length)
        self_kv_inputs, past_self_kvs = _make_kv_cache_inputs(
            num_decoder_layers,
            num_heads,
            head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )
        for v in self_kv_inputs:
            idx = v.name.split(".")[1]
            kv_type = v.name.rsplit(".", 1)[-1]
            v.name = f"past_key_values.{idx}.self.{kv_type}"
        graph_inputs.extend(self_kv_inputs)

        # Cross-attention KV cache (fixed at encoder_seq_len)
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

        logits, present_self_kvs, present_cross_kvs = decoder(
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
