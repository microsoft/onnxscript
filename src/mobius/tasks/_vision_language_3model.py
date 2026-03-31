# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Vision-language 3-model split tasks.

Builds three separate ONNX models:
1. **decoder** (text decoder): inputs_embeds → logits + KV cache
2. **vision** (vision encoder): pixel_values → image_features
3. **embedding**: input_ids + image_features → inputs_embeds
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig, MllamaConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _make_model,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


class VisionLanguageTask(ModelTask):
    """3-model split vision-language task.

    The module must provide three sub-modules as attributes:

    - ``decoder``: text decoder taking ``inputs_embeds``
    - ``vision_encoder``: vision encoder taking ``pixel_values``
    - ``embedding``: embedding model fusing text + image features

    Each sub-module is wired into its own ONNX graph.

    Subclass and override ``_build_vision`` or ``_build_decoder`` for
    non-standard I/O (e.g. Qwen2.5-VL packed attention with cu_seqlens).
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["decoder"] = self._build_decoder(module.decoder, config)
        models["vision"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)

        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build text decoder: inputs_embeds -> logits + KV cache."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build vision encoder: pixel_values [batch, C, H, W] -> image_features."""
        batch = ir.SymbolicDim("batch")
        image_size = config.vision.image_size or 224 if config.vision else 224

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, 3, image_size, image_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [pixel_values]

        graph, graph_builder = _make_graph(graph_inputs, name="vision")
        op = graph_builder.op

        image_features = vision(
            op,
            pixel_values=pixel_values,
        )

        image_features.name = "image_features"
        graph.outputs.append(image_features)

        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding model: input_ids + image_features -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        image_features = ir.Value(
            name="image_features",
            shape=ir.Shape([num_image_tokens, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [input_ids, image_features]

        graph, graph_builder = _make_graph(graph_inputs, name="embedding")
        op = graph_builder.op

        inputs_embeds = embedding(
            op,
            input_ids=input_ids,
            image_features=image_features,
        )

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)

        return _make_model(graph)


class QwenVLTask(VisionLanguageTask):
    """Qwen-family VL 3-model split with packed-attention vision and MRoPE.

    Used by Qwen2.5-VL, Qwen3-VL, and Qwen3.5-VL.  Overrides
    ``_build_vision`` for the packed-attention I/O contract and
    ``_build_decoder`` for MRoPE 3D position_ids.
    """

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build text decoder with MRoPE 3D position_ids."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([3, batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build Qwen VL vision encoder with packed patches and grid_thw."""
        total_patches = ir.SymbolicDim("total_patches")
        num_images = ir.SymbolicDim("num_images")

        patch_size = config.vision.patch_size or 14 if config.vision else 14  # 16 for Qwen3-VL
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.vision.in_channels if config.vision else 3
        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([total_patches, pixel_dim]),
            type=ir.TensorType(config.dtype),
        )
        image_grid_thw = ir.Value(
            name="image_grid_thw",
            shape=ir.Shape([num_images, 3]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [pixel_values, image_grid_thw]

        graph, graph_builder = _make_graph(graph_inputs, name="vision")
        op = graph_builder.op

        image_features = vision(
            op,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        image_features.name = "image_features"
        graph.outputs.append(image_features)

        return _make_model(graph)


class HybridQwenVLTask(QwenVLTask):
    """Qwen VL 3-model split with hybrid KV + DeltaNet cache.

    Used by Qwen3.5-VL which has mixed ``"full_attention"`` and
    ``"linear_attention"`` (DeltaNet) layers.  Vision and embedding
    models are identical to :class:`QwenVLTask`; only the decoder
    uses hybrid cache inputs/outputs.
    """

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build text decoder with MRoPE 3D position_ids and hybrid cache."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # MRoPE: 3D position IDs (temporal, height, width)
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([3, batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        cache_inputs, past_key_values = _make_hybrid_cache_inputs(
            config,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(cache_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_hybrid_cache_outputs(
            graph,
            present_key_values,
            config.layer_types or [],
        )

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return model


class MllamaVisionLanguageTask(VisionLanguageTask):
    """Mllama VL task with cross-attention KV caching.

    Mllama's text decoder has interleaved self-attention and cross-attention
    layers.  Cross-attention layers attend to vision encoder output, which
    is constant during generation.  This task:

    1. Adds ``cross_attention_states`` as a decoder graph input.
    2. Uses separate KV cache symbolic dims for self-attention
       (``past_sequence_len``) and cross-attention (``cross_past_seq_len``).

    At runtime the host passes the full vision features on prefill
    (``cross_attention_states`` has shape ``[B, N, H]``), then an empty
    tensor on decode (``[B, 0, H]``).  The cross-attention KV cache is
    populated during prefill and reused unchanged on every decode step --
    the K/V projection on a 0-length input is essentially free and the
    ``op.Attention`` concat with empty new K/V returns past unchanged.
    """

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder with cross_attention_states input and split KV cache."""
        assert isinstance(config, MllamaConfig)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        cross_seq_len = ir.SymbolicDim("cross_sequence_len")
        cross_past_seq_len = ir.SymbolicDim("cross_past_seq_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # Vision features: full on prefill, empty (0-length) on decode
        cross_attention_states = ir.Value(
            name="cross_attention_states",
            shape=ir.Shape([batch, cross_seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [
            inputs_embeds,
            attention_mask,
            position_ids,
            cross_attention_states,
        ]

        # Create per-layer KV cache with separate dims for
        # self-attention (past_seq_len) and cross-attention
        # (cross_past_seq_len)
        cross_attention_layers = set(config.cross_attention_layers or [])
        flat_kv: list[ir.Value] = []
        past_key_values: list[tuple[ir.Value, ir.Value]] = []

        for i in range(config.num_hidden_layers):
            if i in cross_attention_layers:
                psl = cross_past_seq_len
            else:
                psl = past_seq_len
            past_key = ir.Value(
                name=f"past_key_values.{i}.key",
                shape=ir.Shape([batch, config.num_key_value_heads, psl, config.head_dim]),
                type=ir.TensorType(config.dtype),
            )
            past_value = ir.Value(
                name=f"past_key_values.{i}.value",
                shape=ir.Shape([batch, config.num_key_value_heads, psl, config.head_dim]),
                type=ir.TensorType(config.dtype),
            )
            flat_kv.extend([past_key, past_value])
            past_key_values.append((past_key, past_value))

        graph_inputs.extend(flat_kv)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)
