# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""PyTorch/HuggingFace reference model helpers for integration testing."""

from __future__ import annotations

import numpy as np
import torch


def load_torch_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace causal LM model for reference inference.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Model dtype (default float32 for numerical comparison).
        device: Device to load on.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_torch_multimodal_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace multimodal model for reference inference.

    Uses AutoModelForImageTextToText for vision-language models.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Model dtype (default float32 for numerical comparison).
        device: Device to load on.

    Returns:
        Tuple of (model, tokenizer, image_processor).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor


@torch.no_grad()
def torch_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    past_key_values: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run a single forward pass on a HuggingFace causal LM model.

    Args:
        model: HuggingFace model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        attention_mask: [batch, total_seq_len] int64 numpy array.
        position_ids: [batch, seq_len] int64 numpy array.
        past_key_values: Optional list of (key, value) numpy array tuples.

    Returns:
        Tuple of (logits as numpy, list of (key, value) numpy tuples).
    """
    import inspect

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    pos_t = torch.from_numpy(position_ids).to(device)

    kwargs: dict = {
        "input_ids": ids_t,
        "attention_mask": mask_t,
        "use_cache": True,
    }

    # Some models (Falcon, Mamba) don't accept position_ids
    fwd_sig = inspect.signature(model.forward)
    if "position_ids" in fwd_sig.parameters:
        kwargs["position_ids"] = pos_t

    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(
                torch.from_numpy(k).to(device=device, dtype=dtype),
                torch.from_numpy(v).to(device=device, dtype=dtype),
                layer_idx,
            )
        kwargs["past_key_values"] = cache

    outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()

    # Extract KV cache if available (Mamba models don't have it)
    present_kv: list[tuple[np.ndarray, np.ndarray]] = []
    cache = getattr(outputs, "past_key_values", None)
    if cache is not None and hasattr(cache, "layers"):
        for layer_idx in range(len(cache.layers)):
            k = cache.layers[layer_idx].keys.cpu().numpy()
            v = cache.layers[layer_idx].values.cpu().numpy()
            present_kv.append((k, v))

    return logits, present_kv


@torch.no_grad()
def torch_multimodal_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    pixel_values: np.ndarray,
    past_key_values: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run a single forward pass on a HuggingFace multimodal model.

    Args:
        model: HuggingFace multimodal model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        attention_mask: [batch, total_seq_len] int64 numpy array.
        position_ids: [batch, seq_len] int64 numpy array.
        pixel_values: [batch, channels, height, width] float32 numpy array.
        past_key_values: Optional list of (key, value) numpy array tuples.

    Returns:
        Tuple of (logits as numpy, list of (key, value) numpy tuples).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    pos_t = torch.from_numpy(position_ids).to(device)
    pv_t = torch.from_numpy(pixel_values).to(device=device, dtype=dtype)

    kwargs: dict = {
        "input_ids": ids_t,
        "attention_mask": mask_t,
        "position_ids": pos_t,
        "pixel_values": pv_t,
        "use_cache": True,
    }

    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(
                torch.from_numpy(k).to(device=device, dtype=dtype),
                torch.from_numpy(v).to(device=device, dtype=dtype),
                layer_idx,
            )
        kwargs["past_key_values"] = cache

    outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()

    present_kv = []
    cache = outputs.past_key_values
    for layer_idx in range(len(cache.layers)):
        k = cache.layers[layer_idx].keys.cpu().numpy()
        v = cache.layers[layer_idx].values.cpu().numpy()
        present_kv.append((k, v))

    return logits, present_kv


# ---------------------------------------------------------------------------
# Encoder-only models (BERT, RoBERTa, DistilBERT, etc.)
# ---------------------------------------------------------------------------


def load_torch_encoder_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace encoder-only model for reference inference.

    Uses AutoModel (not AutoModelForCausalLM) for encoder-only architectures.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def torch_encoder_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_type_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace encoder-only model.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, hidden_size].
    """
    device = next(model.parameters()).device
    kwargs: dict = {
        "input_ids": torch.from_numpy(input_ids).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
    }
    if token_type_ids is not None:
        kwargs["token_type_ids"] = torch.from_numpy(token_type_ids).to(device)
    outputs = model(**kwargs)
    return outputs.last_hidden_state.cpu().numpy()


# ---------------------------------------------------------------------------
# Seq2seq models (BART, T5, mBART, etc.)
# ---------------------------------------------------------------------------


def load_torch_seq2seq_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace seq2seq model for reference inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def torch_seq2seq_encoder_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Run the encoder of a seq2seq model and return hidden states.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, d_model].
    """
    device = next(model.parameters()).device
    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    encoder_out = model.get_encoder()(input_ids=ids_t, attention_mask=mask_t)
    return encoder_out.last_hidden_state.cpu().numpy()


@torch.no_grad()
def torch_seq2seq_decoder_forward(
    model,
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray | None = None,
    past_key_values=None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the decoder of a seq2seq model and return logits + KV cache.

    Returns:
        Tuple of (logits as numpy, past_key_values for next step).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    kwargs: dict = {
        "decoder_input_ids": torch.from_numpy(decoder_input_ids).to(device),
        "encoder_outputs": (
            torch.from_numpy(encoder_hidden_states).to(device=device, dtype=dtype),
        ),
        "use_cache": True,
    }
    if encoder_attention_mask is not None:
        kwargs["attention_mask"] = torch.from_numpy(encoder_attention_mask).to(device)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()
    return logits, outputs.past_key_values


# ---------------------------------------------------------------------------
# Vision models (ViT, DeiT, CLIP Vision, etc.)
# ---------------------------------------------------------------------------


def load_torch_vision_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace vision model for reference inference.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    processor = transformers.AutoImageProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = transformers.AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


@torch.no_grad()
def torch_vision_forward(
    model,
    pixel_values: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace vision model.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, hidden_size].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pv = torch.from_numpy(pixel_values).to(device=device, dtype=dtype)
    outputs = model(pixel_values=pv)
    return outputs.last_hidden_state.cpu().numpy()


# ---------------------------------------------------------------------------
# Audio models (Wav2Vec2, HuBERT, WavLM, etc.)
# ---------------------------------------------------------------------------


def load_torch_audio_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace audio model for reference inference.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


@torch.no_grad()
def torch_audio_forward(
    model,
    input_values: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace audio model.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, hidden_size].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    iv = torch.from_numpy(input_values).to(device=device, dtype=dtype)
    outputs = model(input_values=iv)
    return outputs.last_hidden_state.cpu().numpy()


# ---------------------------------------------------------------------------
# Whisper encoder-decoder models
# ---------------------------------------------------------------------------


@torch.no_grad()
def load_torch_whisper_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace Whisper model for reference inference.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


@torch.no_grad()
def torch_whisper_encoder_forward(
    model,
    input_features: np.ndarray,
) -> np.ndarray:
    """Run the Whisper encoder and return encoder hidden states.

    Args:
        model: HuggingFace WhisperForConditionalGeneration in eval mode.
        input_features: [batch, num_mel_bins, audio_seq_len] float32 numpy array.

    Returns:
        encoder_hidden_states as numpy array [batch, seq/2, d_model].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    feats = torch.from_numpy(input_features).to(device=device, dtype=dtype)
    encoder_out = model.model.encoder(feats)
    return encoder_out.last_hidden_state.cpu().numpy()


@torch.no_grad()
def torch_whisper_decoder_forward(
    model,
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    attention_mask: np.ndarray | None = None,
    past_key_values: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the Whisper decoder and return logits + KV cache.

    Args:
        model: HuggingFace WhisperForConditionalGeneration in eval mode.
        decoder_input_ids: [batch, seq_len] int64 numpy array.
        encoder_hidden_states: [batch, enc_seq, d_model] float32 numpy array.
        attention_mask: Optional [batch, total_seq_len] int64.
        past_key_values: Optional list of (key, value) numpy tuples.

    Returns:
        Tuple of (logits as numpy, list of (key, value) numpy tuples).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(decoder_input_ids).to(device)
    enc_t = torch.from_numpy(encoder_hidden_states).to(device=device, dtype=dtype)

    kwargs: dict = {
        "input_ids": ids_t,
        "encoder_hidden_states": enc_t,
        "use_cache": True,
    }

    if attention_mask is not None:
        kwargs["attention_mask"] = torch.from_numpy(attention_mask).to(device)

    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            self_cache.update(
                torch.from_numpy(k).to(device=device, dtype=dtype),
                torch.from_numpy(v).to(device=device, dtype=dtype),
                layer_idx,
            )
        # Whisper decoder expects EncoderDecoderCache wrapping self + cross caches
        cross_cache = DynamicCache()
        kwargs["past_key_values"] = EncoderDecoderCache(self_cache, cross_cache)

    outputs = model.model.decoder(**kwargs)
    hidden_states = outputs.last_hidden_state

    # Project to vocab
    logits = model.proj_out(hidden_states).cpu().numpy()

    # Extract self-attention KV cache (Whisper decoder uses EncoderDecoderCache)
    present_kv = []
    cache = outputs.past_key_values
    # EncoderDecoderCache wraps self_attention_cache + cross_attention_cache
    self_cache = getattr(cache, "self_attention_cache", cache)
    for layer_idx in range(len(self_cache.layers)):
        k = self_cache.layers[layer_idx].keys.cpu().numpy()
        v = self_cache.layers[layer_idx].values.cpu().numpy()
        present_kv.append((k, v))

    return logits, present_kv
