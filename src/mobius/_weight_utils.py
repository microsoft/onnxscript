# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared weight preprocessing utilities for HuggingFace → ONNX conversion.

These helpers handle common weight transformations that appear across
multiple model architectures, such as splitting fused QKV projections
and fused gate/up projections.

Note: Some models (InternLM, Phi3-Small) use grouped/interleaved QKV
layouts that require reshape-based splitting. Those are too
model-specific to generalize here and remain inline.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    default_alpha: float | None = None,
) -> dict[str, torch.Tensor]:
    """Merge LoRA adapter weights into base model weights.

    Detects ``*.lora_A.weight`` / ``*.lora_B.weight`` pairs in
    *lora_state_dict* and merges them into *base_state_dict* using::

        merged = base + (alpha / rank) * (B @ A)

    where ``rank = A.shape[0]`` and ``alpha`` is read from
    ``*.lora_A.alpha`` (a scalar tensor) or falls back to
    *default_alpha* (which defaults to ``rank`` if not provided).

    Keys in *lora_state_dict* that are not LoRA deltas are ignored.

    Args:
        base_state_dict: Base model weights (modified in-place).
        lora_state_dict: PEFT adapter weights containing
            ``lora_A.weight``, ``lora_B.weight``, and optionally
            ``lora_A.alpha`` tensors.
        default_alpha: Fallback scaling alpha when no per-layer alpha
            tensor is found.  Defaults to ``rank`` (i.e. scale = 1.0).

    Returns:
        *base_state_dict* with LoRA deltas merged in.
    """
    # Collect LoRA A matrices keyed by their base weight name.
    # e.g. "model.layers.0.self_attn.q_proj.lora_A.weight"
    #   → base_key = "model.layers.0.self_attn.q_proj.weight"
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}

    for key, value in lora_state_dict.items():
        if key.endswith(".lora_A.weight"):
            base_key = key.replace(".lora_A.weight", ".weight")
            lora_a[base_key] = value
        elif key.endswith(".lora_B.weight"):
            base_key = key.replace(".lora_B.weight", ".weight")
            lora_b[base_key] = value
        elif key.endswith(".alpha"):
            # "...lora_A.alpha" or "...lora_B.alpha" — both map to same base
            base_key = key.rsplit(".lora_", 1)[0] + ".weight"
            alphas[base_key] = float(value)

    merged_count = 0
    for base_key in lora_a:
        if base_key not in lora_b:
            logger.warning(
                "LoRA lora_A found without matching lora_B for '%s'",
                base_key,
            )
            continue
        if base_key not in base_state_dict:
            logger.warning(
                "LoRA target '%s' not found in base model weights",
                base_key,
            )
            continue

        a_matrix = lora_a[base_key].float()  # [rank, in_features]
        b_matrix = lora_b[base_key].float()  # [out_features, rank]
        rank = a_matrix.shape[0]
        alpha = alphas.get(base_key, default_alpha if default_alpha is not None else rank)
        scale = alpha / rank

        # merged = base + scale * (B @ A)
        delta = (b_matrix @ a_matrix) * scale
        base_weight = base_state_dict[base_key]
        base_state_dict[base_key] = (base_weight.float() + delta).to(base_weight.dtype)
        merged_count += 1

    if merged_count > 0:
        logger.info("Merged %d LoRA adapter weights into base model", merged_count)
    elif lora_a:
        logger.warning(
            "Found %d LoRA A matrices but merged 0 — check weight name alignment",
            len(lora_a),
        )

    return base_state_dict


def split_fused_qkv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight tensor into separate Q, K, V tensors.

    Handles the common flat layout where Q, K, V are concatenated
    along dimension 0: ``[q_size + kv_size + kv_size, ...]``.

    Args:
        weight: Fused QKV weight tensor with shape
            ``[num_heads*head_dim + 2*num_kv_heads*head_dim, ...]``.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension per attention head.

    Returns:
        Tuple of (q_weight, k_weight, v_weight) split along dim 0.
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    expected = q_size + 2 * kv_size
    if weight.shape[0] != expected:
        raise ValueError(
            f"QKV weight dim 0 is {weight.shape[0]}, expected "
            f"{expected} (num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim})"
        )
    q = weight[:q_size]
    k = weight[q_size : q_size + kv_size]
    v = weight[q_size + kv_size :]
    return q, k, v


def split_interleaved_qkv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight with per-head interleaved layout.

    Used by models like GPT-NeoX and Persimmon where the fused projection
    groups QKV **per head** rather than grouping all Q heads together:

        [h0_q, h0_k, h0_v,  h1_q, h1_k, h1_v, ...]

    This is the layout produced by ``nn.Linear(H, 3*H)`` when the output
    is then reshaped to ``[num_heads, 3, head_dim]`` and indexed.

    Args:
        weight: Fused QKV tensor of shape ``[num_heads * 3 * head_dim, ...]``
            or ``[num_heads * 3 * head_dim]`` for bias vectors.
        num_heads: Number of query attention heads (MHA only: equals num_kv_heads).
        num_kv_heads: Number of key/value heads (must equal num_heads for MHA).
        head_dim: Dimension per attention head.

    Returns:
        Tuple of (q, k, v) each of shape ``[num_heads * head_dim, ...]``.
    """
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(
            f"Interleaved QKV dim 0 is {weight.shape[0]}, expected "
            f"{expected} (num_heads={num_heads}, head_dim={head_dim})"
        )
    if num_kv_heads != num_heads:
        raise ValueError(
            f"split_interleaved_qkv requires MHA (num_kv_heads == num_heads), "
            f"got GQA (num_kv_heads={num_kv_heads}, num_heads={num_heads})"
        )
    rest = weight.shape[1:]  # () for bias, (hidden_size,) for weight
    # Reshape to [num_heads, 3, head_dim, *rest] to un-interleave
    w = weight.reshape(num_heads, 3, head_dim, *rest)
    q = w[:, 0].reshape(num_heads * head_dim, *rest)
    k = w[:, 1].reshape(num_kv_heads * head_dim, *rest)
    v = w[:, 2].reshape(num_kv_heads * head_dim, *rest)
    return q, k, v


def split_interleaved_qkv_weights(
    state_dict: dict[str, torch.Tensor],
    fused_key: str,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Expand all fused interleaved QKV weights in a state dict.

    Scans *state_dict* for keys containing *fused_key* (e.g.
    ``"attention.query_key_value"``), splits each matched weight with
    :func:`split_interleaved_qkv`, and emits three new keys:
    ``{prefix}{attn_name}.q_proj{suffix}``,
    ``{prefix}{attn_name}.k_proj{suffix}``,
    ``{prefix}{attn_name}.v_proj{suffix}``.

    The ``attn_name`` is the segment of *fused_key* up to
    ``.query_key_value`` (e.g. ``"attention"`` or ``"self_attn"``).
    This consolidates the identical scaffolding code that appears in
    GPT-NeoX and Persimmon ``preprocess_weights`` implementations.

    Args:
        state_dict: Input weight dictionary.
        fused_key: Substring identifying fused QKV keys, e.g.
            ``"attention.query_key_value"`` or
            ``"self_attn.query_key_value"``.
        num_heads: Number of query attention heads.
        kv_heads: Number of key/value attention heads.
        head_dim: Dimension per attention head.

    Returns:
        New dictionary with fused QKV keys replaced by split q/k/v keys.
    """
    attn_name = fused_key.rsplit(".query_key_value", 1)[0]
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if fused_key in key:
            q, k, v = split_interleaved_qkv(value, num_heads, kv_heads, head_dim)
            suffix = key.split(fused_key)[1]  # ".weight" or ".bias"
            prefix = key.split(fused_key)[0]  # e.g. "gpt_neox.layers.N."
            result[f"{prefix}{attn_name}.q_proj{suffix}"] = q
            result[f"{prefix}{attn_name}.k_proj{suffix}"] = k
            result[f"{prefix}{attn_name}.v_proj{suffix}"] = v
        else:
            result[key] = value
    return result


def rename_mlp_projections(
    name: str,
    old_up: str,
    old_down: str,
    new_up: str = "up_proj",
    new_down: str = "down_proj",
) -> str:
    """Rename MLP projection weight keys to the canonical ``up_proj``/``down_proj`` names.

    Many HuggingFace models use architecture-specific MLP projection names
    (``fc_in``/``fc_out``, ``c_fc``/``c_proj``, ``dense_h_to_4h``/
    ``dense_4h_to_h``, ``fc1``/``fc2``) while our ONNX ``FCMLP`` component
    always uses ``up_proj``/``down_proj``.  This helper centralises the
    two-replacement pattern that would otherwise be duplicated in every
    ``preprocess_weights`` implementation.

    Args:
        name: A single weight key string.
        old_up: HF name for the first (up) projection, e.g. ``"fc_in"``.
        old_down: HF name for the second (down) projection, e.g. ``"fc_out"``.
        new_up: Target name for the up projection (default ``"up_proj"``).
        new_down: Target name for the down projection (default ``"down_proj"``).

    Returns:
        The key with ``mlp.{old_up}`` → ``mlp.{new_up}`` and
        ``mlp.{old_down}`` → ``mlp.{new_down}`` applied.
    """
    return name.replace(f".mlp.{old_up}.", f".mlp.{new_up}.").replace(
        f".mlp.{old_down}.", f".mlp.{new_down}."
    )


def split_codegen_qkv(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    mp_num: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a CodeGen fused QKV weight with model-parallel interleaved layout.

    CodeGen uses a QVK (not QKV!) layout interleaved by model-parallel blocks:

        [q_mp0, v_mp0, k_mp0,  q_mp1, v_mp1, k_mp1, ...]

    where each block covers ``local_dim = num_heads * head_dim // mp_num``
    output neurons.  After splitting, the heads from each mp-block are
    concatenated to form the full Q, K, V projections.

    Args:
        weight: Fused QKV weight of shape ``[3 * num_heads * head_dim, hidden]``.
            CodeGen QKV has no bias, so this is always 2D.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        mp_num: Number of model-parallel blocks (default 4, matches CodeGen source).

    Returns:
        Tuple of (q, k, v) each of shape ``[num_heads * head_dim, hidden]``.
    """
    total = 3 * num_heads * head_dim
    if weight.shape[0] != total:
        raise ValueError(
            f"CodeGen QKV dim 0 is {weight.shape[0]}, expected {total} "
            f"(num_heads={num_heads}, head_dim={head_dim})"
        )
    if (num_heads * head_dim) % mp_num != 0:
        raise ValueError(
            f"num_heads * head_dim ({num_heads * head_dim}) must be divisible by "
            f"mp_num ({mp_num})"
        )
    local_dim = num_heads * head_dim // mp_num  # output neurons per mp-block per projection
    hidden = weight.shape[1]
    # [mp_num, 3 * local_dim, hidden] — each row is one mp-block
    w = weight.reshape(mp_num, 3 * local_dim, hidden)
    q = w[:, :local_dim, :].reshape(num_heads * head_dim, hidden)
    v = w[:, local_dim : 2 * local_dim, :].reshape(num_heads * head_dim, hidden)
    k = w[:, 2 * local_dim :, :].reshape(num_heads * head_dim, hidden)
    return q, k, v


def split_gate_up_proj(
    weight: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate_up_proj weight into gate_proj and up_proj.

    Handles the common layout where gate and up projections are
    concatenated along dimension 0: ``[2*intermediate_size, ...]``.

    Args:
        weight: Fused gate_up weight tensor with shape
            ``[2*intermediate_size, ...]``.
        intermediate_size: Size of the MLP intermediate layer.

    Returns:
        Tuple of (gate_weight, up_weight) split along dim 0.
    """
    expected = 2 * intermediate_size
    if weight.shape[0] != expected:
        raise ValueError(
            f"gate_up weight dim 0 is {weight.shape[0]}, expected "
            f"{expected} (intermediate_size={intermediate_size})"
        )
    return weight[:intermediate_size], weight[intermediate_size:]


def strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Remove a common prefix from all keys in a state dict.

    Keys that don't start with the prefix are dropped.

    Args:
        state_dict: Weight dictionary to transform.
        prefix: Prefix to strip. A trailing ``.`` is added if not present.

    Returns:
        New dictionary with stripped keys.
    """
    stripped = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(stripped) :]: v for k, v in state_dict.items() if k.startswith(stripped)}


def tie_word_embeddings(
    state_dict: dict[str, torch.Tensor],
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
) -> None:
    """Ensure both embedding and LM head weights are present (tied).

    If one of the two keys is missing, copies the other. This handles
    the common HuggingFace ``tie_word_embeddings=True`` pattern where
    only one of the two weights is stored in the checkpoint.

    Mutates *state_dict* in place.

    Args:
        state_dict: Weight dictionary (modified in place).
        embed_key: Key for the embedding weight.
        head_key: Key for the LM head weight.
    """
    if head_key not in state_dict and embed_key in state_dict:
        state_dict[head_key] = state_dict[embed_key]
    elif embed_key not in state_dict and head_key in state_dict:
        state_dict[embed_key] = state_dict[head_key]


def vlm_decoder_weights(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "language_model.",
    tie: bool = False,
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
) -> dict[str, torch.Tensor]:
    """Extract and rename decoder weights for a VLM model.

    Filters keys starting with *prefix*, strips the prefix, and
    optionally applies embedding/LM-head weight tying.

    This is the standard pattern for VLM decoder sub-models (LLaVA,
    Gemma3, BLIP-2, Mllama) where decoder weights are nested under
    ``language_model.`` in the HuggingFace checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix identifying decoder weights.
        tie: Whether to apply weight tying.
        embed_key: Embedding key (after prefix strip).
        head_key: LM head key (after prefix strip).

    Returns:
        New dictionary with decoder weights (prefix stripped).
    """
    stripped = prefix if prefix.endswith(".") else prefix + "."
    renamed = {k[len(stripped) :]: v for k, v in state_dict.items() if k.startswith(stripped)}
    if tie:
        tie_word_embeddings(renamed, embed_key, head_key)
    return renamed


def vlm_embedding_weights(
    state_dict: dict[str, torch.Tensor],
    keyword: str = "embed_tokens",
    prefixes: tuple[str, ...] = (
        "language_model.model.",
        "language_model.",
    ),
) -> dict[str, torch.Tensor]:
    """Extract embedding weights for a VLM embedding sub-model.

    Filters keys containing *keyword*, then strips the first matching
    prefix from each key.

    This is the standard pattern for VLM embedding sub-models (LLaVA,
    Gemma3, BLIP-2, Mllama) that need ``embed_tokens`` weights with
    ``language_model.model.`` or ``language_model.`` prefixes removed.

    Args:
        state_dict: Full model state dict.
        keyword: Substring that must appear in the key.
        prefixes: Prefixes to strip, tried in order (first match wins).

    Returns:
        New dictionary with embedding weights.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if keyword not in key:
            continue
        new_key = key
        for pfx in prefixes:
            if new_key.startswith(pfx):
                new_key = new_key[len(pfx) :]
                break
        renamed[new_key] = value
    return renamed


def _reshape_packed_qweight(value: torch.Tensor, blob_size: int) -> torch.Tensor:
    """Transpose and reshape a packed qweight tensor for MatMulNBits.

    Converts ``[K_packed, N]`` int32 → ``[N, n_blocks, blob_size]`` uint8.
    """
    transposed = value.t().contiguous()
    n = transposed.shape[0]
    packed = transposed.view(torch.uint8)
    n_blocks = packed.shape[1] // blob_size
    return packed.reshape(n, n_blocks, blob_size)


def _reshape_packed_qzeros(value: torch.Tensor, bits: int, n_blocks: int) -> torch.Tensor:
    """Transpose and unpack packed qzeros for MatMulNBits.

    Converts ``[n_groups_packed, N]`` int32 → ``[N, zp_cols]`` uint8
    where ``zp_cols = ceil(n_blocks * bits / 8)``.  For 4-bit this
    packs two zero-point values per byte, matching ORT's expectation.

    Args:
        value: Packed qzeros tensor ``[n_groups_packed, N]`` int32.
        bits: Quantization bit-width (4 or 8).
        n_blocks: Actual number of quantization blocks (``ceil(K / block_size)``).
    """
    transposed = value.t().contiguous()
    n = transposed.shape[0]
    flat_uint8 = transposed.flatten().view(torch.uint8).reshape(n, -1)
    zp_cols = math.ceil(n_blocks * bits / 8)
    return flat_uint8[:, :zp_cols]


def preprocess_gptq_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Rename, transpose and reshape GPTQ weights for MatMulNBits.

    GPTQ stores quantized weights with these key suffixes:
      - ``*.qweight`` (int32): packed quantized values, shape [K_packed, N]
      - ``*.scales`` (float16): per-group scales, shape [n_groups, N]
      - ``*.qzeros`` (int32): packed zero points, shape [n_groups_packed, N]
      - ``*.g_idx`` (int32): group index (dropped with warning)

    MatMulNBits expects:
      - ``weight``:  [N, n_blocks, blob_size]  uint8
      - ``scales``:  [N, n_blocks]             float
      - ``zero_points``: [N, ceil(n_blocks * bits / 8)]  uint8 (packed)

    where ``n_blocks = K / group_size`` and
    ``blob_size = group_size * bits / 8``.

    Args:
        state_dict: Model state dict with GPTQ keys.
        bits: Quantization bit-width (typically 4).
        group_size: Number of elements per quantization group.

    Returns:
        State dict with renamed, transposed and reshaped weights.
    """
    import logging

    logger = logging.getLogger(__name__)
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.endswith(".g_idx"):
            if not torch.equal(value, torch.arange(value.numel())):
                logger.warning(
                    "Dropping %s — desc_act models with non-trivial "
                    "g_idx may produce incorrect results",
                    key,
                )
            continue

        if key.endswith(".qweight"):
            new_key = key.replace(".qweight", ".weight")
            result[new_key] = _reshape_packed_qweight(value, blob_size)

        elif key.endswith(".qzeros"):
            new_key = key.replace(".qzeros", ".zero_points")
            # Derive n_blocks from the corresponding qweight shape
            qw_key = key.replace(".qzeros", ".qweight")
            if qw_key not in state_dict:
                raise ValueError(
                    f"Missing {qw_key} — qweight must be present alongside qzeros for {key}"
                )
            k = state_dict[qw_key].shape[0] * 32 // bits
            n_blocks = math.ceil(k / group_size)
            result[new_key] = _reshape_packed_qzeros(value, bits, n_blocks)

        elif key.endswith(".scales"):
            result[key] = value.t().contiguous()

        else:
            result[key] = value

    return result


def preprocess_awq_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Rename, transpose and reshape AWQ weights for MatMulNBits.

    AWQ uses the same int32 packing as GPTQ for qweight/qzeros/scales
    but does **not** include ``g_idx``.  The key difference is that AWQ
    zero points are stored with an implicit ``+1`` offset that must be
    subtracted so MatMulNBits receives the correct raw values.

    Args:
        state_dict: Model state dict with AWQ keys.
        bits: Quantization bit-width (typically 4).
        group_size: Number of elements per quantization group.

    Returns:
        State dict with renamed, transposed and reshaped weights.
    """
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.endswith(".qweight"):
            new_key = key.replace(".qweight", ".weight")
            result[new_key] = _reshape_packed_qweight(value, blob_size)

        elif key.endswith(".qzeros"):
            new_key = key.replace(".qzeros", ".zero_points")
            # Derive n_blocks from the corresponding qweight shape
            qw_key = key.replace(".qzeros", ".qweight")
            if qw_key not in state_dict:
                raise ValueError(
                    f"Missing {qw_key} — qweight must be present alongside qzeros for {key}"
                )
            k = state_dict[qw_key].shape[0] * 32 // bits
            n_blocks = math.ceil(k / group_size)
            # AWQ zero points have an implicit +1 offset; subtract it
            # before unpacking so MatMulNBits sees the raw value.
            # For 4-bit, each byte packs TWO nibbles — subtract per-nibble
            # to avoid cross-nibble borrow (e.g. 0x80 - 1 = 0x7F is wrong).
            zp = _reshape_packed_qzeros(value, bits, n_blocks)
            if bits == 4:
                low = (zp & 0x0F).to(torch.int16) - 1
                high = ((zp >> 4) & 0x0F).to(torch.int16) - 1
                low = low.clamp(min=0).to(torch.uint8)
                high = high.clamp(min=0).to(torch.uint8)
                result[new_key] = (high << 4) | low
            else:
                zp_int16 = zp.to(torch.int16) - 1
                result[new_key] = zp_int16.clamp(min=0).to(torch.uint8)

        elif key.endswith(".scales"):
            result[key] = value.t().contiguous()

        else:
            result[key] = value

    return result
