# Phi4MM Four-Model Split Design

**Author:** Architect Agent 6789f954
**Status:** FINAL v3 — lead-approved, ready for implementation
**Updated:** 2026-03-09

---

## Problem Statement

The current `Phi4MMMultiModalModel` uses `MultiModalTask` to produce a
**single unified ONNX model**. The user directive requires a **four-model
split**:

```
pkg["vision"]     — SigLIP vision encoder + HD transform + projection MLP
pkg["speech"]     — Conformer speech encoder + both projection branches
pkg["embedding"]  — Token embedding + InputMixer fusion (lightweight)
pkg["model"]      — Text decoder with LoRA baked in + lm_head
```

---

## Approved Design Decisions

1. **HD transform + projection in vision model** — vision outputs
   pre-projected features in text dimension (3072)
2. **Both speech projection branches in speech model** — outputs
   `speech_features` (speech branch) and `speech_features_for_vision`
   (vision branch for combined input)
3. **Embedding model is lightweight** — just token embed + InputMixers,
   receives pre-projected features
4. **LoRA baked into decoder** — both vision (r=256) and speech (r=320)
   adapters always active via LoRALinear sum
5. **Zero-length tensors for absent modalities** — `[0, 3072]`
6. **Keep existing MultiModalTask** — new Phi4MMMultiModalTask alongside

---

## Data Flow

```
                    pixel_values
                         │
                    ┌────▼────┐
                    │ vision  │  SigLIP + HD transform (glb_GN/sub_GN)
                    │         │  + projection MLP (1152 → 3072)
                    └────┬────┘
                         │ image_features [num_img_tokens, 3072]
                         │
   audio_features        │
        │                │
   ┌────▼────┐           │
   │ speech  │  Conformer encoder
   │         │  + both projection MLPs (1024 → 3072)
   └────┬────┘
        │ speech_features [num_audio_tokens, 3072]
        │ speech_features_for_vision [num_audio_tokens, 3072]
        │                │
        │   input_ids    │
        │       │        │
        ▼       ▼        ▼
   ┌─────────────────────────┐
   │       embedding         │  embed_tokens(input_ids) → text_embeds
   │                         │  + InputMixer(image) + InputMixer(audio)
   └────────────┬────────────┘
                │ inputs_embeds [batch, seq_len, 3072]
                │
   ┌────────────▼────────────┐
   │         model           │  LoRA decoder layers + RMSNorm + lm_head
   │                         │  + RoPE + KV cache
   └────────────┬────────────┘
                │ logits [batch, seq_len, vocab_size]
```

---

## I/O Contracts

### 1. Vision (`pkg["vision"]`)

| Direction | Name | Shape | DType |
|-----------|------|-------|-------|
| Input | `pixel_values` | `[batch, 3, 448, 448]` | FLOAT |
| Output | `image_features` | `[num_image_tokens, 3072]` | FLOAT |

**Contains:** `_Phi4MMSigLIPEncoder` + `_Phi4MMProjectionMLP(1152→3072)`
+ `glb_GN` + `sub_GN` (HD spatial merge params)

### 2. Speech (`pkg["speech"]`)

| Direction | Name | Shape | DType |
|-----------|------|-------|-------|
| Input | `audio_features` | `[batch, audio_seq_len, 128]` | FLOAT |
| Output | `speech_features` | `[num_audio_tokens, 3072]` | FLOAT |
| Output | `speech_features_for_vision` | `[num_audio_tokens, 3072]` | FLOAT |

**Contains:** `ConformerEncoder` + `audio_projection.speech` MLP +
`audio_projection.vision` MLP

**Runtime:** Use `speech_features` for audio-only input. Use
`speech_features_for_vision` when both vision and audio are present.

### 3. Embedding (`pkg["embedding"]`)

| Direction | Name | Shape | DType | Notes |
|-----------|------|-------|-------|-------|
| Input | `input_ids` | `[batch, seq_len]` | INT64 | |
| Input | `image_features` | `[num_img_tokens, 3072]` | FLOAT | `[0, 3072]` if absent |
| Input | `speech_features` | `[num_audio_tokens, 3072]` | FLOAT | `[0, 3072]` if absent |
| Output | `inputs_embeds` | `[batch, seq_len, 3072]` | FLOAT | |

**Contains:** `Embedding` + `InputMixer(200010)` + `InputMixer(200011)`

### 4. Decoder (`pkg["model"]`)

| Direction | Name | Shape | DType |
|-----------|------|-------|-------|
| Input | `inputs_embeds` | `[batch, seq_len, 3072]` | FLOAT |
| Input | `attention_mask` | `[batch, past_seq+seq]` | INT64 |
| Input | `position_ids` | `[batch, seq_len]` | INT64 |
| Input | `past_key_values.{i}.key` | `[batch, 8, past_seq, 128]` | FLOAT |
| Input | `past_key_values.{i}.value` | `[batch, 8, past_seq, 128]` | FLOAT |
| Output | `logits` | `[batch, seq_len, vocab_size]` | FLOAT |
| Output | `present.{i}.key` | `[batch, 8, total_seq, 128]` | FLOAT |
| Output | `present.{i}.value` | `[batch, 8, total_seq, 128]` | FLOAT |

**Contains:** `DecoderLayer(LoRALinear)` × N + `RMSNorm` + `LongRoPE` +
`lm_head`

---

## Module Tree

```
Phi4MMMultiModalModel
  ├── vision_encoder: _Phi4MMVisionModel (NEW)
  │     ├── img_processor: _Phi4MMSigLIPEncoder
  │     ├── img_projection: _Phi4MMProjectionMLP
  │     ├── glb_GN: Parameter
  │     └── sub_GN: Parameter
  │
  ├── speech_encoder: _Phi4MMSpeechModel (NEW)
  │     ├── encoder: ConformerEncoder
  │     ├── audio_projection.speech: _Phi4MMProjectionMLP
  │     └── audio_projection.vision: _Phi4MMProjectionMLP
  │
  ├── embedding: _Phi4MMEmbeddingModel (NEW)
  │     ├── embed_tokens: Embedding
  │     ├── _image_mixer: InputMixer(200010)
  │     └── _audio_mixer: InputMixer(200011)
  │
  └── decoder: _Phi4MMDecoderModel (NEW)
        ├── layers: ModuleList[DecoderLayer(LoRA)]
        ├── norm: RMSNorm
        ├── rotary_emb: LongRoPE
        └── lm_head: Linear
```

Top-level `forward()` raises `NotImplementedError` (Gemma3 pattern).

---

## Task: `Phi4MMMultiModalTask`

**File:** `src/mobius/tasks/_phi4mm_multimodal.py`

```python
class Phi4MMMultiModalTask(ModelTask):
    """4-model split for Phi4MM."""

    def build(self, module, config) -> ModelPackage:
        return ModelPackage({
            "vision": self._build_vision(module.vision_encoder, config),
            "speech": self._build_speech(module.speech_encoder, config),
            "embedding": self._build_embedding(module.embedding, config),
            "model": self._build_decoder(module.decoder, config),
        }, config=config)
```

Registry:
```python
reg.register("phi4mm", Phi4MMMultiModalModel, task="phi4mm-multimodal")
reg.register("phi4_multimodal", Phi4MMMultiModalModel, task="phi4mm-multimodal")
```

---

## Weight Routing

After `_preprocess_phi4mm_weights()` (LoRA unwrapping + fused splits),
`preprocess_weights()` remaps prefixes:

| HF Key Prefix | → ONNX Prefix | Target |
|---------------|---------------|--------|
| `model.embed_tokens_extend.image_embed.img_processor.*` | `vision_encoder.img_processor.*` | vision |
| `model.embed_tokens_extend.image_embed.img_projection.*` | `vision_encoder.img_projection.*` | vision |
| `model.embed_tokens_extend.image_embed.glb_GN` | `vision_encoder.glb_GN` | vision |
| `model.embed_tokens_extend.image_embed.sub_GN` | `vision_encoder.sub_GN` | vision |
| `model.embed_tokens_extend.audio_embed.encoder.*` | `speech_encoder.encoder.*` | speech |
| `model.embed_tokens_extend.audio_embed.audio_projection.*` | `speech_encoder.audio_projection.*` | speech |
| `model.embed_tokens.weight` | `embedding.embed_tokens.weight` | embedding |
| `model.layers.*` | `decoder.layers.*` | model |
| `model.norm.*` | `decoder.norm.*` | model |
| `lm_head.*` | `decoder.lm_head.*` | model |

Weight application: iterate all models, each matches by initializer name.
Position embedding: squeeze 3D→2D (already committed).
Weight tying: copy embed_tokens.weight → lm_head.weight if tie_word_embeddings.

---

## Decode Step Behavior

- **Prefill:** Run all 4 models (vision → speech → embedding → decoder)
- **Decode:** Only embedding + decoder. Pass `[0, 3072]` for absent
  features. Embedding just does token lookup (InputMixers are no-ops).
- Vision and speech models are NOT called on decode steps.

---

## Implementation Order

1. Create 4 new sub-module classes in `models/phi.py`
2. Refactor `Phi4MMMultiModalModel.__init__()` to compose them
3. Update `preprocess_weights()` with prefix remapping
4. Create `Phi4MMMultiModalTask` in `tasks/_phi4mm_multimodal.py`
5. Register in `tasks/__init__.py` and `_registry.py`
6. Update unit tests in `build_graph_test.py`
7. Update integration tests in `phi4mm_integration_test.py`
8. Create e2e example in `examples/`
