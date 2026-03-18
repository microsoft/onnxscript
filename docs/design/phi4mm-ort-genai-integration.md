# Phi4MM ORT-GenAI Integration Spec

This document consolidates the ORT-GenAI integration plan for Phi4MM: the
official reference config, I/O contract comparison, GenaiConfigGenerator
extension design, target configs, and implementation plan.

---

## 1. Official Reference Config

**Source:** https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx/tree/main/gpu/gpu-int4-rtn-block-32

The official ONNX export uses a **4-model split**:

```json
{
    "model": {
        "type": "phi4mm",
        "decoder": {
            "filename": "phi-4-mm-text.onnx",
            "inputs": {
                "inputs_embeds": "inputs_embeds",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            }
        },
        "vision": {
            "filename": "phi-4-mm-vision.onnx",
            "config_filename": "vision_processor.json",
            "adapter_filename": "phi-4-mm-vision.onnx_adapter",
            "inputs": {
                "pixel_values": "pixel_values",
                "attention_mask": "image_attention_mask",
                "image_sizes": "image_sizes"
            },
            "outputs": {
                "image_features": "image_features"
            }
        },
        "speech": {
            "filename": "phi-4-mm-speech.onnx",
            "config_filename": "speech_processor.json",
            "adapter_filename": "phi-4-mm-speech.onnx_adapter",
            "inputs": {
                "audio_embeds": "audio_embeds",
                "attention_mask": "audio_attention_mask",
                "audio_sizes": "audio_sizes",
                "audio_projection_mode": "audio_projection_mode"
            },
            "outputs": {
                "audio_features": "audio_features"
            }
        },
        "embedding": {
            "filename": "phi-4-mm-embedding.onnx",
            "inputs": {
                "input_ids": "input_ids",
                "image_features": "image_features",
                "audio_features": "audio_features"
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds"
            }
        }
    }
}
```

### Key Observations

1. **Speech model has `audio_projection_mode` input** — integer selector
   between speech and vision projection branches. More efficient than dual
   outputs. Values likely: 0 = speech-only, 1 = combined vision+audio.

2. **Vision model has additional inputs** — `image_attention_mask` and
   `image_sizes` for HD dynamic resolution handling.

3. **Speech model input is `audio_embeds`** (not `audio_features`), with
   `audio_sizes` and `audio_attention_mask` inputs.

4. **LoRA adapters are separate files** — `phi-4-mm-vision.onnx_adapter`
   and `phi-4-mm-speech.onnx_adapter`. Our design bakes LoRA into the
   decoder as a deliberate simplification.

5. **Decoder uses `inputs_embeds`** (not `input_ids`), confirming the
   4-model split where the embedding model handles token lookup.

6. **No `position_ids` in decoder config** — may be computed internally.

### Processor Configs (Official)

- **vision_processor.json**: DecodeImage → Phi4VisionDynamicPreprocess
  (dynamic_hd=36, base=448) → Rescale → Normalize(mean=0.5, std=0.5) →
  Phi4VisionProcessor
- **speech_processor.json**: AudioDecoderEx (8kHz + 16kHz) → Phi4AudioEmbed
  (audio_compression_rate=8, n_mel=80, various STFT params)

---

## 2. I/O Contract Comparison

### Official vs Our Implementation

#### Decoder

| Dir | Official | Ours | Status |
|-----|----------|------|--------|
| In | `inputs_embeds` | `inputs_embeds` | ✅ Match |
| In | `attention_mask` | `attention_mask` | ✅ Match |
| In | — | `position_ids [B, S]` | ⚠️ We add this (codebase convention) |
| In | `past_key_values.%d.key/value` | `past_key_values.{i}.key/value` | ✅ Match |
| Out | `logits` | `logits` | ✅ Match |
| Out | `present.%d.key/value` | `present.{i}.key/value` | ✅ Match |

#### Vision

| Dir | Official | Ours | Status |
|-----|----------|------|--------|
| In | `pixel_values` | `pixel_values` | ✅ Match |
| In | `image_attention_mask` | — | ❌ Missing |
| In | `image_sizes` | — | ❌ Missing |
| Out | `image_features` | `image_features` | ✅ Match |

#### Speech

| Dir | Official | Ours | Status |
|-----|----------|------|--------|
| In | `audio_embeds` | `audio_features` | ⚠️ Name differs |
| In | `audio_attention_mask` | — | ❌ Missing |
| In | `audio_sizes` | — | ❌ Missing |
| In | `audio_projection_mode` | — | ❌ We use dual outputs |
| Out | `audio_features` (single) | `speech_features` + `speech_features_for_vision` | ⚠️ Dual outputs |

#### Embedding

| Dir | Official | Ours | Status |
|-----|----------|------|--------|
| In | `input_ids` | `input_ids` | ✅ Match |
| In | `image_features` | `image_features` | ✅ Match |
| In | `audio_features` | `speech_features` | ⚠️ Name differs |
| Out | `inputs_embeds` | `inputs_embeds` | ✅ Match |

### Codebase Comparison: How Other Models Handle Similar Inputs

| Model | Vision Extra Inputs | Audio Extra Inputs | Decoder position_ids |
|-------|--------------------|--------------------|---------------------|
| **Gemma3** | None | N/A | 2D `[B, S]` |
| **Qwen2.5-VL** | `image_grid_thw [N, 3]` | N/A | 3D `[3, B, S]` MRoPE |
| **Qwen3-ASR** | N/A | None | 3D `[3, B, S]` MRoPE |
| **Official Phi4MM** | `image_attention_mask`, `image_sizes` | `audio_attention_mask`, `audio_sizes`, `audio_projection_mode` | None |
| **Our Phi4MM** | None | None | 2D `[B, S]` |

### Gap Analysis

**GAP 1: `image_sizes` / `image_attention_mask` on Vision**
Needed for HD dynamic crop with variable sub-images per input. Our vision
forward only takes `pixel_values`. Works for fixed-size test inputs.
**Decision:** DEFER to follow-up PR. Document as known limitation.

**GAP 2: `audio_sizes` / `audio_attention_mask` on Speech**
Needed for variable-length audio with batched sequences.
**Decision:** DEFER to follow-up PR. Same rationale.

**GAP 3: `audio_projection_mode` vs dual outputs**
Official uses integer input to select branch (single output). Ours runs
both branches (two outputs). Dual outputs are simpler (no conditional ops)
but slightly wasteful.
**Decision:** Keep dual outputs. Switch to mode input if ORT-GenAI requires.

**GAP 4: `position_ids` on decoder**
Official has no position_ids. All other models in our codebase use explicit
position_ids.
**Decision:** KEEP position_ids. The genai_config handles the mapping.

**GAP 5: Input naming differences**
Cosmetic differences (`audio_features` vs `audio_embeds`, `speech_features`
vs `audio_features`). Handled by genai_config input/output name mapping.
**Decision:** Keep our names for codebase consistency.

### Intentional Divergences Summary

| Aspect | Official | Ours | Reason |
|--------|----------|------|--------|
| `position_ids` | Not in decoder | Explicit input | Codebase convention |
| Projection branch | `audio_projection_mode` input | Dual outputs | Simpler graph |
| LoRA | Separate adapter files | Baked into decoder | Single file per model |
| HD inputs | `image_sizes`, masks | Not yet | Deferred |
| Audio metadata | `audio_sizes`, masks | Not yet | Deferred |

---

## 3. GenaiConfigGenerator Extension

### Problem Statement

`GenaiConfigGenerator` supports LLM (decoder-only) and VLM (vision +
embedding + decoder via `with_vision()`). Phi4MM requires a 4-model
multimodal config with vision, speech, embedding, and decoder sections.

### Design: `with_speech()` + extended `with_vision()`

Follow the same builder pattern as `with_vision()`. Keep the generator
model-agnostic so future audio/multimodal models reuse it.

#### New `with_speech()` method

```python
def with_speech(
    self,
    *,
    audio_token_id: int | None = None,
    filename: str = "speech/model.onnx",
    config_filename: str = "speech_processor.json",
    input_names: dict[str, str] | None = None,
    output_names: dict[str, str] | None = None,
) -> GenaiConfigGenerator:
    """Add speech/audio model section for multimodal models."""
    if input_names is None:
        input_names = {
            "audio_embeds": "audio_embeds",
            "attention_mask": "audio_attention_mask",
            "audio_sizes": "audio_sizes",
            "audio_projection_mode": "audio_projection_mode",
        }
    if output_names is None:
        output_names = {"audio_features": "audio_features"}

    self._speech = {
        "filename": filename,
        "config_filename": config_filename,
        "inputs": input_names,
        "outputs": output_names,
    }
    if audio_token_id is not None:
        self._vlm_token_ids["audio_token_id"] = audio_token_id
    return self
```

#### Extended `with_vision()` — optional overrides

Add optional `input_names`/`output_names`/`config_filename` parameters.
If not provided, existing defaults apply. Zero breakage for current callers.

```python
def with_vision(
    self,
    *,
    image_token_id: int,
    filename: str = "vision/model.onnx",
    embedding_filename: str = "embedding/model.onnx",
    spatial_merge_size: int = 2,
    config_filename: str = "processor_config.json",
    input_names: dict[str, str] | None = None,   # NEW
    output_names: dict[str, str] | None = None,   # NEW
    embedding_input_names: dict[str, str] | None = None,  # NEW
    embedding_output_names: dict[str, str] | None = None,  # NEW
) -> GenaiConfigGenerator:
```

#### Updated `generate()` method

```python
def generate(self) -> dict[str, Any]:
    # ... existing code ...
    if self._vision is not None:
        model["vision"] = self._vision
    if self._embedding is not None:
        model["embedding"] = self._embedding
    if self._speech is not None:
        model["speech"] = self._speech
    model.update(self._vlm_token_ids)
    return {"model": model, "search": _default_search_params()}
```

#### Embedding construction for multimodal

Defer embedding construction to `generate()` based on enabled modalities:

```python
if self._vision is not None or self._speech is not None:
    emb_inputs = {"input_ids": "input_ids"}
    if self._vision is not None:
        emb_inputs["image_features"] = "image_features"
    if self._speech is not None:
        emb_inputs["audio_features"] = "audio_features"
    model["embedding"] = {
        "filename": self._embedding_filename or "embedding/model.onnx",
        "inputs": emb_inputs,
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
```

#### Decoder inputs for multimodal

Multimodal decoders use `inputs_embeds` (like VLMs):

```python
if is_vlm or is_multimodal:
    inputs["inputs_embeds"] = "inputs_embeds"
else:
    inputs["input_ids"] = "input_ids"
```

### Differences from Existing VLM Pattern

| Aspect | VLM (Qwen2.5-VL) | Phi4MM Multimodal |
|--------|-------------------|-------------------|
| Model sections | decoder, vision, embedding | decoder, vision, **speech**, embedding |
| Vision inputs | pixel_values, image_grid_thw | pixel_values, **image_attention_mask**, **image_sizes** |
| Vision extras | spatial_merge_size | config_filename: vision_processor.json |
| Speech section | N/A | filename, config_filename, inputs (4), outputs (1) |
| Embedding inputs | input_ids, image_features | input_ids, image_features, **audio_features** |
| Processor configs | processor_config.json (unified) | **vision_processor.json** + **speech_processor.json** |
| search.past_present_share_buffer | false | **true** |

**Backward compatibility**: Zero breaking changes. All new parameters are
optional with defaults matching current behavior.

### Detection and Trigger Logic (auto_export.py)

```python
# Current:
is_vlm = "vision" in pkg and "embedding" in pkg

# Extended:
is_vlm = "vision" in pkg and "embedding" in pkg
is_multimodal = is_vlm and "speech" in pkg

# Add to _ORT_GENAI_MODEL_TYPE:
"phi4mm": "phi4mm",
"phi4_multimodal": "phi4mm",
```

---

## 4. Target genai_config.json

The complete target config for Phi4MM, mapping between ORT-GenAI expected
names (left side) and ONNX tensor names (right side):

```json
{
    "model": {
        "bos_token_id": 199999,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            },
            "filename": "model/model.onnx",
            "head_size": 128,
            "hidden_size": 3072,
            "inputs": {
                "inputs_embeds": "inputs_embeds",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            },
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8
        },
        "vision": {
            "filename": "vision/model.onnx",
            "config_filename": "vision_processor.json",
            "inputs": {
                "pixel_values": "pixel_values",
                "attention_mask": "image_attention_mask",
                "image_sizes": "image_sizes"
            },
            "outputs": {
                "image_features": "image_features"
            }
        },
        "speech": {
            "filename": "speech/model.onnx",
            "config_filename": "speech_processor.json",
            "inputs": {
                "audio_embeds": "audio_embeds",
                "attention_mask": "audio_attention_mask",
                "audio_sizes": "audio_sizes",
                "audio_projection_mode": "audio_projection_mode"
            },
            "outputs": {
                "audio_features": "audio_features"
            }
        },
        "embedding": {
            "filename": "embedding/model.onnx",
            "inputs": {
                "input_ids": "input_ids",
                "image_features": "image_features",
                "audio_features": "audio_features"
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds"
            }
        },
        "eos_token_id": [200020, 199999],
        "pad_token_id": 199999,
        "type": "phi4mm",
        "vocab_size": 200064
    },
    "search": {
        "diversity_penalty": 0.0,
        "do_sample": false,
        "early_stopping": true,
        "length_penalty": 1.0,
        "max_length": 131072,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": true,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0
    }
}
```

---

## 5. Processor Configs

### vision_processor.json

```json
{
    "processor": {
        "name": "phi_4_vision_processor",
        "transforms": [
            {
                "operation": {
                    "name": "decode_image",
                    "type": "DecodeImage",
                    "attrs": { "color_space": "RGB" }
                }
            },
            {
                "operation": {
                    "name": "phi4_vision_dynamic_preprocess",
                    "type": "Phi4VisionDynamicPreprocess",
                    "attrs": {
                        "dynamic_hd": 36,
                        "dyhd_base_resolution": 448
                    }
                }
            },
            {
                "operation": {
                    "name": "rescale",
                    "type": "Rescale",
                    "inputs": [":0"]
                }
            },
            {
                "operation": {
                    "name": "normalize",
                    "type": "Normalize",
                    "attrs": {
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.5, 0.5, 0.5]
                    }
                }
            },
            {
                "operation": {
                    "name": "phi4_vision_processor",
                    "type": "Phi4VisionProcessor",
                    "inputs": [":0", "phi4_vision_dynamic_preprocess:1"],
                    "attrs": {
                        "dyhd_base_resolution": 448,
                        "interpolation": "CUBIC"
                    }
                }
            }
        ],
        "output_aligner": "phi4-vision-aligner"
    }
}
```

**Key parameters**: `dynamic_hd`: 36 (max HD crops), `dyhd_base_resolution`:
448, normalization mean/std=[0.5, 0.5, 0.5] (SigLIP standard).

### speech_processor.json

```json
{
    "feature_extraction": {
        "sequence": [
            {
                "operation": {
                    "name": "audio_decoder",
                    "type": "AudioDecoderEx",
                    "attrs": {
                        "target_sample_rates": [8000, 16000]
                    }
                }
            },
            {
                "operation": {
                    "name": "phi_4_audio_embed",
                    "type": "Phi4AudioEmbed",
                    "attrs": {
                        "audio_compression_rate": 8,
                        "stft_normal/n_fft": 512,
                        "stft_normal/frame_length": 400,
                        "stft_normal/hop_length": 160,
                        "stft_normal/win_fn": "hamming",
                        "logmel/chunk_size": 30,
                        "logmel/hop_length": 160,
                        "logmel/n_fft": 512,
                        "logmel/n_mel": 80,
                        "logmel/feature_first": 0,
                        "logmel/no_padding": 1,
                        "stft_normal_8k/n_fft": 256,
                        "stft_normal_8k/frame_length": 200,
                        "stft_normal_8k/hop_length": 80,
                        "stft_normal_8k/win_fn": "hamming",
                        "logmel_8k/chunk_size": 30,
                        "logmel_8k/hop_length": 80,
                        "logmel_8k/n_fft": 512,
                        "logmel_8k/n_mel": 80,
                        "logmel_8k/feature_first": 0,
                        "logmel_8k/no_padding": 1
                    }
                }
            }
        ],
        "output_aligner": "phi4-audio-aligner"
    }
}
```

**Key parameters**: `audio_compression_rate`: 8, `n_mel`: 80, supports
8kHz and 16kHz sample rates, hamming window STFT.

**Generation approach**: Static templates in examples (recommended).
Auto-generation via `auto_export.py` is a follow-up.

---

## 6. Implementation Plan

### Phase 1: Example with static config (this PR)

- Static genai_config dict in `phi4mm_ort_genai.py` example
- Static `_write_vision_processor_config()` in example
- Static `_write_speech_processor_config()` in example

### Phase 2: GenaiConfigGenerator extension (follow-up)

**genai_config.py changes:**
- Add `self._speech: dict | None = None` to `__init__()`
- Add `with_speech()` method
- Extend `with_vision()` with optional I/O name overrides
- Update `generate()` to include speech section
- Handle embedding construction with audio_features

**genai_config_test.py changes:**
- Test `with_speech()` produces speech section
- Test `with_vision().with_speech()` chaining produces all 4 sections
- Test embedding inputs include audio_features when speech enabled
- Test backward compat: existing VLM tests still pass
- Test Phi4MM config matches official reference

### Phase 3: auto_export.py integration (follow-up)

- Add `phi4mm` / `phi4_multimodal` to `_ORT_GENAI_MODEL_TYPE`
- Extend multimodal detection
- Call `with_speech()` when speech model detected in package
- Extend `_write_processor_config()` for Phi4MM

### Deferred Items

1. **I/O name alignment with official export** — `audio_embeds` vs
   `audio_features`, missing metadata inputs, `position_ids` in decoder
2. **LoRA adapter files** — official uses separate `.onnx_adapter` files;
   our approach bakes LoRA in
3. **`past_present_share_buffer: true`** — official uses this for KV cache
   efficiency; our default is false
4. **Additional search params** — `diversity_penalty`, `length_penalty`,
   `no_repeat_ngram_size` (all at no-op values, safe to omit)
