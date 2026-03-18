---
name: ort-genai-config
description: >
  Complete reference for the onnxruntime-genai genai_config.json format,
  processor_config.json format, model type registry, and the MultiModal
  pipeline architecture. Use this skill when generating genai_config.json,
  processor_config.json, debugging ORT GenAI model loading, or integrating
  exported ONNX models with the onnxruntime-genai runtime.
---

# Skill: ORT GenAI Config Format

## When to use

Use this skill when:

- Writing `genai_config.json` for a new model export
- Writing `processor_config.json` for image/audio preprocessing
- Debugging ORT GenAI model loading errors (protobuf parsing, missing keys)
- Understanding how the ORT GenAI pipeline feeds inputs to vision, embedding,
  and decoder models
- Adding support for a new model type in ORT GenAI

## Overview

onnxruntime-genai loads models from a directory containing:

```
model_dir/
├── genai_config.json          # Required — model config + search params
├── model.onnx                 # Decoder model
├── model.onnx.data            # External weights (optional)
├── vision.onnx                # Vision encoder (multimodal only)
├── embedding.onnx             # Embedding model (multimodal only)
├── tokenizer.json             # Tokenizer (HuggingFace format)
├── tokenizer_config.json      # Tokenizer config
├── chat_template.jinja        # Chat template (optional)
└── processor_config.json      # Image processor (multimodal only)
```

## genai_config.json — Top-level structure

```json
{
  "model": { ... },
  "search": { ... },
  "engine": { ... }
}
```

The `engine` section is optional and only used for batched serving.

---

## model section

### Model-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | **yes** | Model type identifier (see registry below) |
| `vocab_size` | int | yes | Vocabulary size |
| `context_length` | int | **yes** | Maximum context length; must be > 0 |
| `bos_token_id` | int | no | Beginning-of-sequence token |
| `eos_token_id` | int \| int[] | no | End-of-sequence token(s); defaults to `pad_token_id` |
| `pad_token_id` | int | no | Padding token |
| `sep_token_id` | int | no | Separator token |
| `decoder_start_token_id` | int | no | Decoder start token (encoder-decoder models) |
| `image_token_id` | int | VLM | Token ID for image placeholders (e.g. 151655 for Qwen2.5-VL). **Required** for 3D M-RoPE position ID computation. |
| `video_token_id` | int | no | Token ID for video placeholders (e.g. 151656) |
| `vision_start_token_id` | int | VLM | Token ID for `<\|vision_start\|>` (e.g. 151652). Used to locate image/video regions in input_ids. |

### Model type registry

**LLM** (decoder-only, maps to `DecoderOnly_Model`):

```
chatglm, decoder, ernie4_5, gemma, gemma2, gemma3_text, gpt2,
gptoss, granite, internlm2, llama, mistral, nemotron, olmo,
phi, phimoe, phi3, phi3small, qwen2, qwen3, smollm3
```

> `gpt2` has a special code path (`Gpt_Model`) but is also in the LLM list.

**VLM** (vision-language, maps to `MultiModalLanguageModel`):

```
fara, gemma3, phi3v, qwen2_5_vl
```

**MMM** (multi-modal with vision + audio, maps to `MultiModalLanguageModel`):

```
phi4mm
```

**ALM** (audio-language, maps to `WhisperModel`):

```
whisper
```

**Pipeline models** (maps to `DecoderOnlyPipelineModel`):

```
phi3small_pipeline, qwen2_5_vl_pipeline
```

**Special handling:**

- `fara` and `qwen2_5_vl` with non-empty `model.decoder.pipeline` →
  `Qwen2_5_VL_PipelineModel`
- `IsQwen25VL()` check (type == `"fara"` or `"qwen2_5_vl"`) enables 3D
  MRoPE position ID handling

### Multimodal processor factory

When `model.create_multimodal_processor()` is called:

| model.type | Processor class |
|---|---|
| `phi3v` | PhiImageProcessor |
| `whisper` | WhisperProcessor |
| `phi4mm` | PhiMultiModalProcessor |
| `gemma3` | GemmaImageProcessor |
| `fara` | QwenImageProcessor |
| `qwen2_5_vl` | QwenImageProcessor |

> Models not in this table cannot use `create_multimodal_processor()`.

---

## model.decoder

The decoder (text model) configuration.

### Core fields

| Field | Type | Required | Description |
|---|---|---|---|
| `filename` | string | **yes** | ONNX model filename (e.g. `"model.onnx"`) |
| `hidden_size` | int | yes | Hidden dimension |
| `head_size` | int | yes | Size per attention head |
| `num_attention_heads` | int | yes | Number of query attention heads |
| `num_key_value_heads` | int | yes | Number of KV heads (for GQA) |
| `num_hidden_layers` | int | yes | Number of transformer layers |
| `session_options` | object | no | ORT session configuration |
| `run_options` | object | no | ORT run options |

### Decoder inputs

```json
"inputs": {
  "input_ids": "input_ids",
  "inputs_embeds": "inputs_embeds",
  "attention_mask": "attention_mask",
  "position_ids": "position_ids",
  "past_key_names": "past_key_values.%d.key",
  "past_value_names": "past_key_values.%d.value"
}
```

The `%d` in `past_key_names` / `past_value_names` is replaced with the layer
index (0 to num_hidden_layers-1) at load time.

Additional optional inputs for advanced scenarios:

```json
"past_names": "",
"cross_past_key_names": "",
"cross_past_value_names": "",
"past_key_values_length": "past_key_values_length",
"past_sequence_length": "past_sequence_length",
"current_sequence_length": "current_sequence_length",
"total_sequence_length": "total_sequence_length",
"cache_indirection": "cache_indirection",
"encoder_hidden_states": "encoder_hidden_states",
"encoder_attention_mask": "encoder_attention_mask",
"cumulative_sequence_lengths": "cumulative_sequence_lengths",
"past_sequence_lengths": "past_sequence_lengths",
"block_table": "block_table"
```

### Decoder outputs

```json
"outputs": {
  "logits": "logits",
  "present_key_names": "present.%d.key",
  "present_value_names": "present.%d.value"
}
```

### Sliding window (optional)

```json
"sliding_window": {
  "window_size": 4096,
  "pad_value": -1,
  "alignment": "right",
  "slide_key_value_cache": true,
  "slide_inputs": true,
  "layers": [0, 2, 4]
}
```

---

## model.embedding

Required for VLM and MMM models. Merges text token embeddings with vision/audio
features.

```json
"embedding": {
  "filename": "embedding.onnx",
  "inputs": {
    "input_ids": "input_ids",
    "image_features": "image_features",
    "audio_features": "audio_features"
  },
  "outputs": {
    "inputs_embeds": "inputs_embeds"
  }
}
```

---

## model.vision

Required for VLM and MMM models.

| Field | Type | Default | Description |
|---|---|---|---|
| `filename` | string | — | Vision ONNX model |
| `config_filename` | string | `"processor_config.json"` | Processor config file |
| `adapter_filename` | string | — | Optional adapter model |
| `spatial_merge_size` | int | 2 | **Required for Qwen2.5-VL.** Controls how many vision patches are merged into one token. Used to compute grid dimensions for 3D M-RoPE position IDs (h/merge × w/merge). |
| `tokens_per_second` | float | 2.0 | Video tokens/second |

### Vision inputs

```json
"inputs": {
  "pixel_values": "pixel_values",
  "image_sizes": "image_sizes",
  "image_grid_thw": "image_grid_thw",
  "attention_mask": "image_attention_mask"
}
```

### Vision outputs

```json
"outputs": {
  "image_features": "image_features"
}
```

### Vision pipeline (optional)

For models that split vision into stages (e.g. patch_embed → attention →
merger):

```json
"pipeline": [
  {
    "filename": "patch_embed.onnx",
    "model_id": "patch_embed",
    "inputs": ["pixel_values"],
    "outputs": ["patch_embeddings"],
    "run_on_cpu": false,
    "session_options": {}
  }
]
```

---

## model.speech

For audio-language models (whisper, phi4mm).

```json
"speech": {
  "filename": "speech.onnx",
  "config_filename": "audio_processor_config.json",
  "inputs": {
    "audio_embeds": "audio_embeds",
    "attention_mask": "audio_attention_mask",
    "audio_sizes": "audio_sizes",
    "audio_projection_mode": "audio_projection_mode"
  },
  "outputs": {
    "audio_features": "audio_features"
  }
}
```

---

## model.encoder

For encoder-decoder models (whisper).

```json
"encoder": {
  "filename": "encoder.onnx",
  "hidden_size": 1280,
  "num_attention_heads": 20,
  "num_hidden_layers": 32,
  "head_size": 64,
  "inputs": {
    "input_ids": "input_ids",
    "attention_mask": "attention_mask"
  },
  "outputs": {
    "encoder_hidden_states": "encoder_hidden_states"
  }
}
```

---

## search section

Controls generation behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `do_sample` | bool | false | Sampling vs greedy |
| `min_length` | int | 0 | Minimum output length |
| `max_length` | int | context_length | Maximum total length (prompt + output) |
| `batch_size` | int | 1 | Batch size |
| `num_beams` | int | 1 | Beam width (1 = greedy) |
| `num_return_sequences` | int | 1 | Sequences to return |
| `top_k` | int | 50 | Top-K sampling |
| `top_p` | float | 0.0 | Nucleus sampling |
| `temperature` | float | 1.0 | Sampling temperature |
| `repetition_penalty` | float | 1.0 | Repetition penalty (1.0 = none) |
| `length_penalty` | float | 1.0 | Beam search length penalty |
| `early_stopping` | bool | true | Stop beam search early |
| `past_present_share_buffer` | bool | false | Share KV cache buffer (CUDA) |
| `random_seed` | int | -1 | RNG seed (-1 = random) |
| `chunk_size` | int | — | Prefill chunking size |

### Minimal search config

```json
"search": {
  "do_sample": false,
  "max_length": 4096,
  "num_beams": 1,
  "past_present_share_buffer": false
}
```

---

## engine section (optional)

For batched serving.

```json
"engine": {
  "dynamic_batching": {
    "block_size": 256,
    "num_blocks": 16,
    "gpu_utilization_factor": 0.9,
    "max_batch_size": 16
  }
}
```

Or static batching:

```json
"engine": {
  "static_batching": {
    "max_batch_size": 4
  }
}
```

Dynamic and static batching are mutually exclusive.

---

## session_options

Nested inside `decoder`, `encoder`, `vision`, `speech`, or `embedding`.

```json
"session_options": {
  "intra_op_num_threads": 8,
  "inter_op_num_threads": 1,
  "log_id": "onnxruntime-genai",
  "log_severity_level": 2,
  "enable_cpu_mem_arena": true,
  "enable_mem_pattern": true,
  "enable_profiling": "profile.json",
  "graph_optimization_level": "ORT_ENABLE_EXTENDED",
  "provider_options": [
    {
      "cuda": {
        "device_id": "0"
      }
    }
  ]
}
```

Graph optimization levels: `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`,
`ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`.

Provider names are normalized: `"qnn"` → `"QNN"`, `"dml"` → `"DML"`,
`"webgpu"` → `"WebGPU"`, `"openvino"` → `"OpenVINO"`.

---

## processor_config.json (ort-extensions format)

> **Critical:** ORT GenAI expects the ort-extensions format — NOT the
> HuggingFace `processor_config.json` format. The HF format wraps data under
> `"image_processor"` with different keys; ORT extensions expects a `"processor"`
> key with an ordered transform pipeline.

### Qwen2.5-VL example

```json
{
  "processor": {
    "name": "qwen2_5_image_processor",
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
          "name": "convert_to_rgb",
          "type": "ConvertRGB"
        }
      },
      {
        "operation": {
          "name": "resize",
          "type": "Resize",
          "attrs": {
            "width": 540,
            "height": 360,
            "smart_resize": 1,
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "patch_size": 14,
            "merge_size": 2
          }
        }
      },
      {
        "operation": {
          "name": "rescale",
          "type": "Rescale",
          "attrs": { "rescale_factor": 0.00392156862745098 }
        }
      },
      {
        "operation": {
          "name": "normalize",
          "type": "Normalize",
          "attrs": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "qwen2_5_vl": 1
          }
        }
      },
      {
        "operation": {
          "name": "patch_image",
          "type": "PatchImage",
          "attrs": {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2
          }
        }
      }
    ]
  }
}
```

### Transform types

| Type | Purpose | Key attrs |
|---|---|---|
| `DecodeImage` | Decode from bytes | `color_space` |
| `ConvertRGB` | Ensure RGB | — |
| `Resize` | Smart resize | `width`, `height`, `smart_resize`, `min_pixels`, `max_pixels`, `patch_size`, `merge_size` |
| `Rescale` | Scale pixel values | `rescale_factor` |
| `Normalize` | Mean/std normalization | `mean`, `std` |
| `PatchImage` | Extract patches | `patch_size`, `temporal_patch_size`, `merge_size` |

### Generating from HuggingFace config

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
ip = processor.image_processor

processor_config = {
    "processor": {
        "name": "qwen2_5_image_processor",
        "transforms": [
            {"operation": {"name": "decode_image", "type": "DecodeImage",
                           "attrs": {"color_space": "RGB"}}},
            {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
            {"operation": {"name": "resize", "type": "Resize",
                           "attrs": {
                               "width": 540, "height": 360, "smart_resize": 1,
                               "min_pixels": ip.size.get("shortest_edge", 3136),
                               "max_pixels": ip.size.get("longest_edge", 12845056),
                               "patch_size": ip.patch_size,
                               "merge_size": ip.merge_size,
                           }}},
            {"operation": {"name": "rescale", "type": "Rescale",
                           "attrs": {"rescale_factor": ip.rescale_factor}}},
            {"operation": {"name": "normalize", "type": "Normalize",
                           "attrs": {
                               "mean": list(ip.image_mean),
                               "std": list(ip.image_std),
                               "qwen2_5_vl": 1,
                           }}},
            {"operation": {"name": "patch_image", "type": "PatchImage",
                           "attrs": {
                               "patch_size": ip.patch_size,
                               "temporal_patch_size": ip.temporal_patch_size,
                               "merge_size": ip.merge_size,
                           }}},
        ],
    }
}
```

---

## MultiModal pipeline architecture

### VLM prompt flow (3-model split)

```
pixel_values + image_grid_thw  →  [vision.onnx]  →  image_features
                                                          │
input_ids + image_features     →  [embedding.onnx] → inputs_embeds
                                                          │
inputs_embeds + position_ids   →  [model.onnx]    →  logits
              + past_kv
```

### VLM generation flow

```
Prompt stage:
  1. VisionState.Run()         →  image_features
  2. EmbeddingState.ReuseFeaturesBuffer(image_features)
  3. EmbeddingState.Run()      →  inputs_embeds
  4. DecoderState.Run()        →  logits + present_kv
  5. VisionState destroyed (no longer needed)

Token generation stage (loop):
  1. EmbeddingState.Run()      →  inputs_embeds (from single token)
  2. DecoderState.Run()        →  logits + present_kv
```

### Input flow

When `generator.set_inputs(named_tensors)` is called:

1. Tensors matching vision model input names → fed to VisionState
2. Tensors matching embedding model input names → fed to EmbeddingState
3. `input_ids` → used for token counting and embedding lookup
4. `num_image_tokens` → used to allocate image_features buffer size

### The QwenImageProcessor produces

| Tensor | Shape | Description |
|---|---|---|
| `input_ids` | (1, seq_len) | Tokenized prompt with image_pad tokens |
| `pixel_values` | (total_patches, C×T×P×P) | Flattened image patches |
| `image_grid_thw` | (num_images, 3) | Grid dimensions per image |
| `num_image_tokens` | (1,) | Total merged image tokens |

> **Important:** The ORT GenAI QwenImageProcessor does NOT produce
> `cu_seqlens`, `cu_window_seqlens`, or `rotary_pos_ids`. If the vision
> ONNX model requires these, they must be computed externally and injected
> into the NamedTensors.

---

## Common patterns in this package

### Writing genai_config.json from ArchitectureConfig

```python
def _write_genai_config(config, output_dir, model_type="qwen2_5_vl"):
    genai_config = {
        "model": {
            "bos_token_id": config.bos_token_id or 151643,
            "context_length": 4096,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "model.onnx",
                "head_size": config.head_dim,
                "hidden_size": config.hidden_size,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": config.num_attention_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "num_key_value_heads": config.num_key_value_heads,
            },
            "embedding": {
                "filename": "embedding.onnx",
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
            "vision": {
                "filename": "vision.onnx",
                "spatial_merge_size": 2,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
            "eos_token_id": config.eos_token_id or [151645, 151643],
            "pad_token_id": config.pad_token_id or 151643,
            "image_token_id": 151655,
            "vision_start_token_id": 151652,
            "type": model_type,
            "vocab_size": config.vocab_size,
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": 4096,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(genai_config, f, indent=4)
```

### Writing processor_config.json from HuggingFace

```python
def _write_processor_config(processor, output_dir):
    ip = processor.image_processor
    config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage",
                               "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 540, "height": 360, "smart_resize": 1,
                    "min_pixels": ip.size.get("shortest_edge", 3136),
                    "max_pixels": ip.size.get("longest_edge", 12845056),
                    "patch_size": ip.patch_size, "merge_size": ip.merge_size,
                }}},
                {"operation": {"name": "rescale", "type": "Rescale",
                               "attrs": {"rescale_factor": ip.rescale_factor}}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": list(ip.image_mean), "std": list(ip.image_std),
                    "qwen2_5_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size": ip.patch_size,
                    "temporal_patch_size": ip.temporal_patch_size,
                    "merge_size": ip.merge_size,
                }}},
            ],
        }
    }
    with open(os.path.join(output_dir, "processor_config.json"), "w") as f:
        json.dump(config, f, indent=2)
```

---

## Common errors and fixes

### "Protobuf parsing failed"

Missing `model.vision` and/or `model.embedding` sections in genai_config.json.
VLM models require all three model sections.

### "key 'processor' not found"

The `processor_config.json` is in HuggingFace format instead of ort-extensions
format. The HF format has `"image_processor"` as the top key; ORT extensions
needs `"processor"` with a transforms pipeline.

### "Missing Input: cu_window_seqlens"

The vision ONNX model expects packed-attention inputs that the ORT GenAI
processor doesn't provide. Either:
1. Compute them externally and inject via NamedTensors, or
2. Modify the vision model to compute them from `image_grid_thw` internally

### "input_ids size exceeds max length"

For image prompts, the tokenized input_ids (including image_pad tokens) can
be much longer than the default `max_length` in search options. Use
`params.set_search_options(max_length=4096)` or a sufficiently large value.

### "OrtValue shape verification failed"

Mismatch between `num_image_tokens` (computed by the processor) and the
actual vision model output shape. Ensure the same image processor is used
consistently — don't mix ORT GenAI processor output with HF processor
pixel_values.

### Image not recognized despite being processed

If the model generates coherent text but fails to describe image contents
(e.g. describes "snow" but not the cat in the image):

1. **Missing `image_token_id` or `spatial_merge_size`:** Without these
   config fields, ORT GenAI cannot compute 3D M-RoPE position IDs for
   image tokens. The model runs but has no spatial understanding. Add
   `image_token_id`, `vision_start_token_id` at model level and
   `spatial_merge_size` under model.vision.

2. **processor_config.json resize mismatch:** The ORT GenAI processor
   uses `width`/`height` in the Resize transform as direct target
   dimensions (unlike HF which computes from original image size +
   min/max pixels). If set too small, the image loses detail. Compute
   correct dimensions per-image:
   ```python
   factor = patch_size * merge_size  # 28
   new_h = max(factor, round(orig_h / factor) * factor)
   new_w = max(factor, round(orig_w / factor) * factor)
   ```

3. **ONNX model numerical accuracy:** The ONNX model's logits may differ
   from HF (typical max_diff ~8 for VLMs). This causes greedy decoding
   to diverge after 3-4 tokens even though the first tokens match.

---

## Reference files

- **ORT GenAI config structs:**
  `/home/justinchu/dev/onnxruntime-genai/src/config.h`
- **ORT GenAI config parsing:**
  `/home/justinchu/dev/onnxruntime-genai/src/config.cpp`
- **Model type registry:**
  `/home/justinchu/dev/onnxruntime-genai/src/model_type.h`
- **VLM pipeline:**
  `/home/justinchu/dev/onnxruntime-genai/src/models/multi_modal.cpp`
- **Qwen image processor:**
  `/home/justinchu/dev/onnxruntime-genai/src/models/qwen2_5_vl_image_processor.cpp`
- **Reference processor_config.json:**
  `/home/justinchu/dev/onnxruntime-genai/test/test_models/qwen-vision-preprocessing/processor_config.json`
- **Example genai_config generation:**
  `examples/qwen25_vl_ort_genai.py`
