---
name: phi4mm-component-parity
description: >
  How to ensure each multimodal model component (vision encoder, speech encoder,
  embedding/projector, text decoder) matches HuggingFace output. Covers pipeline
  isolation methodology, common failure modes from real debugging experience,
  step-by-step debugging process, and integration test patterns. Applicable to
  any multimodal model with similar architecture (Phi4MM, future audio+vision
  models). Use this skill when multimodal ONNX model output diverges from
  HuggingFace, or when adding a new multimodal model with multiple encoders.
---

# Skill: Multimodal Component Parity Debugging

## When to use

Use this skill when:

- A multimodal ONNX model's logits diverge systematically from HuggingFace
- You're adding a new multimodal model and need to verify each component
- Integration tests fail with large numerical differences (not just tolerance)
- Weights appear to load but the model produces wrong outputs
- You need to isolate which component (vision, speech, embedding, decoder)
  is causing divergence

For vision-language-only models (no speech), see also the
`debugging-vl-pipeline` skill which covers VLM-specific issues like 3D M-RoPE.

## Pipeline isolation methodology

Multimodal models with N encoders have N+2 stages (encoders + embedding +
decoder). Debug by comparing each stage independently against HuggingFace
at every boundary.

### 4-model multimodal pipeline (e.g., Phi4MM)

```
pixel_values ──► [1. Vision Encoder] ──► image_features ──┐
                                                           │
audio_embeds ──► [2. Speech Encoder] ──► speech_features ──┤
                                                           │
input_ids    ──► [3. Embedding/Fusion] ◄───────────────────┘
                        │
                        ▼
                   inputs_embeds
                        │
                        ▼
                 [4. Decoder + LoRA] ──► logits
```

**Golden rule:** Start from the simplest case (text-only, no encoders),
verify it matches HF, then add one modality at a time.

### Stage 1: Vision encoder

Compare SigLIP/ViT encoder output between ONNX and HF.

```python
# ONNX
vision_session = OnnxModelSession(pkg["vision"])
onnx_out = vision_session.run({
    "pixel_values": pixel_values,
    "image_sizes": image_sizes,  # for HD multi-crop models
})
onnx_features = onnx_out["image_features"]

# HuggingFace reference
with torch.no_grad():
    hf_features = hf_model.model.embed_tokens_extend.image_embed(
        pixel_values_tensor, image_sizes_tensor
    )

# Compare
cos_sim = cosine_similarity(onnx_features.flatten(), hf_features.flatten())
print(f"Vision cos_sim: {cos_sim:.6f}")  # Should be > 0.99
```

**What to check:**
- Output shape: `(num_image_tokens, text_hidden_size)` after projection
- For HD models: token count varies by image resolution
- Projection MLP maps vision hidden dim → text hidden dim
- Position embeddings added correctly (2D vs 3D shape)

### Stage 2: Speech encoder

Compare Conformer encoder output between ONNX and HF.

```python
# ONNX
speech_session = OnnxModelSession(pkg["speech"])
onnx_out = speech_session.run({
    "audio_embeds": mel_features,
    "audio_sizes": audio_sizes,
    "audio_projection_mode": np.array(0, dtype=np.int64),  # 0=speech
})
onnx_features = onnx_out["audio_features"]

# HuggingFace reference
with torch.no_grad():
    hf_features = hf_model.model.embed_tokens_extend.audio_embed(
        mel_tensor, audio_sizes_tensor, input_mode=2  # speech mode
    )
```

**What to check:**
- Output shape: `(num_audio_tokens, text_hidden_size)` after projection
- Compression rate: Conformer typically does 8× time reduction
- Projection branch selection: speech vs vision projection
- Conv subsampling produces correct output length

### Stage 3: Embedding / fusion

Compare token embedding + feature fusion between ONNX and HF.

```python
# ONNX
emb_session = OnnxModelSession(pkg["embedding"])
onnx_embeds = emb_session.run({
    "input_ids": input_ids,
    "image_features": image_features,  # or zeros([0, H]) if text-only
    "audio_features": speech_features,  # or zeros([0, H]) if no audio
})["inputs_embeds"]

# HuggingFace reference (text-only baseline)
with torch.no_grad():
    hf_embeds = hf_model.model.embed_tokens(input_ids_tensor)

# For text-only, these should match exactly
max_diff = np.abs(onnx_embeds - hf_embeds).max()
print(f"Embedding max_diff: {max_diff:.6f}")  # Should be < 1e-5
```

**What to check:**
- Text-only: should match HF `embed_tokens` exactly (no fusion)
- With features: verify token replacement at correct positions
- InputMixer handles zero-length feature tensors without crashing
- Feature positions align with special token positions in input_ids

### Stage 4: Decoder

Compare logits between ONNX and HF for the full forward pass.

```python
# ONNX decoder with KV cache
decoder_session = OnnxModelSession(pkg["model"])
onnx_logits = decoder_session.run({
    "inputs_embeds": inputs_embeds,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    # ... past_key_values (zeros for prefill)
})["logits"]

# HuggingFace full model forward
with torch.no_grad():
    hf_out = hf_model(input_ids=input_ids_tensor, ...)
hf_logits = hf_out.logits.numpy()

cos_sim = cosine_similarity(onnx_logits[0, -1], hf_logits[0, -1])
max_diff = np.abs(onnx_logits - hf_logits).max()
print(f"Decoder: cos_sim={cos_sim:.4f}, max_diff={max_diff:.2f}")
```

**Typical acceptable metrics (float32):**
- max_diff: 5-10
- mean_diff: 0.5-1.5
- cosine similarity: > 0.98
- First token argmax: matches HF

## Common failure modes and fixes

### 1. Weight name alignment (missing weights)

**Symptoms:** Hundreds or thousands of weights reported as "unmatched" by
`apply_weights`. Model runs but produces garbage output.

**Root causes encountered:**

#### a. Module forward() bypass (240 missing weights in Phi4MM)

The most insidious bug. When a component's `forward()` method directly
accesses nested sub-module parameters (e.g., `self.glu.ext_pw_conv_1d.weight`)
instead of calling the sub-module's `forward()` method, onnxscript cannot
resolve the full module path for the parameter. The weight ends up as an
unnamed, non-initializer constant in the graph.

```python
# BAD — weights become unnamed
def forward(self, op, x):
    return op.Conv(x, self.glu.ext_pw_conv_1d.weight,
                      self.glu.ext_pw_conv_1d.bias, ...)

# GOOD — onnxscript resolves full module path
def forward(self, op, x):
    return self.glu(op, x)  # GLU.forward() calls op.Conv internally
```

**Detection:** Check the ONNX graph for Conv/MatMul nodes where weight
inputs have `is_initializer=False` and generic names like "weight"/"bias".

**Fix:** Add `forward()` methods to sub-modules and call them instead of
directly accessing their parameters.

#### b. ModuleList subclass causing name doubling (8 weights)

Subclassing `nn.ModuleList` causes the module's own name to appear twice
in the parameter path: `img_projection.img_projection.0.weight` instead
of `img_projection.0.weight`.

```python
# BAD — name doubling
class ProjectionMLP(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.append(nn.Linear(1152, 3072))
        self.append(nn.Linear(3072, 3072))

# GOOD — use nn.Module with indexed children
class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1152, 3072), nn.Linear(3072, 3072)]
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
```

#### c. setattr with dotted names

Using `setattr(self, "audio_projection.speech", module)` creates a single
attribute with a dot in its name, rather than a nested module. The resulting
ONNX parameter names won't match HuggingFace's `ModuleDict`-style naming.

**Fix:** Use `nn.ModuleDict` or create proper nested attributes.

### 2. Shape mismatches (position embedding 2D vs 3D)

**Symptoms:** `RuntimeError: shape mismatch` during weight loading.

**Root cause:** The ONNX component declares a parameter with a different
number of dimensions than the HuggingFace weight. Example: PatchEmbedding
declares `position_embedding.weight` as `[num_patches, hidden_size]` (2D),
but HF stores `[1, num_patches, hidden_size]` (3D).

**Fix in `preprocess_weights()`:**
```python
# Squeeze the extra batch dimension to match ONNX declaration
if "position_embedding.weight" in key and state_dict[key].dim() == 3:
    state_dict[key] = state_dict[key].squeeze(0)  # [1,N,H] → [N,H]
```

**General rule:** Check whether the preprocess_weights transform goes
in the correct direction (squeeze vs unsqueeze). A common mistake is
writing the transform backwards.

### 3. Dtype mismatches (float64 vs float32)

**Symptoms:** ONNX Runtime error: "type mismatch in Mul/Add node" during
inference.

**Root causes:**

#### a. NumPy default float64

`numpy.array(python_float)` defaults to float64. Any constant created
from a Python scalar without explicit dtype will be float64 in the graph.

```python
# BAD — float64 constant
scale = numpy.array(alpha / rank)  # defaults to float64
op.Mul(x, scale)  # Mul(float32, float64) → type error

# GOOD — explicit float32
scale = numpy.array(alpha / rank, dtype=numpy.float32)
op.Mul(x, scale)
```

#### b. Python int auto-promotion

When passing a Python `int` to an op that expects a tensor, onnxscript
may auto-promote to float64 (implementation-dependent).

```python
# RISKY — Python int may become float64
op.Mul(int64_tensor, self.max_position_embeddings)

# SAFE — explicit constant
op.Mul(int64_tensor,
       op.Constant(value_int=self.max_position_embeddings))
```

**Detection:** Run the ONNX model and look for type mismatch errors.
The error message includes the node name — trace it back to the source.

### 4. LoRA application mismatch (conditional vs unconditional)

**Symptoms:** Systematic divergence (> 80% logits mismatch) across ALL
test cases, but the model structurally runs correctly.

**Root cause:** Some models apply LoRA adapters conditionally based on
input modality. For example, Phi4MM applies:
- `input_mode=0` (text): no adapters
- `input_mode=1` (vision): vision LoRA only
- `input_mode=2` (speech): speech LoRA only
- `input_mode=3` (combined): both adapters

If the ONNX model unconditionally applies all adapters (both vision and
speech LoRA always active), it diverges from HF when HF uses a different
input mode.

**Quick fix for integration tests:** Run the HF reference with the mode
that matches the ONNX model's behavior (e.g., `input_mode=3` to match
unconditional application of both adapters).

**Proper fix:** Add an `input_mode` input to the decoder model and use
conditional logic to selectively apply adapters.

**Detection:** If text-only inference diverges but the model generates
reasonable (not garbage) output, suspect LoRA mode mismatch. Temporarily
zero out all LoRA weights — if base model matches HF perfectly, the
LoRA application mode is the issue.

### 5. Empty tensor handling (zero-length features)

**Symptoms:** Crash during text-only inference when no image/audio
features are present.

**Root cause:** The embedding model's `InputMixer` uses `GatherElements`
to place features at special token positions. With zero features, the
gather indices are empty but the operation may still execute on the
padded dimension, causing shape errors.

**Fix pattern:** Zero-pad before Gather, then use Where to mask results:
```python
# Pad with one zero row so Gather never accesses out-of-bounds
padded = op.Concat(
    op.ConstantOfShape(op.Constant(value_ints=[1, hidden_size])),
    features,  # may be [0, hidden_size]
    axis=0,
)
# After Gather, mask out the padding positions with Where
result = op.Where(feature_mask, gathered, text_embeddings)
```

### 6. HD transform image format (5D vs 4D)

**Symptoms:** Vision model crashes or produces wrong output with multi-crop
HD images.

**Root cause:** HD-capable vision models expect images in different formats:
- Some expect `[batch, channels, height, width]` (4D, single crop per batch)
- Others expect `[num_images, num_crops, channels, height, width]` (5D)

The HF processor output format must match the ONNX model's input format.
If using the HF processor for test input preparation, verify it produces
the expected format.

**Fix:** Check the HF model's preprocessing code for the expected format,
and ensure the ONNX model's input signature matches. For tests, either:
- Use the HF processor: `processor(images=image, return_tensors="np")`
- Or manually construct the correct format for simple test cases

### 7. Causal mask construction (inputs_embeds vs input_ids)

**Symptoms:** Attention mask has wrong length, causing decoder crash or
wrong output.

**Root cause:** When the decoder receives `inputs_embeds` instead of
`input_ids`, the sequence length must be derived from the embeds tensor
shape, not from input_ids. If the mask is built from input_ids length but
the actual sequence includes fused image/audio tokens, the lengths diverge.

**Fix:** Always derive `seq_len` from `inputs_embeds.shape[1]` when the
model uses inputs_embeds as input.

## Step-by-step debugging process

### Phase 1: Text-only baseline

1. **Build ONNX model** with tiny config (for fast iteration) or full
   weights (for accuracy).
2. **Run text-only** through embedding → decoder (skip encoders).
3. **Compare embedding output** against `hf_model.model.embed_tokens(ids)`.
   If this diverges, the issue is in weight loading or embedding model.
4. **Compare decoder logits** against HF full forward.
   If embedding matches but logits diverge, issue is in decoder.

### Phase 2: Isolate decoder issues

5. **Check weight count** — verify all expected weights are loaded:
   ```python
   pkg = build(model_id, load_weights=True)
   # apply_weights prints statistics: applied, skipped, unmatched
   ```
6. **Disable LoRA** — if the model uses LoRA, zero out adapter weights and
   compare base model output against HF with adapters disabled.
7. **Layer-by-layer** — add intermediate outputs to the ONNX graph (see
   `debugging-vl-pipeline` skill) to find which decoder layer first diverges.

### Phase 3: Add modalities

8. **Vision only** — run vision encoder, feed features to embedding, compare.
9. **Audio only** — run speech encoder, feed features to embedding, compare.
10. **Combined** — all modalities together.

At each step, if a newly added component causes divergence, isolate that
component's output against HF.

### Phase 4: LoRA verification

11. **Match input modes** — ensure HF reference uses the same adapter
    activation mode as ONNX (e.g., `input_mode=3` for both adapters).
12. **Compare with LoRA** — verify LoRA scaling factor: `alpha / rank`.
13. **Check adapter routing** — for multi-adapter models, verify the correct
    adapter set is active for each modality combination.

## Integration test patterns

### Test configuration

```python
# Always use for HF reference:
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="eager",  # No flash_attn dependency
    torch_dtype=torch.float32,     # Match ONNX precision
)

# For models with conditional LoRA:
hf_model.input_mode = 3  # Match ONNX unconditional LoRA
# Or: pass input_mode=3 to forward() if supported
```

### Text-only test

```python
def test_text_only_prefill_logits_match(self):
    input_ids = tokenizer.encode("Hello world", return_tensors="np")
    empty_image = np.zeros((0, hidden_size), dtype=np.float32)
    empty_audio = np.zeros((0, hidden_size), dtype=np.float32)

    embeds = embedding_session.run({
        "input_ids": input_ids,
        "image_features": empty_image,
        "audio_features": empty_audio,
    })["inputs_embeds"]

    onnx_logits = decoder_session.run({
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len).reshape(1, -1),
        # ... zero KV cache
    })["logits"]

    hf_logits = hf_model(input_ids=..., input_mode=3).logits.numpy()
    assert_logits_close(onnx_logits, hf_logits)
```

### Audio test

```python
def test_audio_prefill_logits_match(self):
    # Prepare mel spectrogram input
    mel = load_audio_as_mel(audio_path)  # [1, n_mel, time]

    speech_out = speech_session.run({
        "audio_embeds": mel,
        "audio_sizes": np.array([[mel.shape[-1]]], dtype=np.int64),
        "audio_projection_mode": np.array(0, dtype=np.int64),
    })
    speech_features = speech_out["audio_features"]

    # Build input_ids with audio placeholder tokens
    input_ids = build_audio_input_ids(prompt, num_audio_tokens)

    embeds = embedding_session.run({
        "input_ids": input_ids,
        "image_features": np.zeros((0, hidden_size), dtype=np.float32),
        "audio_features": speech_features,
    })["inputs_embeds"]

    onnx_logits = decoder_session.run(...)["logits"]
    hf_logits = hf_forward_with_audio(...)
    assert_logits_close(onnx_logits, hf_logits)
```

### Tolerance guidelines

| Precision | atol | rtol | Notes |
|-----------|------|------|-------|
| float32 | 1e-4 | 2e-2 | Standard for single-forward-pass |
| float32 (deep model, 32+ layers) | 1e-3 | 5e-2 | Error compounds over layers |
| float16 / bfloat16 | 0.01 | 0.05 | Wider tolerance for mixed precision |
| Cosine similarity (last token) | > 0.98 | — | Primary correctness metric |
| Argmax match (first prediction) | exact | — | Should always match |

### Weight loading verification

After `apply_weights`, check the statistics:
```python
# Expected output:
# Applied: 485/485 weights
# Skipped: 0  (weights in state_dict but not in graph)
# Unmatched: 0  (weights in state_dict with no graph match)

# If unmatched > 0, dump the names to find alignment issues:
pkg = build(model_id)
state_dict = download_weights(model_id)
state_dict = module.preprocess_weights(state_dict)
# Compare state_dict.keys() vs graph initializer names
```

## Vision-specific verification (HD multi-crop)

For models with HD dynamic resolution (Phi4MM, Phi3-Vision):

### Input preparation

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    model_id, trust_remote_code=True
)
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
)
pixel_values = inputs["pixel_values"]  # [num_crops, C, H, W] or 5D
image_sizes = inputs["image_sizes"]     # [num_images, 2]
```

### HD transform verification

The HD transform typically:
1. Splits image into base (global) + sub-image crops
2. Encodes each crop through vision encoder → `[num_patches, hidden_size]`
3. Applies spatial merge (e.g., AvgPool2d + reshape) → compressed tokens
4. Adds learned separators (glb_GN between global/sub, sub_GN between subs)
5. Projects to text dimension via MLP

```python
# Verify token count matches expected:
# global: (image_size/patch_size)^2 / merge^2 tokens
# per sub-image: same count
# separators: 1 glb_GN + (num_subs - 1) sub_GN rows
total_expected = global_tokens + num_subs * sub_tokens + separator_count
assert image_features.shape[0] == total_expected
```

### Testing without HD (simpler)

For initial validation, use base resolution (single crop, no HD):
```python
# Single image at base resolution — bypasses HD transform
pixel_values = np.random.randn(1, 3, 384, 384).astype(np.float32)
image_sizes = np.array([[384, 384]], dtype=np.int64)
```

## Reference files

- **Integration tests:** `tests/phi4mm_integration_test.py`,
  `tests/integration_test.py`
- **VL debugging skill:** `.github/skills/debugging-vl-pipeline/SKILL.md`
- **Model implementation:** `src/mobius/models/phi.py`
- **Audio components:** `src/mobius/components/_audio.py`
- **Vision components:** `src/mobius/components/_vision.py`
- **LoRA component:** `src/mobius/components/_lora.py`
- **Weight loading:** `src/mobius/_weight_loading.py`
- **ORT GenAI config skill:** `.github/skills/ort-genai-config/SKILL.md`
- **Weight name alignment skill:**
  `.github/skills/weight-name-alignment/SKILL.md`
