---
name: debugging-vl-pipeline
description: >
  How to debug vision-language (VL) model output issues in mobius.
  Covers the systematic pipeline isolation methodology, common failure modes,
  stage-by-stage comparison with HuggingFace, and numerical tolerance
  expectations. Use this skill when ORT GenAI multimodal output is wrong,
  garbled, or doesn't match HuggingFace.
---

# Skill: Debugging VL Pipeline Issues

## When to use

Use this skill when:

- ORT GenAI produces wrong or irrelevant output for image inputs
- ONNX model logits diverge significantly from HuggingFace
- The model generates text-only descriptions ignoring the image
- Image features appear correct but decoder output is wrong

## Debugging methodology: isolate each stage

VL models have 3 stages. Debug by isolating and validating each stage
independently, comparing against HuggingFace at every boundary.

```
pixel_values ──► [1. Vision] ──► image_features
                                       │
input_ids    ──► [2. Embedding] ◄──────┘
                       │
                       ▼
              inputs_embeds + position_ids + attention_mask
                       │
                       ▼
                 [3. Decoder] ──► logits
```

### Stage 1: Vision model

**What to check:**
- Output shape: `(num_patches, hidden_size)`
- Expected patches = `t * (h / merge) * (w / merge)` from `grid_thw`
- Compare features against HF vision encoder output

```python
# HF reference
with torch.no_grad():
    hf_vision_out = hf_model.model.visual(
        pixel_values, grid_thw=grid_thw
    )
# ONNX
session = OnnxModelSession(pkg["vision"])
onnx_out = session.run({"pixel_values": pv, "grid_thw": grid_thw})

# Compare
cos_sim = np.dot(hf_flat, onnx_flat) / (norm_hf * norm_onnx)
print(f"Vision cos_sim: {cos_sim:.6f}")  # Should be > 0.99
```

**Common issues:**
- Wrong pixel value normalization (mean/std mismatch)
- `grid_thw` shape or values don't match HF processor output
- Missing `temporal_patch_size` in patch embedding

### Stage 2: Embedding model

**What to check:**
- Image features injected at correct token positions
- Non-image positions have correct text embeddings
- Output shape: `(1, seq_len, hidden_size)`

```python
# Verify image token positions
image_mask = (input_ids[0] == image_token_id)  # 151655
num_image_positions = image_mask.sum()
assert num_image_positions == image_features.shape[0]

# Compare embeddings at text positions (should match HF exactly)
text_mask = ~image_mask
cos_sim_text = cosine_similarity(
    onnx_embeds[0, text_mask], hf_embeds[0, text_mask]
)
print(f"Text embedding cos_sim: {cos_sim_text:.6f}")  # Should be 1.0
```

**Common issues:**
- Image token count mismatch between processor and vision model
- Missing zero-padding row in embedding model (for text-only inputs)
- Wrong `image_token_id` used for Gather/Where mask

### Stage 3: Decoder

**What to check:**
- Logits shape matches HF: `(1, seq_len, vocab_size)`
- First token prediction matches HF (argmax of last position)
- Cosine similarity of logit vectors

```python
# With HF-computed position_ids (ground truth):
onnx_logits = decoder_session.run(feeds)["logits"]
hf_logits = hf_model(**hf_inputs).logits.numpy()

max_diff = np.abs(onnx_logits - hf_logits).max()
cos_sim = cosine_similarity(onnx_logits[0, -1], hf_logits[0, -1])
print(f"max_diff={max_diff:.2f}, cos_sim={cos_sim:.4f}")
# Typical: max_diff=5-10, cos_sim>0.98
```

**Common issues:**
- Wrong position_ids (see "3D M-RoPE" section below)
- Missing KV cache initialization
- Wrong attention_mask length

## Critical: 3D M-RoPE position IDs

Qwen2-VL / Qwen2.5-VL / Qwen3-VL use **3D Multimodal RoPE** where
`position_ids` has shape `(3, batch, seq_len)`:

```
position_ids[0] = temporal positions
position_ids[1] = height positions
position_ids[2] = width positions
```

**Text tokens:** all 3 dimensions have the same sequential value.

**Image tokens:** temporal is constant, height/width vary over the
image grid `(h/merge, w/merge)`:
```
temporal: [offset, offset, offset, ..., offset]
height:   [offset, offset+1, offset+1, ..., offset+h/merge-1]
width:    [offset, offset+1, offset, offset+1, ..., offset+w/merge-1]
```

**Text after image:** all 3 dimensions resume from
`max(temporal, height, width) + 1`.

### ORT GenAI config requirements

For ORT GenAI to compute 3D M-RoPE automatically, the following
`genai_config.json` fields are **required**:

| Field | Level | Purpose |
|-------|-------|---------|
| `model.image_token_id` | model | Token ID for `<\|image_pad\|>` (e.g. 151655) |
| `model.vision_start_token_id` | model | Token ID for `<\|vision_start\|>` (e.g. 151652) |
| `model.vision.spatial_merge_size` | vision | Grid merge factor (typically 2) |

**Without these fields**, ORT GenAI falls back to standard 1D positions,
which produces completely wrong output for image inputs (the model may
describe a "snowy landscape" instead of the actual image content).

## Common failure modes and fixes

### 1. Image not recognized (wrong output for image inputs)

**Symptoms:** Model produces generic or hallucinated descriptions that
don't match the input image. Text-only generation works correctly.

**Root causes (in order of likelihood):**

1. **Missing genai_config fields** — `image_token_id`,
   `vision_start_token_id`, or `spatial_merge_size` not set.
   Without these, position_ids are 1D instead of 3D M-RoPE.

2. **Image resize mismatch** — ORT processor resizes image to different
   dimensions than HF processor, producing different number of vision
   tokens. ORT's `width`/`height` in `processor_config.json` are used
   as direct resize targets, unlike HF's smart_resize which computes
   target from original image dimensions.

3. **Processor config format** — ORT GenAI expects ort-extensions format
   `processor_config.json`, not HuggingFace format. The file must include
   `DecodeImage`, `ConvertRGB`, `Resize`, `Rescale`, `Normalize`, and
   `PatchImage` transforms with correct attributes.

**Fix for resize mismatch:**
```python
def _update_resize_for_image(processor_config_path, image_path):
    """Recompute resize dimensions from actual image like HF does."""
    from PIL import Image
    img = Image.open(image_path)
    w, h = img.size
    factor = 14 * 2  # patch_size * merge_size
    new_w = round(w / factor) * factor
    new_h = round(h / factor) * factor
    # Update width/height in processor_config.json
```

### 2. Numerical divergence in greedy decoding

**Symptoms:** First 1-3 tokens match HF, then output diverges.

**Expected behavior:** This is inherent to ONNX vs PyTorch numerical
differences. ONNX models use different operator implementations that
accumulate small floating-point errors.

**Typical metrics for Qwen2.5-VL 3B:**
- max_diff in logits: 5-10
- mean_diff in logits: 0.5-1.5
- cosine similarity: 0.98-0.99
- First token: matches HF
- Greedy decoding: diverges at token 3-5

**This is NOT a bug** if the metrics above are within range. Both models
produce semantically similar descriptions.

### 3. Vision model output shape mismatch

**Symptoms:** Vision model produces wrong number of patches.

**Debug:** Check `grid_thw` values:
```python
# For Qwen2.5-VL with merge_size=2:
t, h, w = grid_thw[0]
expected_patches = t * (h // 2) * (w // 2)
actual_patches = vision_output.shape[0]
assert expected_patches == actual_patches
```

### 4. Embedding model text-only failure

**Symptoms:** Error when running without images (num_image_tokens=0).

**Fix:** Ensure embedding model pads `image_features` with a zero row
before Gather, then uses a Where mask to select only real features:
```python
# Pad with zero row so Gather with index 0 doesn't fail
padded = op.Concat(
    op.ConstantOfShape(...),  # (1, hidden_size) zeros
    image_features,
    axis=0,
)
```

## Extracting intermediate ONNX values

When a stage (e.g., the vision encoder) diverges, drill down by extracting
intermediate values from the ONNX graph.  This lets you compare block-by-block
or even op-by-op against HuggingFace.

### Method 1: Add intermediate outputs to the ONNX graph

The most reliable approach — expose any internal node's output as a graph
output so ORT returns it alongside normal outputs.

```python
import onnx

model = onnx.load("vision.onnx")
graph = model.graph

# Find the node whose output you want to inspect
for node in graph.node:
    if node.op_type == "RMSNormalization" and "block_0" in node.output[0]:
        # Name the output (if unnamed, give it a name)
        target_output = node.output[0]
        break

# Add as a graph output
graph.output.append(
    onnx.helper.make_tensor_value_info(target_output, onnx.TensorProto.FLOAT, None)
)
onnx.save(model, "vision_debug.onnx")

# Now ORT will return this value alongside image_features
session = ort.InferenceSession("vision_debug.onnx")
results = session.run(None, feeds)
# results[-1] is the intermediate value
```

### Method 2: Use `ir.Model` graph manipulation (preferred for mobius)

When working with `ir.Model` objects from the build pipeline, manipulate
the graph directly without saving/loading:

```python
from mobius._testing.ort_inference import OnnxModelSession

pkg = build(model_id, dtype="f32", load_weights=True)
vision_model = pkg["vision"]
graph = vision_model.graph

# Find target nodes by op type or name pattern
target_nodes = [n for n in graph if n.op_type == "RMSNormalization"]

# RMSNorm input nodes are natural block boundaries:
# rms_nodes[0] = block 0 norm1 (input = patch_embed output)
# rms_nodes[2] = block 1 norm1 (input = block 0 output)
# rms_nodes[2*i] = block i norm1 (input = block i-1 output)
block_0_output = target_nodes[2].inputs[0]  # block 0's output
block_0_output.name = "block_0_output"
graph.outputs.append(block_0_output)

session = OnnxModelSession(vision_model)
out = session.run({"pixel_values": pv, "grid_thw": grid_thw})
block_0_out = out["block_0_output"]
session.close()
```

### Method 3: Hook HuggingFace model for reference values

Use PyTorch hooks to extract intermediate values from HuggingFace at
the same points:

```python
intermediates = {}

def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().cpu().numpy()
        else:
            intermediates[name] = output.detach().cpu().numpy()
    return fn

# Register hooks on specific blocks
for i, block in enumerate(hf_model.model.visual.blocks):
    block.register_forward_hook(hook_fn(f"block_{i}"))

# Run forward pass — hooks capture all intermediate values
with torch.no_grad():
    hf_out = hf_model.model.visual(pixel_values, grid_thw=grid_thw)

# Now compare block by block
for i in range(num_blocks):
    hf_block_out = intermediates[f"block_{i}"]
    cos = cosine_similarity(onnx_block_out, hf_block_out)
    print(f"Block {i}: cos={cos:.6f}")
```

### Block-by-block comparison strategy

When overall output diverges, narrow down by comparing each transformer
block's output sequentially:

```python
for i in range(num_blocks):
    onnx_out_i = extract_onnx_block_output(vision_model, i, feeds)
    hf_out_i = intermediates[f"block_{i}"]

    cos = cosine_similarity(onnx_out_i.flatten(), hf_out_i.flatten())
    max_diff = np.max(np.abs(onnx_out_i - hf_out_i))
    print(f"Block {i:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}")
```

Typical pattern for a bug in block N:
```
Block 0:  cos=1.000000  max_diff=0.0001  ← perfect
Block 1:  cos=1.000000  max_diff=0.0001  ← perfect
...
Block N:  cos=0.961000  max_diff=4.8500  ← divergence starts!
Block N+1: cos=0.892000  max_diff=25.00  ← error compounds
```

Once you identify the divergent block, drill deeper into that block's
sub-operations (attention, MLP, normalization) to find the root cause.

### Comparing specific weight values

To verify weights loaded correctly, compare ONNX initializers against
HuggingFace state dict:

```python
from safetensors import safe_open

# Load HF weights
with safe_open(safetensors_path, framework="numpy") as f:
    hf_weight = f.get_tensor("visual.blocks.0.attn.qkv.weight")

# Get ONNX weight from ir.Model
onnx_weight = vision_model.graph.initializers["blocks.0.attn.qkv.weight"]
onnx_np = onnx_weight.const_value.numpy()

max_diff = np.max(np.abs(hf_weight.astype(np.float32) - onnx_np))
print(f"Weight diff: {max_diff}")  # Should be 0.0
```

## Integration test patterns

### Full VL forward test
```python
# Build model → process image with HF processor → run ONNX → compare logits
assert_logits_close(onnx_logits, hf_logits, rtol=2e-2, atol=2e-1)
```

### 3-model pipeline test
```python
# Vision → Embedding → Decoder, each stage uses OnnxModelSession
# Compare final decoder logits against HF single-model forward
```

### ORT GenAI end-to-end test
```python
# Build → save flat → write genai_config → load with ort_genai → generate
# Verify output length > input (basic sanity)
```

### 5. Vision encoder internal divergence (cos < 0.5)

**Symptoms:** Vision features have very low cosine similarity (< 0.5)
against HuggingFace, even though patch embedding and weights are correct.

**Debug with block-by-block comparison** (see "Extracting intermediate
ONNX values" above). Common root causes:

1. **Wrong rotary embedding dimension** — Qwen2.5-VL vision uses 2D
   position encoding (height + width). The rotary dim must be
   `head_dim // 2`, not `head_dim`. Each half (head_dim // 4 frequencies)
   covers one spatial dimension. With full `head_dim`, you get 2× too
   many frequencies with wrong values. **Result: cos ≈ 0.25.**

2. **Missing `fullatt_block_indexes`** — Qwen2.5-VL alternates between
   windowed attention (local windows of `window_size` patches) and full
   attention (all patches attend to all). Blocks at indexes `[7, 15, 23, 31]`
   use full attention. If `fullatt_block_indexes` is not extracted from
   HF config, all blocks use windowed attention. **Result: blocks 0-6
   are perfect (they're windowed anyway), but block 7+ diverges.**

3. **Wrong attention bias construction** — Full-attention blocks should
   have an all-zeros bias (everything attends to everything). Windowed
   blocks have a block-diagonal bias. Check the bias by inspecting
   sparsity: `(bias == -inf).float().mean()` should be ~0% for full
   attention, ~98% for windowed.

**Config extraction checklist for vision encoders:**
```python
# These fields MUST be extracted from HF vision_config:
fullatt_block_indexes = getattr(vc, "fullatt_block_indexes", None)
window_size = getattr(vc, "window_size", None)
spatial_merge_size = getattr(vc, "spatial_merge_size", 2)
temporal_patch_size = getattr(vc, "temporal_patch_size", 2)
```

## Reference files

- **Integration tests:** `tests/integration_test.py`
  (`TestVLFullForward`, `TestQwen25VL3Model`)
- **ORT GenAI tests:** `tests/ort_genai_test.py`
  (`TestOrtGenaiQwen25VL.test_multimodal_image_generation`)
- **Example scripts:** `examples/qwen25_vl_ort_genai.py`,
  `examples/qwen3_vl_ort_genai.py`
- **genai_config reference:** `.github/skills/ort-genai-config/SKILL.md`
- **ORT GenAI position_ids code (external):**
  `onnxruntime-genai/src/models/position_inputs.cpp:617-814`
