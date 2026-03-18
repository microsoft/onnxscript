---
name: scan-and-multi-image
description: >
  How to use the ONNX Scan op and the Scan + Padding + Compaction pattern in
  mobius. Covers building Scan body subgraphs, implicit inputs,
  carry states, the rename-to-avoid-SSA-violations workaround, and the
  compact_scan_output helper. Primary use case: multi-image vision models
  where per-image computations have variable output sizes.
  Use this skill when building Scan/Loop subgraphs or adding multi-image
  support to a vision model.
---

# Skill: ONNX Scan Op & Multi-Image Vision

## When to use

Use this skill when:

- Adding multi-image support to a vision encoder (iterating over
  `image_grid_thw` rows).
- Building any ONNX subgraph that iterates over a variable-length sequence
  with per-element computation that has dynamic output sizes.
- Working with the `Scan` or `Loop` ONNX ops through the `onnxscript`
  builder API.

## Background: the multi-image problem

ORT GenAI calls the vision model **once** with all images packed together:

- `pixel_values`: `(total_patches, pixel_dim)` — all images concatenated.
- `image_grid_thw`: `(num_images, 3)` INT64 — one `[T, H, W]` per image.

HuggingFace iterates with Python `for t, h, w in grid_thw.tolist()` loops,
accumulating per-image results.  ONNX has no Python loops, so we use the
**Scan** op.

### Why not vectorize?

Per-image computations like rotary position IDs and windowed attention
indices involve `arange(H)`, `Reshape(…, H_m, ms, W_m, ms)`, etc., where
H and W differ across images.  These cannot be batched into a single tensor
operation — we need an explicit loop.

## The Scan + Padding + Compaction pattern

Since ONNX Scan requires **fixed-size outputs per iteration** but each
image produces variable-size results, we:

1. **Pre-compute** `max_size = ReduceMax(per_image_sizes)` in the main
   graph.
2. **Pad** each iteration's output to `max_size` inside the Scan body.
3. **Compact** the concatenated Scan output by removing padding using a
   boolean mask + `Compress`.

```
Main graph:
  max_patches = ReduceMax(T * H * W for each image)
                    │
Scan body (per image):
  pos_ids = compute(T_i, H_i, W_i)     →  (T_i*H_i*W_i, 2)
  padded  = Pad(pos_ids, max_patches)   →  (max_patches, 2)
                    │
Scan output:        →  (num_images, max_patches, 2)
                    │
Compact:
  mask[i,j] = (j < patches_per_image[i])
  result = Compress(flatten(scan_output), flatten(mask))
                    →  (total_patches, 2)
```

## Helpers: `_scan_utils.py`

Location: `src/mobius/components/_scan_utils.py`

### `create_body_graph(state_inputs, scan_inputs, name)`

Creates a Scan body `ir.Graph` and its `GraphBuilder`.

```python
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)

# No carry state, one scan input per iteration
body_thw = ir.Value(
    name="body_thw", shape=ir.Shape([3]),
    type=ir.TensorType(ir.DataType.INT64),
)
body_graph, body_builder = create_body_graph([], [body_thw])
body_op = body_builder.op
```

### `rename_subgraph_values(graph, prefix)`

**Critical step.** ONNX Scan body graphs share a value namespace with the
parent graph in ORT.  Without renaming, node outputs like `v_Constant_11`
in the body collide with identically named values in the main graph,
causing an "SSA form violation" error.

Call this **after** building all body graph ops and **before** calling
`op.Scan(...)` on the main graph:

```python
rename_subgraph_values(body_graph, "rotary_body_")
```

The prefix must be unique per Scan in the model (e.g. `"rotary_body_"`,
`"win_body_"`, `"cu_body_"`).  Graph input/output names are NOT renamed
— they define the Scan interface.

### `compact_scan_output(op, scan_result, lengths_per_iter)`

Removes padding from a `(num_iters, max_len, ...)` Scan output using a
boolean mask built from actual per-iteration lengths.

```python
# After Scan
result = compact_scan_output(op, scan_result, patches_per_image)
# → (total_patches, ...)
```

## Step-by-step: building a Scan

### 1. Compute per-image sizes in the main graph

Extract column vectors from `grid_thw` using Slice + Squeeze.

**Important:** Always specify the squeeze axis to avoid collapsing the
batch dimension when `num_images == 1`:

```python
# GOOD — squeeze only the column axis
T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])

# BAD — squeezes ALL size-1 dims; scalar when num_images=1
T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]))
```

Compute derived values:

```python
H_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])
W_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])
patches_per_image = op.Mul(T_col, op.Mul(H_col, W_col))   # (N,)
max_patches = op.ReduceMax(patches_per_image, keepdims=False)  # scalar
```

### 2. Create the body graph

```python
body_thw = ir.Value(
    name="body_thw", shape=ir.Shape([3]),
    type=ir.TensorType(ir.DataType.INT64),
)
body_graph, body_builder = create_body_graph([], [body_thw])
body_op = body_builder.op
```

### 3. Build per-image computation in the body

Extract T, H, W from the scan input and compute:

```python
bT = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=0)))
bH = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=1)))
bW = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=2)))

result = some_per_image_computation(body_op, bT, bH, bW)
```

**Helper function pattern:**  Extract the per-image logic into a
standalone function that takes `op` as a parameter.  This function works
with either the main graph's `op` or the Scan body's `body_op`:

```python
def _compute_one_image(op, T, H, W, ms):
    """Works with any OpBuilder."""
    H_m = op.Div(H, op.Constant(value_int=ms))
    # ... computation ...
    return result

# In main graph (single-image fast path):
result = _compute_one_image(op, T, H, W, ms)

# In Scan body:
result = _compute_one_image(body_op, bT, bH, bW, ms)
```

### 4. Pad the output to max_size

```python
num_p = body_op.Mul(bT, body_op.Mul(bH, bW))
pad_len = body_op.Reshape(body_op.Sub(max_patches, num_p), [1])

# For 2D output (patches, D): pads = [0, 0, pad_len, 0]
pads = body_op.Concat(
    body_op.Constant(value_ints=[0, 0]),
    pad_len,
    body_op.Constant(value_ints=[0]),
    axis=0,
)
padded = body_op.Pad(result, pads, body_op.Constant(value_int=-1))
```

`max_patches` is an **implicit input** from the main graph — the Scan
body references it directly.  The ONNX Scan spec allows body graphs to
reference outer-scope values.  This is handled correctly by
`onnxscript`'s builder and `onnx_ir`'s serializer.

### 5. Set body outputs and rename

```python
padded.name = "padded_output"
body_graph.outputs.append(padded)

rename_subgraph_values(body_graph, "my_scan_body_")
```

### 6. Call Scan on the main graph

```python
scan_result = op.Scan(
    grid_thw,                    # scan input (iterated over axis 0)
    body=body_graph,             # body subgraph
    num_scan_inputs=1,           # number of scan inputs
    _outputs=1,                  # number of outputs
)
# scan_result: (num_images, max_patches, D)
```

### 7. Compact the result

```python
result = compact_scan_output(op, scan_result, patches_per_image)
# → (total_patches, D)
```

## Advanced: carry states

Carry states (also called "state variables") persist across Scan
iterations.  Use them for accumulating offsets.

**Body graph inputs:** `[state_1, state_2, ..., scan_input_1, ...]`
**Body graph outputs:** `[new_state_1, new_state_2, ..., scan_out_1, ...]`

```python
# State inputs
body_offset = ir.Value(
    name="offset", shape=ir.Shape([]),
    type=ir.TensorType(ir.DataType.INT64),
)
body_thw = ir.Value(
    name="body_thw", shape=ir.Shape([3]),
    type=ir.TensorType(ir.DataType.INT64),
)

# 1 carry state + 1 scan input
body_graph, body_builder = create_body_graph(
    [body_offset], [body_thw], name="window_body",
)
body_op = body_builder.op

# ... compute per-image result ...
win_idx = body_op.Add(local_indices, body_offset)     # add offset
new_offset = body_op.Add(body_offset, total_merged)   # update carry

# Outputs: carry states FIRST, then scan outputs
new_offset.name = "new_offset"
padded_idx.name = "padded_window_index"
body_graph.outputs.extend([new_offset, padded_idx])

rename_subgraph_values(body_graph, "win_body_")

# Call with initial state values
init_offset = op.Constant(value_int=0)
final_offset, scan_idx = op.Scan(
    init_offset,       # initial carry state
    grid_thw,          # scan input
    body=body_graph,
    num_scan_inputs=1,
    _outputs=2,        # 1 carry output + 1 scan output
)
```

### Carry state use cases

| Use case | Carry state | Updated as |
|----------|-------------|------------|
| Window index global offsets | `merged_offset` (INT64 scalar) | `+= T * llm_h * llm_w` per image |
| cu_window_seqlens offsets | `cu_offset` (INT64 scalar) | `= last cu_window value` per image |
| Running patch count | `patch_offset` (INT64 scalar) | `+= T * H * W` per image |

## Pitfalls and gotchas

### 1. SSA violations from name collisions

**Problem:** ORT rejects models where a body graph value name matches a
main graph value name (e.g. both have `v_Constant_11`).

**Fix:** Always call `rename_subgraph_values(body_graph, "unique_prefix_")`
before `op.Scan(...)`.

### 2. Squeeze removes batch dim when N=1

**Problem:** `op.Squeeze(tensor)` removes ALL size-1 dims.  When
`num_images == 1`, a `(1,)` tensor becomes a scalar, causing downstream
`Unsqueeze` or `Reshape` failures.

**Fix:** Always specify the axis: `op.Squeeze(tensor, [1])`.

### 3. Pad format for multi-dim outputs

ONNX `Pad` pads format for an N-dim tensor:
`[d0_begin, d1_begin, ..., dN_begin, d0_end, d1_end, ..., dN_end]`

For a 2D `(rows, cols)` tensor padded only on rows:
`pads = [0, 0, pad_rows, 0]`

For a 1D `(len,)` tensor:
`pads = [0, pad_len]`

### 4. Implicit inputs from parent graph

Scan body graphs can reference values from the parent graph (implicit
inputs).  This is supported by the ONNX spec, `onnxscript`, and ORT.
Use this for `max_patches`, `max_merged`, learned parameters like
`self.pos_embed`, etc.

**No special syntax needed:** just use the main-graph `ir.Value` directly
in `body_op` operations.

### 5. Body graph needs opset imports

`create_body_graph()` already handles this (sets `opset_imports={"": 23}`).
If building manually, ensure the body graph has opset imports.

## Reference files

| File | Content |
|------|---------|
| `src/mobius/components/_scan_utils.py` | `create_body_graph`, `rename_subgraph_values`, `compact_scan_output` |
| `src/mobius/components/_qwen25_vl_vision.py` | Qwen2.5-VL multi-image: rotary, window index, cu_seqlens via Scan |
| `src/mobius/components/_qwen3_vl_vision.py` | Qwen3-VL multi-image: rotary, cu_seqlens, pos embed interpolation via Scan |

### Qwen2.5-VL examples (3 Scans)

1. **`_compute_rotary_pos_ids`** — No carry state.  Pads `(T*H*W, 2)` to
   `(max_patches, 2)`.
2. **`_compute_window_index`** — Two carry states (`merged_offset`,
   `cu_offset`).  Two scan outputs (window index, cu_window).
3. **`_compute_cu_seqlens`** — No carry state.  Pads `(T,)` of hw values
   to `(max_T,)`.  Post-Scan: compact → CumSum → Pad with leading 0.

### Qwen3-VL examples (3 Scans)

1. **`_compute_rotary_pos_ids`** — Same pattern as Qwen2.5-VL but with
   block-row/col indexing.
2. **`_compute_cu_seqlens`** — Same as Qwen2.5-VL.
3. **`_interpolate_pos_embed`** — No carry state.  Bilinear interpolation
   of learned embeddings per image.  References `self.pos_embed` as
   implicit input.  Pads `(T*H*W, hidden_size)` to `(max_patches, D)`.

## Testing Scan-based code

Unit tests (`build_graph_test.py`) verify graph construction only.  To
verify Scan correctness at runtime, build the vision model, fill
initializers with random weights, and run with ORT:

```python
import numpy as np
import onnx_ir as ir
import onnxruntime as ort

# Build and fill weights...
sess = ort.InferenceSession(model_path)

# Single image
r1 = sess.run(None, {
    "pixel_values": np.random.randn(4, pd).astype(np.float32),
    "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int64),
})
assert r1[0].shape[0] == 1  # 4 patches / smu(4) = 1 merged

# Two different-size images
r2 = sess.run(None, {
    "pixel_values": np.random.randn(12, pd).astype(np.float32),
    "image_grid_thw": np.array([[1, 2, 2], [1, 2, 4]], dtype=np.int64),
})
assert r2[0].shape[0] == 3  # (4+8) / 4 = 3 merged
```
