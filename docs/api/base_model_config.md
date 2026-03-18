# `BaseModelConfig`

Base configuration class for all model architectures.

```python
from mobius import BaseModelConfig, ArchitectureConfig
```

## Class Hierarchy

```
BaseModelConfig
├── ArchitectureConfig          # Decoder-only models
│   ├── CausalLMConfig          # With additional causal LM fields
│   ├── Gemma2Config            # Gemma-2 specific
│   ├── MambaConfig             # SSM models
│   └── VisionLanguageConfig    # Vision-language models
│       └── MllamaConfig
├── EncoderConfig               # Encoder-only (BERT)
├── WhisperConfig               # Whisper speech-to-text
├── Sam2Config                  # SAM2 segmentation
├── SegformerConfig             # Segformer segmentation
├── DepthAnythingConfig         # Depth estimation
└── YolosConfig                 # Object detection
```

## `BaseModelConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | — | Vocabulary size. |
| `hidden_size` | `int` | — | Hidden dimension. |
| `intermediate_size` | `int` | — | Feed-forward intermediate size. |
| `num_hidden_layers` | `int` | — | Number of transformer layers. |
| `num_attention_heads` | `int` | — | Number of attention heads. |
| `num_key_value_heads` | `int` | — | Number of KV heads (for GQA). |
| `head_dim` | `int` | — | Dimension per attention head. |
| `hidden_act` | `str \| None` | `None` | Activation function (`"silu"`, `"gelu"`, etc.). |
| `pad_token_id` | `int` | — | Padding token ID. |
| `tie_word_embeddings` | `bool` | `False` | Whether to tie input/output embeddings. |
| `dtype` | `ir.DataType` | `FLOAT` | Model weight dtype. |

## `ArchitectureConfig` Additional Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `max_position_embeddings` | `int` | — | Maximum sequence length. |
| `rms_norm_eps` | `float` | `1e-6` | RMSNorm epsilon. |
| `rope_type` | `str` | `"default"` | RoPE type (`"default"`, `"llama3"`, etc.). |
| `rope_theta` | `float` | `10000.0` | RoPE base frequency. |
| `rope_scaling` | `dict \| None` | `None` | RoPE scaling configuration. |
| `sliding_window` | `int \| None` | `None` | Sliding window attention size. |
| `num_local_experts` | `int \| None` | `None` | Number of MoE experts. |
| `num_experts_per_tok` | `int \| None` | `None` | Active experts per token. |

## Usage

Configs are typically created automatically by `build()` from HuggingFace
configs. For manual construction:

```python
from mobius import ArchitectureConfig

config = ArchitectureConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=4096,
    hidden_act="silu",
    head_dim=128,
    pad_token_id=0,
)

# Validate the config
config.validate()
```

## Validation

`config.validate()` checks that critical fields like `hidden_size`,
`num_attention_heads`, and `vocab_size` have positive values and are
internally consistent (e.g. `head_dim` divides `hidden_size`).
