# Model Catalog

**mobius** supports 273 registered model types across 10 categories.
This catalog lists every supported architecture with its module class, task type,
and example HuggingFace model IDs.

> **Auto-fallback**: Models not explicitly registered but architecturally
> compatible with Llama (standard CausalLM transformers) are automatically
> supported via fallback detection. MoE variants are also auto-detected.

## Decoder-Only LLMs (RoPE)

Standard autoregressive language models using rotary position embeddings.

| Model Type | Module Class | Example HuggingFace Model |
|---|---|---|
| `llama` | `CausalLMModel` | `meta-llama/Llama-3.2-1B` |
| `mistral` | `CausalLMModel` | `mistralai/Mistral-7B-v0.3` |
| `qwen2` | `CausalLMModel` | `Qwen/Qwen2.5-7B` |
| `qwen3` | `Qwen3CausalLMModel` | `Qwen/Qwen3-8B` |
| `gemma` | `GemmaCausalLMModel` | `google/gemma-7b` |
| `gemma2` | `Gemma2CausalLMModel` | `google/gemma-2-9b` |
| `gemma3` | `Gemma3CausalLMModel` | `google/gemma-3-4b-pt` |
| `gemma3_text` | `Gemma3CausalLMModel` | `google/gemma-3-4b-pt` |
| `gemma3n` | `Gemma3nCausalLMModel` | — |
| `gemma3n_text` | `Gemma3nCausalLMModel` | — |
| `phi` | `PhiCausalLMModel` | `microsoft/phi-2` |
| `phi3` | `Phi3CausalLMModel` | `microsoft/Phi-3-mini-4k-instruct` |
| `phi3small` | `Phi3SmallCausalLMModel` | `microsoft/Phi-3-small-8k-instruct` |
| `smollm3` | `SmolLM3CausalLMModel` | `HuggingFaceTB/SmolLM3-3B` |
| `qwen3_5_text` | `Qwen35CausalLMModel` | `Qwen/Qwen3.5-VL-32B` |
| `falcon` | `FalconCausalLMModel` | `tiiuae/falcon-7b` |
| `bloom` | `FalconCausalLMModel` | `bigscience/bloom-560m` |
| `mpt` | `FalconCausalLMModel` | `mosaicml/mpt-7b` |
| `chatglm` | `ChatGLMCausalLMModel` | `THUDM/chatglm3-6b` |
| `internlm2` | `InternLM2CausalLMModel` | `internlm/internlm2-7b` |
| `granite` | `GraniteCausalLMModel` | `ibm-granite/granite-3b-code-base` |
| `nemotron` | `NemotronCausalLMModel` | `nvidia/Nemotron-4-340B-Base` |
| `olmo` | `OLMoCausalLMModel` | `allenai/OLMo-7B` |
| `olmo2` | `OLMo2CausalLMModel` | `allenai/OLMo-2-7B` |
| `olmo3` | `OLMo2CausalLMModel` | `allenai/OLMo-3-8B` |
| `ernie4_5` | `ErnieCausalLMModel` | — |
| `cohere` | `CausalLMModel` | `CohereForAI/c4ai-command-r-v01` |
| `cohere2` | `CausalLMModel` | `CohereForAI/c4ai-command-r7b-12-2024` |
| `modernbert-decoder` | `ModernBertDecoderModel` | — |

Also registered with `CausalLMModel`: `apertus`, `arcee`, `baichuan`,
`code_llama`, `codegen`, `codegen2`, `command_r`, `csm`, `diffllama`,
`doge`, `dots1`, `evolla`, `exaone`, `exaone4`, `glm`, `glm4`,
`gpt_neox`, `gpt_neox_japanese`, `gptj`, `helium`, `hunyuan_v1_dense`,
`llama4_text`, `longcat_flash`, `minicpm`, `minicpm3`, `ministral`,
`ministral3`, `mistral3`, `nanochat`, `open-llama`, `openelm`,
`persimmon`, `seed_oss`, `solar_open`, `stablelm`, `starcoder2`, `yi`,
`youtu`, `zamba`, `zamba2`.

## Decoder-Only LLMs (Absolute Position Embeddings)

GPT-2 style models using learned absolute positional embeddings.

| Model Type | Module Class | Example HuggingFace Model |
|---|---|---|
| `gpt2` | `GPT2CausalLMModel` | `openai-community/gpt2` |
| `opt` | `OPTCausalLMModel` | `facebook/opt-1.3b` |

Also registered with `GPT2CausalLMModel`: `biogpt`, `ctrl`, `gpt-sw3`,
`gpt_bigcode`, `gpt_neo`, `imagegpt`, `openai-gpt`, `xglm`, `xlm`.

## Mixture-of-Experts (MoE)

Models that route tokens to a subset of expert MLPs.

| Model Type | Module Class | Example HuggingFace Model |
|---|---|---|
| `mixtral` | `MoECausalLMModel` | `mistralai/Mixtral-8x7B-v0.1` |
| `qwen2_moe` | `MoECausalLMModel` | `Qwen/Qwen1.5-MoE-A2.7B` |
| `qwen3_moe` | `MoECausalLMModel` | `Qwen/Qwen3-30B-A3B` |
| `qwen3_5_moe` | `Qwen35MoECausalLMModel` | — |
| `qwen3_next` | `Qwen3NextCausalLMModel` | — |
| `deepseek_v2` | `DeepSeekV3CausalLMModel` | `deepseek-ai/DeepSeek-V2-Lite` |
| `deepseek_v3` | `DeepSeekV3CausalLMModel` | `deepseek-ai/DeepSeek-V3` |
| `phimoe` | `Phi3MoECausalLMModel` | `microsoft/Phi-3.5-MoE-instruct` |
| `gptoss` | `GPTOSSCausalLMModel` | — |

Also registered with `MoECausalLMModel`: `arctic`, `dbrx`,
`ernie4_5_moe`, `flex_olmo`, `glm4_moe`, `granitemoe`,
`granitemoehybrid`, `granitemoeshared`, `hunyuan_v1_moe`, `jetmoe`,
`minimax`, `olmoe`, `qwen3_omni_moe`, `qwen3_vl_moe`.

## SSM / State-Space Models

Mamba and Mamba2 architectures using selective state-space layers.

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `mamba` | `MambaCausalLMModel` | `ssm-text-generation` | `state-spaces/mamba-2.8b` |
| `falcon_mamba` | `MambaCausalLMModel` | `ssm-text-generation` | `tiiuae/falcon-mamba-7b` |
| `mamba2` | `Mamba2CausalLMModel` | `ssm2-text-generation` | `state-spaces/mamba2-2.7b` |

## Hybrid SSM+Attention

Models combining Mamba/SSM layers with transformer attention layers.

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `jamba` | `JambaCausalLMModel` | `hybrid-text-generation` | `ai21labs/Jamba-v0.1` |
| `bamba` | `BambaCausalLMModel` | `hybrid-text-generation` | `ibm-fms/Bamba-9B` |

## Vision-Language (Multimodal)

Models that process both images and text.

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `llava` | `LLaVAModel` | `vision-language` | `llava-hf/llava-1.5-7b-hf` |
| `llava_next` | `LLaVAModel` | `vision-language` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| `qwen2_vl` | `Qwen25VLCausalLMModel` | `qwen-vl` | `Qwen/Qwen2-VL-7B-Instruct` |
| `qwen2_5_vl` | `Qwen25VLCausalLMModel` | `qwen-vl` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen3_vl` | `Qwen3VL3ModelCausalLMModel` | `qwen-vl` | `Qwen/Qwen3-VL-2B-Instruct` |
| `qwen3_5_vl` | `Qwen35VL3ModelCausalLMModel` | `hybrid-qwen-vl` | — |
| `gemma3_multimodal` | `Gemma3MultiModalModel` | `vision-language` | `google/gemma-3-4b-it` |
| `mllama` | `MllamaCausalLMModel` | `mllama-vision-language` | `meta-llama/Llama-3.2-11B-Vision` |
| `phi4mm` | `Phi4MMMultiModalModel` | `multimodal` | `microsoft/Phi-4-multimodal-instruct` |
| `phi4_multimodal` | `Phi4MMMultiModalModel` | `multimodal` | `microsoft/Phi-4-multimodal-instruct` |
| `blip-2` | `Blip2Model` | `vision-language` | `Salesforce/blip2-opt-2.7b` |
| `internvl2` | `InternVL2Model` | `vision-language` | `OpenGVLab/InternVL2-8B` |
| `deepseek_vl_v2` | `DeepSeekOCR2CausalLMModel` | `vision-language` | — |

Also registered with `LLaVAModel` (task: `vision-language`):
`aya_vision`, `chameleon`, `cohere2_vision`, `deepseek_vl`,
`deepseek_vl_hybrid`, `florence2`, `fuyu`, `glm4v`, `glm4v_moe`,
`got_ocr2`, `idefics2`, `idefics3`, `instructblip`,
`instructblipvideo`, `janus`, `llava_next_video`, `llava_onevision`,
`molmo`, `ovis2`, `paligemma`, `pixtral`, `smolvlm`, `video_llava`,
`vipllava`.

## Encoder-Only (BERT Family)

Models for text embeddings, classification, and feature extraction.

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `bert` | `BertModel` | `feature-extraction` | `google-bert/bert-base-uncased` |
| `roberta` | `BertModel` | `feature-extraction` | `FacebookAI/roberta-base` |
| `distilbert` | `DistilBertModel` | `feature-extraction` | `distilbert/distilbert-base-uncased` |
| `modernbert` | `ModernBertModel` | `feature-extraction` | `answerdotai/ModernBERT-base` |
| `layoutlmv3` | `LayoutLMv3Model` | `feature-extraction` | `microsoft/layoutlmv3-base` |
| `clip_text_model` | `CLIPTextModel` | `feature-extraction` | — |

Also registered with `BertModel` (task: `feature-extraction`):
`albert`, `camembert`, `data2vec-text`, `deberta`, `deberta-v2`,
`electra`, `ernie`, `ernie_m`, `esm`, `flaubert`, `ibert`,
`megatron-bert`, `mobilebert`, `mpnet`, `nezha`, `qdqbert`, `rembert`,
`roberta-prelayernorm`, `roc_bert`, `roformer`, `splinter`,
`squeezebert`, `xlm-roberta`, `xlm-roberta-xl`, `xlnet`, `xmod`,
`bros`, `layoutlm`, `layoutlmv2`, `lilt`, `markuplm`, `mega`, `mra`,
`nystromformer`, `yoso`.

## Encoder-Decoder (Seq2Seq)

Sequence-to-sequence models for translation, summarization, etc.

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `bart` | `BartForConditionalGeneration` | `seq2seq` | `facebook/bart-large` |
| `t5` | `T5ForConditionalGeneration` | `seq2seq` | `google-t5/t5-small` |
| `mt5` | `T5ForConditionalGeneration` | `seq2seq` | `google/mt5-small` |
| `mbart` | `BartForConditionalGeneration` | `seq2seq` | `facebook/mbart-large-50` |
| `trocr` | `TrOCRForConditionalGeneration` | `seq2seq` | `microsoft/trocr-base-handwritten` |

Also registered with `BartForConditionalGeneration` (task: `seq2seq`):
`bigbird_pegasus`, `blenderbot`, `blenderbot-small`, `fsmt`, `led`,
`m2m_100`, `marian`, `mvp`, `nllb-moe`, `nllb_moe`, `pegasus`,
`pegasus_x`, `plbart`, `prophetnet`, `xlm-prophetnet`.

Also registered with `T5ForConditionalGeneration` (task: `seq2seq`):
`longt5`, `switch_transformers`, `umt5`.

## Speech & Audio

### Speech-to-Text

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `whisper` | `WhisperForConditionalGeneration` | `speech-to-text` | `openai/whisper-tiny` |
| `qwen3_asr` | `Qwen3ASRForConditionalGeneration` | `speech-language` | — |
| `qwen3_forced_aligner` | `Qwen3ASRForConditionalGeneration` | `speech-language` | — |

### Text-to-Speech

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `qwen3_tts` | `Qwen3TTSForConditionalGeneration` | `tts` | — |
| `qwen3_tts_tokenizer_12hz` | `Qwen3TTSTokenizerV2Model` | `codec` | — |

### Audio Feature Extraction

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `wav2vec2` | `Wav2Vec2Model` | `audio-feature-extraction` | `facebook/wav2vec2-base` |
| `hubert` | `Wav2Vec2Model` | `audio-feature-extraction` | `facebook/hubert-base-ls960` |
| `wavlm` | `Wav2Vec2Model` | `audio-feature-extraction` | `microsoft/wavlm-base` |

Also registered with `Wav2Vec2Model`: `data2vec-audio`, `mctct`,
`musicgen`, `seamless_m4t`, `seamless_m4t_v2`, `sew`, `sew-d`,
`speecht5`, `unispeech`, `unispeech-sat`, `voxtral_encoder`,
`wav2vec2-bert`, `wav2vec2-conformer`.

## Vision

### Image Classification & Feature Extraction

| Model Type | Module Class | Task | Example HuggingFace Model |
|---|---|---|---|
| `vit` | `ViTModel` | `image-classification` | `google/vit-base-patch16-224` |
| `clip_vision_model` | `CLIPVisionModel` | `image-classification` | — |
| `siglip_vision_model` | `CLIPVisionModel` | `image-classification` | — |
| `blip` | `BlipVisionModel` | `image-classification` | — |
| `depth_anything` | `DepthAnythingForDepthEstimation` | `image-classification` | `LiheYoung/depth-anything-base-hf` |
| `sam2` | `Sam2VisionModel` | `image-classification` | `facebook/sam2-hiera-large` |
| `segformer` | `SegformerForSemanticSegmentation` | `image-classification` | `nvidia/segformer-b0-finetuned-ade-512-512` |
| `yolos` | `YolosForObjectDetection` | `object-detection` | `hustvl/yolos-tiny` |

Also registered with `ViTModel`: `beit`, `cvt`, `data2vec-vision`,
`deit`, `dinov2`, `dinov2_with_registers`, `dinov3_vit`, `hiera`,
`ijepa`, `mobilevit`, `mobilevitv2`, `pvt`, `pvt_v2`, `swin`,
`swin2sr`, `swinv2`, `vit_hybrid`, `vit_mae`, `vit_msn`.

## Diffusion (via Diffusers)

Diffusion models are built through the `build_diffusers_pipeline()` API
which auto-detects the pipeline type from HuggingFace. These are not
registered in the model registry but are supported as diffusers pipeline
components.

Supported component classes include:

- `UNet2DConditionModel` — Stable Diffusion, SDXL UNet
- `AutoencoderKLModel` — VAE encoder/decoder
- `FluxTransformer2DModel` — Flux transformer
- `SD3Transformer2DModel` — Stable Diffusion 3 transformer
- `DiTTransformer2DModel` — DiT (Diffusion Transformer)
- `HunyuanDiT2DModel` — Hunyuan-DiT
- `ControlNetModel` — ControlNet conditioning
- `CogVideoXTransformer3DModel` — CogVideoX
- `VideoAutoencoderModel` — Video VAE
- `QwenImageTransformer2DModel` — Qwen image generation
- `AutoencoderKLQwenImageModel` — Qwen image VAE

```python
from mobius import build

# Auto-detect diffusers pipeline
pkg = build("stabilityai/stable-diffusion-xl-base-1.0")
```

## Quantization Support

All decoder-only LLMs and MoE models support quantized weight loading:

| Format | Source | How |
|---|---|---|
| **GPTQ** | HuggingFace (e.g. `-GPTQ` suffix models) | `build("TheBloke/Llama-2-7B-GPTQ")` |
| **AWQ** | HuggingFace (e.g. `-AWQ` suffix models) | `build("TheBloke/Llama-2-7B-AWQ")` |
| **GGUF** | Local `.gguf` files | `build_from_gguf("model.gguf")` |

GGUF support (Phase 1) dequantizes weights to float before building the
ONNX graph. Phase 2 (`--keep-quantized`) preserves quantization using
MatMulNBits ops.

## Summary

| Category | Count | Primary Classes |
|---|---|---|
| Decoder-only LLMs | ~100 | `CausalLMModel`, `GPT2CausalLMModel` |
| Mixture of Experts | ~25 | `MoECausalLMModel`, `DeepSeekV3CausalLMModel` |
| SSM / Hybrid | 5 | `MambaCausalLMModel`, `JambaCausalLMModel` |
| Vision-Language | ~40 | `LLaVAModel`, `Qwen25VLCausalLMModel` |
| Encoder-only | ~40 | `BertModel`, `DistilBertModel` |
| Encoder-decoder | ~20 | `BartForConditionalGeneration`, `T5ForConditionalGeneration` |
| Speech & Audio | ~20 | `WhisperForConditionalGeneration`, `Wav2Vec2Model` |
| Vision | ~25 | `ViTModel`, `CLIPVisionModel` |
| Diffusion | ~10 | `UNet2DConditionModel`, `FluxTransformer2DModel` |
| **Total** | **~273** | |
