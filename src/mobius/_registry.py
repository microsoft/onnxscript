# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model registry mapping architecture names to module classes.

The registry is the central lookup table that maps HuggingFace model
architecture names (``config.model_type``) to the ``nn.Module`` subclass,
default task, and config class used to build the ONNX graph.
"""

from __future__ import annotations

__all__ = [
    "MODEL_MAP",
    "ModelRegistration",
    "ModelRegistry",
    "registry",
]

import dataclasses
import difflib

from onnxscript import nn

from mobius._configs import (
    BaseModelConfig,
    WhisperConfig,
)


@dataclasses.dataclass(frozen=True)
class ModelRegistration:
    """A single entry in the model registry.

    Attributes:
        module_class: The ``nn.Module`` subclass that builds the ONNX graph.
        task: Default task name (e.g. ``"text-generation"``).  When ``None``
            the task is read from ``module_class.default_task`` at resolution time.
        config_class: Config class for parsing HuggingFace configs.  When ``None``
            the class is read from ``module_class.config_class`` at resolution time.
        test_model_id: HuggingFace model ID used for L2 architecture validation.
            The config.json is downloaded (no weights) to verify that the real-size
            ONNX graph can be built.  ``None`` means no L2 test is defined.
        family: Dashboard family grouping (e.g. ``"phi"`` for phi3, phi3small,
            phimoe).  ``None`` means auto-derive from the model_type prefix.
        variant: Short label identifying the code-path variant (e.g. ``"mla"``,
            ``"moe"``, ``"sliding_window"``).  Used for dashboard display.
    """

    module_class: type[nn.Module]
    task: str | None = None
    config_class: type[BaseModelConfig] | None = None
    test_model_id: str | None = None
    family: str | None = None
    variant: str | None = None


class ModelRegistry:
    """Registry mapping architecture names to module classes, tasks, and configs.

    The registry is used by :func:`build` to auto-detect the module class,
    default task, and config class for a given HuggingFace model.  Users can
    register custom architectures::

        from mobius import registry

        # Simple (module class only — backward compatible)
        registry.register("my_arch", MyModelClass)

        # Full (module + task + config)
        registry.register(
            "my_arch", MyModelClass,
            task="text-generation",
            config_class=MyConfig,
        )
    """

    def __init__(self) -> None:
        self._map: dict[str, ModelRegistration] = {}

    def register(
        self,
        architecture: str,
        module_class: type[nn.Module],
        *,
        task: str | None = None,
        config_class: type[BaseModelConfig] | None = None,
        test_model_id: str | None = None,
        family: str | None = None,
        variant: str | None = None,
    ) -> None:
        """Register a module class for an architecture name.

        Args:
            architecture: The architecture name (matching HF ``config.model_type``).
            module_class: The module class to use for this architecture.
            task: Default task name for this architecture. When ``None``,
                the task is read from ``module_class.default_task``.
            config_class: Config class for this architecture. When ``None``,
                the class is read from ``module_class.config_class``.
            test_model_id: HuggingFace model ID for L2 architecture validation.
            family: Dashboard family grouping override.
            variant: Short label for the code-path variant.
        """
        self._map[architecture] = ModelRegistration(
            module_class,
            task,
            config_class,
            test_model_id,
            family,
            variant,
        )

    def get(self, architecture: str) -> type[nn.Module]:
        """Look up the module class for an architecture.

        Args:
            architecture: The architecture name.

        Returns:
            The registered module class.

        Raises:
            KeyError: If the architecture is not registered.
        """
        if architecture not in self._map:
            raise KeyError(self._not_found_message(architecture))
        return self._map[architecture].module_class

    def get_registration(self, architecture: str) -> ModelRegistration:
        """Look up the full registration entry for an architecture.

        Args:
            architecture: The architecture name.

        Returns:
            The :class:`ModelRegistration` entry.

        Raises:
            KeyError: If the architecture is not registered.
        """
        if architecture not in self._map:
            raise KeyError(self._not_found_message(architecture))
        return self._map[architecture]

    def _not_found_message(self, architecture: str) -> str:
        """Build a helpful error message for unknown architectures."""
        suggestions = difflib.get_close_matches(
            architecture, self._map.keys(), n=3, cutoff=0.6
        )
        msg = f"Unknown model_type '{architecture}'."
        if suggestions:
            quoted = ", ".join(f"'{s}'" for s in suggestions)
            msg += f" Did you mean: {quoted}?"
        msg += f" Use registry.register('{architecture}', YourModuleClass) to add it."
        return msg

    def get_task(self, architecture: str) -> str | None:
        """Return the registered default task, or ``None``."""
        return self._map[architecture].task if architecture in self._map else None

    def get_config_class(self, architecture: str) -> type[BaseModelConfig] | None:
        """Return the registered config class, or ``None``."""
        return self._map[architecture].config_class if architecture in self._map else None

    def __contains__(self, architecture: str) -> bool:
        return architecture in self._map

    def __len__(self) -> int:
        return len(self._map)

    def architectures(self) -> list[str]:
        """Return a sorted list of registered architecture names."""
        return sorted(self._map)


def _detect_fallback_registration(hf_config) -> ModelRegistration | None:
    """Detect a compatible model class for an unregistered model type.

    Analyzes a HuggingFace config to determine if the model is
    architecturally compatible with a built-in base class.  This enables
    automatic support for new Llama-like or MoE decoder-only models
    without explicit registration.

    Only returns a fallback when the config clearly indicates a standard
    causal-LM transformer.  Composite models (multimodal, speech),
    encoder-decoder, and encoder-only architectures return ``None``.

    Args:
        hf_config: A HuggingFace ``PretrainedConfig`` (or compatible object).

    Returns:
        A :class:`ModelRegistration` if a compatible fallback is found,
        or ``None`` otherwise.
    """
    # Reject encoder-decoder models — too varied for auto-fallback
    if getattr(hf_config, "is_encoder_decoder", False):
        return None

    # Reject composite models that need custom encoders/projectors
    if hasattr(hf_config, "vision_config") or hasattr(hf_config, "audio_config"):
        return None

    # Reject SSM/recurrent models — they have CausalLM in architectures
    # but use fundamentally different computation (not transformer attention)
    _ssm_indicators = (
        "d_state",
        "d_conv",
        "ssm_cfg",
        "recurrent_block_type",
        "rescale_every",  # RWKV linear-attention models
    )
    if any(getattr(hf_config, attr, None) is not None for attr in _ssm_indicators):
        return None

    # Check HF architectures field for causal LM indicator
    architectures = getattr(hf_config, "architectures", None) or []
    if not any("CausalLM" in arch for arch in architectures):
        return None

    # Require minimum config fields for graph construction
    required_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    if not all(getattr(hf_config, f, 0) > 0 for f in required_fields):
        return None

    from mobius.models import CausalLMModel, MoECausalLMModel

    # MoE detection: route to MoE class if experts are present
    num_experts = getattr(hf_config, "num_local_experts", 0)
    if num_experts and num_experts > 1:
        return ModelRegistration(MoECausalLMModel)

    return ModelRegistration(CausalLMModel)


def _create_default_registry() -> ModelRegistry:
    """Create the default registry with all built-in architectures."""
    from mobius.models import (
        CausalLMModel,
        ChatGLMCausalLMModel,
        DeepSeekOCR2CausalLMModel,
        DeepSeekV3CausalLMModel,
        ErnieCausalLMModel,
        Gemma2CausalLMModel,
        Gemma3CausalLMModel,
        Gemma3MultiModalModel,
        GemmaCausalLMModel,
        GPTOSSCausalLMModel,
        GraniteCausalLMModel,
        InternLM2CausalLMModel,
        MoECausalLMModel,
        NemotronCausalLMModel,
        OLMo2CausalLMModel,
        OLMoCausalLMModel,
        Phi3CausalLMModel,
        Phi3MoECausalLMModel,
        Phi3SmallCausalLMModel,
        Phi4MMMultiModalModel,
        PhiCausalLMModel,
        Qwen3CausalLMModel,
        Qwen3NextCausalLMModel,
        Qwen3VL3ModelCausalLMModel,
        Qwen3VLCausalLMModel,
        Qwen3VLTextModel,
        Qwen25VLCausalLMModel,
        Qwen25VLTextModel,
        Qwen35CausalLMModel,
        Qwen35MoECausalLMModel,
        Qwen35VL3ModelCausalLMModel,
        Qwen35VLTextModel,
        QwenCausalLMModel,
        SmolLM3CausalLMModel,
        WhisperForConditionalGeneration,
    )
    from mobius.models.bart import BartForConditionalGeneration
    from mobius.models.bert import BertModel
    from mobius.models.blip import BlipVisionModel
    from mobius.models.clip import CLIPTextModel, CLIPVisionModel
    from mobius.models.depth_anything import DepthAnythingForDepthEstimation
    from mobius.models.distilbert import DistilBertModel
    from mobius.models.falcon import FalconCausalLMModel
    from mobius.models.gemma3n import Gemma3nCausalLMModel
    from mobius.models.gpt2 import GPT2CausalLMModel
    from mobius.models.layoutlmv3 import LayoutLMv3Model
    from mobius.models.llava import LLaVAModel
    from mobius.models.mllama import MllamaCausalLMModel
    from mobius.models.modernbert import ModernBertDecoderModel, ModernBertModel
    from mobius.models.opt import OPTCausalLMModel
    from mobius.models.sam2 import Sam2VisionModel
    from mobius.models.segformer import SegformerForSemanticSegmentation
    from mobius.models.t5 import T5ForConditionalGeneration
    from mobius.models.trocr import TrOCRForConditionalGeneration
    from mobius.models.vit import ViTModel
    from mobius.models.wav2vec2 import Wav2Vec2Model
    from mobius.models.yolos import YolosForObjectDetection

    reg = ModelRegistry()

    # --- Text Generation (Llama-compatible) ---
    for name in (
        "apertus",
        "arcee",
        "baichuan",
        "code_llama",
        "codegen",
        "codegen2",
        "cohere",
        "cohere2",
        "command_r",
        "csm",
        "diffllama",
        "doge",
        "dots1",
        "evolla",
        "exaone",
        "exaone4",
        "glm",
        "glm4",
        "gpt_neox",
        "gpt_neox_japanese",
        "gptj",
        "helium",
        "hunyuan_v1_dense",
        "llama",
        "llama4_text",
        "longcat_flash",
        "minicpm",
        "minicpm3",
        "ministral",
        "ministral3",
        "mistral",
        "mistral3",
        "nanochat",
        "open-llama",
        "openelm",
        "persimmon",
        "qwen2",
        "seed_oss",
        "solar_open",
        "stablelm",
        "starcoder2",
        "yi",
        "youtu",
        "zamba",
        "zamba2",
    ):
        reg.register(name, CausalLMModel)

    # --- Text Generation (architecture-specific) ---
    for name, cls in {
        "bloom": FalconCausalLMModel,
        "chatglm": ChatGLMCausalLMModel,
        "ernie4_5": ErnieCausalLMModel,
        "falcon": FalconCausalLMModel,
        "falcon_h1": FalconCausalLMModel,
        "mpt": FalconCausalLMModel,
        "gemma": GemmaCausalLMModel,
        "gemma2": Gemma2CausalLMModel,
        "shieldgemma2": Gemma2CausalLMModel,
        "gemma3_text": Gemma3CausalLMModel,
        "gemma3n_text": Gemma3nCausalLMModel,
        "granite": GraniteCausalLMModel,
        "internlm2": InternLM2CausalLMModel,
        "modernbert-decoder": ModernBertDecoderModel,
        "nemotron": NemotronCausalLMModel,
        "olmo": OLMoCausalLMModel,
        "olmo2": OLMo2CausalLMModel,
        "olmo3": OLMo2CausalLMModel,
        "phi": PhiCausalLMModel,
        "phi3": Phi3CausalLMModel,
        "phi3small": Phi3SmallCausalLMModel,
        "qwen": QwenCausalLMModel,
        "qwen3": Qwen3CausalLMModel,
        "qwen3_5_text": Qwen35CausalLMModel,
        "smollm3": SmolLM3CausalLMModel,
    }.items():
        reg.register(name, cls)

    # --- Mixture of Experts ---
    for name in (
        "arctic",
        "dbrx",
        "ernie4_5_moe",
        "flex_olmo",
        "glm4_moe",
        "granitemoe",
        "granitemoehybrid",
        "granitemoeshared",
        "hunyuan_v1_moe",
        "jetmoe",
        "minimax",
        "mixtral",
        "olmoe",
        "qwen2_moe",
        "qwen3_moe",
        "qwen3_omni_moe",
        "qwen3_vl_moe",
    ):
        reg.register(name, MoECausalLMModel)
    reg.register("gptoss", GPTOSSCausalLMModel)
    reg.register("gpt_oss", GPTOSSCausalLMModel)
    reg.register("phimoe", Phi3MoECausalLMModel)
    reg.register("qwen3_5_moe", Qwen35MoECausalLMModel)
    reg.register("qwen3_next", Qwen3NextCausalLMModel)

    # --- DeepSeek (MLA + MoE) ---
    for name in (
        "deepseek_v2",
        "deepseek_v2_moe",
        "deepseek_v3",
    ):
        reg.register(name, DeepSeekV3CausalLMModel)

    # --- DeepSeek-OCR-2 (VL) ---
    reg.register("deepseek_vl_v2", DeepSeekOCR2CausalLMModel)

    # --- SSM (Mamba / Mamba2) ---
    from mobius.models.mamba import (
        Mamba2CausalLMModel,
        MambaCausalLMModel,
    )

    reg.register("mamba", MambaCausalLMModel)
    reg.register("falcon_mamba", MambaCausalLMModel)
    reg.register("mamba2", Mamba2CausalLMModel)

    # --- Hybrid SSM+Attention (Jamba) ---
    from mobius.models.jamba import JambaCausalLMModel

    reg.register("jamba", JambaCausalLMModel)

    # --- Hybrid Mamba2+Attention (Bamba) ---
    from mobius.models.bamba import BambaCausalLMModel

    reg.register("bamba", BambaCausalLMModel)

    # --- Multimodal ---
    for name in (
        "chameleon",
        "cohere2_vision",
        "florence2",
        "fuyu",
        "idefics2",
        "idefics3",
        "instructblip",
        "instructblipvideo",
        "llava",
        "llava_next",
        "llava_next_video",
        "llava_onevision",
        "molmo",
        "paligemma",
        "pixtral",
        "smolvlm",
        "video_llava",
        "aya_vision",
        "deepseek_vl",
        "deepseek_vl_hybrid",
        "got_ocr2",
        "janus",
        "ovis2",
        "vipllava",
    ):
        reg.register(name, LLaVAModel, task="vision-language")

    from mobius.models.internvl import InternVL2Model

    for name in ("internvl_chat", "internvl2", "internvl"):
        reg.register(name, InternVL2Model, task="vision-language")
    reg.register("gemma3", Gemma3CausalLMModel)
    reg.register("gemma3_multimodal", Gemma3MultiModalModel, task="vision-language")
    reg.register("gemma3n", Gemma3nCausalLMModel)
    reg.register("glm4v", LLaVAModel, task="vision-language")
    reg.register("glm4v_moe", LLaVAModel, task="vision-language")
    reg.register("glm4v_moe_text", MoECausalLMModel)
    reg.register("glm4v_text", CausalLMModel)
    reg.register("mllama", MllamaCausalLMModel, task="mllama-vision-language")

    from mobius.models.blip2 import Blip2Model

    reg.register("blip-2", Blip2Model, task="vision-language")
    reg.register("phi4mm", Phi4MMMultiModalModel, task="phi4mm-multimodal")
    reg.register("phi4_multimodal", Phi4MMMultiModalModel, task="phi4mm-multimodal")
    reg.register("qwen2_vl", Qwen25VLCausalLMModel, task="qwen-vl")
    reg.register("qwen2_vl_text", Qwen25VLTextModel)
    reg.register("qwen2_5_vl", Qwen25VLCausalLMModel, task="qwen-vl")
    reg.register("qwen2_5_vl_text", Qwen25VLTextModel)
    reg.register("qwen3_vl", Qwen3VL3ModelCausalLMModel, task="qwen-vl")
    reg.register("qwen3_vl_single", Qwen3VLCausalLMModel, task="qwen3-vl-vision-language")
    reg.register("qwen3_vl_text", Qwen3VLTextModel)
    reg.register("qwen3_5", Qwen35VL3ModelCausalLMModel, task="hybrid-qwen-vl")
    reg.register("qwen3_5_vl", Qwen35VL3ModelCausalLMModel, task="hybrid-qwen-vl")
    reg.register("qwen3_5_vl_text", Qwen35VLTextModel)

    # --- Speech ---
    reg.register(
        "whisper",
        WhisperForConditionalGeneration,
        task="speech-to-text",
        config_class=WhisperConfig,
    )

    from mobius.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    reg.register("qwen3_asr", Qwen3ASRForConditionalGeneration, task="speech-language")
    reg.register(
        "qwen3_forced_aligner", Qwen3ASRForConditionalGeneration, task="speech-language"
    )

    from mobius.models.qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    reg.register("qwen3_tts", Qwen3TTSForConditionalGeneration)

    from mobius.models.qwen3_tts_tokenizer import (
        Qwen3TTSTokenizerV2Model,
    )

    reg.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Model, task="codec")

    # --- Encoder-only ---
    for name in (
        "albert",
        "bert",
        "camembert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "electra",
        "ernie",
        "ernie_m",
        "esm",
        "flaubert",
        "ibert",
        "megatron-bert",
        "mobilebert",
        "mpnet",
        "nezha",
        "qdqbert",
        "rembert",
        "roberta",
        "roberta-prelayernorm",
        "roc_bert",
        "roformer",
        "splinter",
        "squeezebert",
        "xlm-roberta",
        "xlm-roberta-xl",
        "xlnet",
        "xmod",
        # Document/layout models (BERT-like encoder)
        "bros",
        "layoutlm",
        "layoutlmv2",
        "lilt",
        "markuplm",
        "mega",
        "mra",
        "nystromformer",
        "yoso",
    ):
        reg.register(name, BertModel, task="feature-extraction")
    reg.register("distilbert", DistilBertModel, task="feature-extraction")
    reg.register("clip_text_model", CLIPTextModel, task="feature-extraction")
    reg.register("layoutlmv3", LayoutLMv3Model, task="feature-extraction")
    reg.register("modernbert", ModernBertModel, task="feature-extraction")

    # --- Absolute positional embeddings (non-RoPE) ---
    reg.register("gpt2", GPT2CausalLMModel)
    for name in (
        "biogpt",
        "ctrl",
        "gpt-sw3",
        "gpt_bigcode",
        "gpt_neo",
        "imagegpt",
        "openai-gpt",
        "xglm",
        "xlm",
    ):
        reg.register(name, GPT2CausalLMModel)
    reg.register("opt", OPTCausalLMModel)

    # --- Encoder-decoder ---
    for name in (
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "fsmt",
        "led",
        "m2m_100",
        "marian",
        "mbart",
        "mvp",
        "nllb-moe",
        "nllb_moe",
        "pegasus",
        "pegasus_x",
        "plbart",
        "prophetnet",
        "xlm-prophetnet",
    ):
        reg.register(name, BartForConditionalGeneration, task="seq2seq")
    for name in ("longt5", "mt5", "switch_transformers", "t5", "umt5"):
        reg.register(name, T5ForConditionalGeneration, task="seq2seq")
    reg.register("trocr", TrOCRForConditionalGeneration, task="seq2seq")

    # --- Vision ---
    reg.register("blip", BlipVisionModel, task="image-classification")
    reg.register(
        "depth_anything", DepthAnythingForDepthEstimation, task="image-classification"
    )
    for name in (
        "beit",
        "cvt",
        "data2vec-vision",
        "deit",
        "dinov2",
        "dinov2_with_registers",
        "dinov3_vit",
        "hiera",
        "ijepa",
        "mobilevit",
        "mobilevitv2",
        "pvt",
        "pvt_v2",
        "swin",
        "swin2sr",
        "swinv2",
        "vit",
        "vit_hybrid",
        "vit_mae",
        "vit_msn",
    ):
        reg.register(name, ViTModel, task="image-classification")
    for name in ("clip_vision_model", "siglip_vision_model", "siglip2_vision_model"):
        reg.register(name, CLIPVisionModel, task="image-classification")

    # --- Object detection ---
    reg.register("yolos", YolosForObjectDetection, task="object-detection")

    # --- Segmentation ---
    reg.register("sam2", Sam2VisionModel, task="image-classification")
    reg.register("segformer", SegformerForSemanticSegmentation, task="image-classification")

    # --- Audio ---
    for name in (
        "data2vec-audio",
        "hubert",
        "mctct",
        "musicgen",
        "seamless_m4t",
        "seamless_m4t_v2",
        "sew",
        "sew-d",
        "speecht5",
        "unispeech",
        "unispeech-sat",
        "voxtral_encoder",
        "wav2vec2",
        "wav2vec2-bert",
        "wav2vec2-conformer",
        "wavlm",
    ):
        reg.register(name, Wav2Vec2Model, task="audio-feature-extraction")

    # -----------------------------------------------------------------
    # Test metadata — applied after all registrations
    #
    # test_model_id: HF model whose config.json is used for L2
    #     architecture validation (full-size graph, no weights).
    # family: Groups related model_types in the dashboard.
    # variant: Labels the code-path variant for dashboard display.
    # -----------------------------------------------------------------
    _apply_test_metadata(reg)

    return reg


# fmt: off
# -- Test model IDs for L2 architecture validation --
# Each maps a registered model_type to a public HuggingFace model.
# Only the config.json is downloaded (no weights).
_TEST_MODEL_IDS: dict[str, str] = {
    # --- CausalLM (Llama-compatible) ---
    "llama": "meta-llama/Llama-3.2-1B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "cohere": "CohereForAI/c4ai-command-r7b-12-2024",
    "cohere2": "CohereForAI/c4ai-command-r7b-12-2024",
    "exaone": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    "glm": "THUDM/glm-4-9b-chat-hf",
    "glm4": "THUDM/glm-4-9b-chat-hf",
    "gpt_neox": "EleutherAI/pythia-70m",
    "gptj": "EleutherAI/gpt-j-6b",
    "stablelm": "stabilityai/stablelm-2-1_6b",
    "starcoder2": "bigcode/starcoder2-3b",
    "yi": "01-ai/Yi-6B",
    "code_llama": "meta-llama/CodeLlama-7b-hf",
    "llama4_text": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",

    # --- CausalLM (architecture-specific) ---
    "falcon": "tiiuae/falcon-7b",
    "bloom": "bigscience/bloom-560m",
    "gemma": "google/gemma-2b",
    "gemma2": "google/gemma-2-2b",
    "gemma3": "google/gemma-3-1b-pt",
    "gemma3_text": "google/gemma-3-1b-pt",
    "gemma3n": "google/gemma-3n-E2B-pt",
    "gemma3n_text": "google/gemma-3n-E2B-pt",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
    "internlm2": "internlm/internlm2_5-7b-chat",
    "nemotron": "nvidia/Nemotron-Mini-4B-Instruct",
    "olmo": "allenai/OLMo-1B-hf",
    "olmo2": "allenai/OLMo-2-1124-7B",
    "phi": "microsoft/phi-1_5",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "phi3small": "microsoft/Phi-3-small-8k-instruct",
    "qwen": "Qwen/Qwen-1_8B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_5_text": "Qwen/Qwen3.5-VL-3B-Instruct",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "gpt2": "openai-community/gpt2",
    "opt": "facebook/opt-125m",
    "mpt": "mosaicml/mpt-7b",

    # --- Mixture of Experts ---
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "phimoe": "microsoft/Phi-tiny-MoE-instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_5_moe": "Qwen/Qwen3.5-MoE-A3B-128K",
    "qwen3_next": "Qwen/Qwen3-235B-A22B",
    "granitemoe": "ibm-granite/granite-3.0-1b-a400m-instruct",
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "dbrx": "databricks/dbrx-instruct",
    "arctic": "Snowflake/snowflake-arctic-instruct",
    "jetmoe": "jetmoe/jetmoe-8b",

    # --- DeepSeek (MLA + MoE) ---
    "deepseek_v2": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",

    # --- SSM (Mamba) ---
    "mamba": "state-spaces/mamba-130m-hf",
    "mamba2": "state-spaces/mamba2-130m",
    "falcon_mamba": "tiiuae/falcon-mamba-7b",

    # --- Hybrid SSM+Attention ---
    "jamba": "ai21labs/Jamba-v0.1",
    "bamba": "ibm-fms/Bamba-9B",

    # --- Multimodal ---
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_text": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_text": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_text": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_5": "Qwen/Qwen3.5-VL-3B-Instruct",
    "qwen3_5_vl": "Qwen/Qwen3.5-VL-3B-Instruct",
    "qwen3_5_vl_text": "Qwen/Qwen3.5-VL-3B-Instruct",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "mllama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "gemma3_multimodal": "google/gemma-3-4b-it",
    "internvl2": "OpenGVLab/InternVL2-1B",
    "phi4mm": "microsoft/Phi-4-multimodal-instruct",
    "phi4_multimodal": "microsoft/Phi-4-multimodal-instruct",
    "blip-2": "Salesforce/blip2-opt-2.7b",

    # --- Speech ---
    "whisper": "openai/whisper-tiny",
    "qwen3_asr": "Qwen/Qwen3-ASR-2B-Instruct",

    # --- Encoder-only ---
    "bert": "google-bert/bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "albert": "albert/albert-base-v2",
    "electra": "google/electra-small-generator",
    "deberta": "microsoft/deberta-base",
    "deberta-v2": "microsoft/deberta-v3-base",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "clip_text_model": "openai/clip-vit-base-patch32",

    # --- Encoder-decoder ---
    "bart": "facebook/bart-base",
    "t5": "google-t5/t5-small",
    "mt5": "google/mt5-small",
    "marian": "Helsinki-NLP/opus-mt-en-de",
    "mbart": "facebook/mbart-large-cc25",
    "pegasus": "google/pegasus-xsum",
    "trocr": "microsoft/trocr-small-handwritten",

    # --- Vision ---
    "vit": "google/vit-base-patch16-224",
    "dinov2": "facebook/dinov2-small",
    "beit": "microsoft/beit-base-patch16-224",
    "clip_vision_model": "openai/clip-vit-base-patch32",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    "deit": "facebook/deit-small-patch16-224",
    "blip": "Salesforce/blip-image-captioning-base",
    "depth_anything": "LiheYoung/depth-anything-small-hf",
    "yolos": "hustvl/yolos-tiny",
    "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",

    # --- Audio ---
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wav2vec2-conformer": "facebook/wav2vec2-conformer-rope-large-960h-ft",
}
# fmt: on

# -- Family overrides for dashboard grouping --
# Models that share an architecture family but have different model_type prefixes.
_FAMILY_OVERRIDES: dict[str, str] = {
    "phi": "phi",
    "phi3": "phi",
    "phi3small": "phi",
    "phimoe": "phi",
    "phi4mm": "phi",
    "phi4_multimodal": "phi",
    "gemma": "gemma",
    "gemma2": "gemma",
    "shieldgemma2": "gemma",
    "gemma3": "gemma",
    "gemma3_text": "gemma",
    "gemma3n": "gemma",
    "gemma3n_text": "gemma",
    "gemma3_multimodal": "gemma",
    "internlm2": "internlm",
    "internvl_chat": "internlm",
    "internvl2": "internlm",
    "internvl": "internlm",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
    "qwen3_5_text": "qwen",
    "qwen3_5_moe": "qwen",
    "qwen3_next": "qwen",
    "qwen2_vl": "qwen",
    "qwen2_vl_text": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen2_5_vl_text": "qwen",
    "qwen3_vl": "qwen",
    "qwen3_vl_text": "qwen",
    "qwen3_vl_single": "qwen",
    "qwen3_vl_moe": "qwen",
    "qwen3_5": "qwen",
    "qwen3_5_vl": "qwen",
    "qwen3_5_vl_text": "qwen",
    "qwen3_omni_moe": "qwen",
    "qwen3_asr": "qwen",
    "qwen3_forced_aligner": "qwen",
    "qwen3_tts": "qwen",
    "qwen3_tts_tokenizer_12hz": "qwen",
    "deepseek_v2": "deepseek",
    "deepseek_v2_moe": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_vl_v2": "deepseek",
    "olmo": "olmo",
    "olmo2": "olmo",
    "olmo3": "olmo",
    "olmoe": "olmo",
    "llama": "llama",
    "code_llama": "llama",
    "llama4_text": "llama",
    "mllama": "llama",
    "mistral": "mistral",
    "mistral3": "mistral",
    "ministral": "mistral",
    "ministral3": "mistral",
    "mixtral": "mistral",
    "pixtral": "mistral",
    "falcon": "falcon",
    "falcon_h1": "falcon",
    "falcon_mamba": "falcon",
    "mamba": "mamba",
    "mamba2": "mamba",
    "bloom": "bloom",
    "gpt2": "gpt2",
    "gpt_neo": "gpt2",
    "gpt_bigcode": "gpt2",
    "gpt_neox": "gpt_neox",
    "gpt_neox_japanese": "gpt_neox",
    "bart": "bart",
    "mbart": "bart",
    "t5": "t5",
    "mt5": "t5",
    "longt5": "t5",
    "umt5": "t5",
    "switch_transformers": "t5",
    "bert": "bert",
    "albert": "bert",
    "roberta": "bert",
    "xlm-roberta": "bert",
    "xlm-roberta-xl": "bert",
    "distilbert": "bert",
    "deberta": "deberta",
    "deberta-v2": "deberta",
    "wav2vec2": "wav2vec2",
    "wav2vec2-bert": "wav2vec2",
    "wav2vec2-conformer": "wav2vec2",
    "hubert": "wav2vec2",
    "wavlm": "wav2vec2",
    "vit": "vit",
    "vit_hybrid": "vit",
    "vit_mae": "vit",
    "vit_msn": "vit",
    "deit": "vit",
    "beit": "vit",
    "dinov2": "vit",
    "dinov2_with_registers": "vit",
    "swin": "swin",
    "swin2sr": "swin",
    "swinv2": "swin",
    "clip_text_model": "clip",
    "clip_vision_model": "clip",
    "siglip_vision_model": "clip",
    "siglip2_vision_model": "clip",
}

# -- Variant labels for code-path identification --
_VARIANT_LABELS: dict[str, str] = {
    "deepseek_v2": "mla",
    "deepseek_v3": "mla+moe",
    "deepseek_v2_moe": "mla+moe",
    "phi3small": "blocksparse",
    "falcon_h1": "hybrid-ssm",
    "mamba": "ssm",
    "mamba2": "ssm",
    "falcon_mamba": "ssm",
    "jamba": "hybrid-ssm+attn",
    "bamba": "hybrid-mamba2+attn",
    "qwen3_next": "moe+linear-attn",
}


def _apply_test_metadata(reg: ModelRegistry) -> None:
    """Apply test_model_id, family, and variant metadata to registrations.

    Called at the end of ``_create_default_registry()`` to attach test
    metadata without disturbing the existing registration patterns.
    """
    for arch, model_id in _TEST_MODEL_IDS.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, test_model_id=model_id)
    for arch, family in _FAMILY_OVERRIDES.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, family=family)
    for arch, variant in _VARIANT_LABELS.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, variant=variant)


#: The default model registry with all built-in architectures.
registry: ModelRegistry = _create_default_registry()

# Backward-compatible alias — exposes internal dict directly.
# Deprecated: use registry.get() / registry.register() instead.
# This will be removed in a future version.
MODEL_MAP = registry._map
