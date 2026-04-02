# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model tasks that define graph I/O structure for different use cases.

A ModelTask encapsulates how to wire an onnxscript.nn.Module into an ONNX
graph: what inputs to create, how to invoke the module, and how to name
the outputs. Different tasks produce models with different I/O contracts.

Example::

    from mobius.tasks import CausalLMTask
    from mobius import build_from_module

    task = CausalLMTask()
    model = build_from_module(my_module, config, task=task)
"""

from __future__ import annotations

__all__ = [
    "AdapterTask",
    "AudioFeatureExtractionTask",
    "CausalLMTask",
    "ClapAudioFeatureExtractionTask",
    "CodecTask",
    "ContrastiveTask",
    "ControlNetTask",
    "DenoisingTask",
    "FeatureExtractionTask",
    "HybridCausalLMTask",
    "HybridQwenVLTask",
    "ImageClassificationTask",
    "ImageSegmentationTask",
    "ModelTask",
    "MllamaVisionLanguageTask",
    "MultiModalTask",
    "OPSET_VERSION",
    "ObjectDetectionTask",
    "Owlv2ObjectDetectionTask",
    "Phi4MMMultiModalTask",
    "Qwen3VLVisionLanguageTask",
    "QwenImageVAETask",
    "QwenVLTask",
    "RwkvCausalLMTask",
    "Rwkv6CausalLMTask",
    "Speech2TextSeq2SeqTask",
    "SSM2CausalLMTask",
    "SSMCausalLMTask",
    "Seq2SeqTask",
    "SpeechLanguageTask",
    "SpeechToTextTask",
    "TASK_REGISTRY",
    "TTSTask",
    "VAETask",
    "VideoDenoisingTask",
    "VisionLanguageTask",
    "get_task",
]

from mobius._constants import OPSET_VERSION
from mobius.tasks._adapter import AdapterTask
from mobius.tasks._audio_feature_extraction import (
    AudioFeatureExtractionTask,
    ClapAudioFeatureExtractionTask,
)
from mobius.tasks._base import ModelTask
from mobius.tasks._causal_lm import (
    CausalLMTask,
    HybridCausalLMTask,
)
from mobius.tasks._codec import CodecTask
from mobius.tasks._contrastive import ContrastiveTask
from mobius.tasks._controlnet import ControlNetTask
from mobius.tasks._denoising import DenoisingTask
from mobius.tasks._feature_extraction import FeatureExtractionTask
from mobius.tasks._image_classification import ImageClassificationTask
from mobius.tasks._image_segmentation import ImageSegmentationTask
from mobius.tasks._multimodal import MultiModalTask
from mobius.tasks._object_detection import ObjectDetectionTask
from mobius.tasks._owlv2_detection import Owlv2ObjectDetectionTask
from mobius.tasks._phi4mm_multimodal import Phi4MMMultiModalTask
from mobius.tasks._qwen_image_vae import QwenImageVAETask
from mobius.tasks._rwkv6_causal_lm import Rwkv6CausalLMTask
from mobius.tasks._speech2text_seq2seq import Speech2TextSeq2SeqTask
from mobius.tasks._rwkv_causal_lm import RwkvCausalLMTask
from mobius.tasks._seq2seq import Seq2SeqTask
from mobius.tasks._speech_language import SpeechLanguageTask
from mobius.tasks._speech_to_text import SpeechToTextTask
from mobius.tasks._ssm_causal_lm import SSM2CausalLMTask, SSMCausalLMTask
from mobius.tasks._tts import TTSTask
from mobius.tasks._vae import VAETask
from mobius.tasks._video_denoising import VideoDenoisingTask
from mobius.tasks._vision_language import Qwen3VLVisionLanguageTask
from mobius.tasks._vision_language_3model import (
    HybridQwenVLTask,
    MllamaVisionLanguageTask,
    QwenVLTask,
    VisionLanguageTask,
)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, type[ModelTask]] = {
    "adapter": AdapterTask,
    "audio-feature-extraction": AudioFeatureExtractionTask,
    "clap-audio-feature-extraction": ClapAudioFeatureExtractionTask,
    "codec": CodecTask,
    "contrastive": ContrastiveTask,
    "controlnet": ControlNetTask,
    "denoising": DenoisingTask,
    "feature-extraction": FeatureExtractionTask,
    "image-classification": ImageClassificationTask,
    "image-segmentation": ImageSegmentationTask,
    "object-detection": ObjectDetectionTask,
    "owlv2-object-detection": Owlv2ObjectDetectionTask,
    "seq2seq": Seq2SeqTask,
    "text-generation": CausalLMTask,
    "hybrid-text-generation": HybridCausalLMTask,
    "vae": VAETask,
    "qwen-image-vae": QwenImageVAETask,
    "vision-language": VisionLanguageTask,
    "mllama-vision-language": MllamaVisionLanguageTask,
    "qwen-vl": QwenVLTask,
    "hybrid-qwen-vl": HybridQwenVLTask,
    "qwen3-vl-vision-language": Qwen3VLVisionLanguageTask,
    "multimodal": MultiModalTask,
    "phi4mm-multimodal": Phi4MMMultiModalTask,
    "speech-language": SpeechLanguageTask,
    "speech-to-text": SpeechToTextTask,
    "ssm-text-generation": SSMCausalLMTask,
    "ssm2-text-generation": SSM2CausalLMTask,
    "rwkv-text-generation": RwkvCausalLMTask,
    "rwkv6-text-generation": Rwkv6CausalLMTask,
    "speech2text-seq2seq": Speech2TextSeq2SeqTask,
    "tts": TTSTask,
    "video-denoising": VideoDenoisingTask,
}


def get_task(task: str | ModelTask) -> ModelTask:
    """Resolve a task name or instance to a ModelTask.

    Args:
        task: Either a task name string (e.g. ``"text-generation"``) or
            a ``ModelTask`` instance.

    Returns:
        A ``ModelTask`` instance.

    Raises:
        ValueError: If the task name is not registered.
    """
    if isinstance(task, ModelTask):
        return task
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Available tasks: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task]()
