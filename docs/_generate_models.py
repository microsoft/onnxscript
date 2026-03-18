"""Generate model documentation pages from the registry.

Run before building docs:
    python docs/_generate_models.py

This inspects the model registry and class metadata to produce
one Markdown page per model_type, plus an index page.
"""

from __future__ import annotations

import inspect
import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mobius._registry import registry

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Text Generation": "Standard autoregressive language models (CausalLM).",
    "Mixture of Experts": "Models that route tokens to a subset of expert MLPs.",
    "Multimodal": "Models that process images, audio, or other modalities alongside text.",
    "Speech-to-Text": "Encoder-decoder models for speech recognition.",
    "Audio": "Audio encoder models for feature extraction (Wav2Vec2, HuBERT, WavLM).",
    "encoder-only": "Encoder-only models for embeddings and classification (BERT, RoBERTa).",
    "encoder-decoder": "Encoder-decoder sequence-to-sequence models (BART, T5, mBART).",
    "vision": "Vision-only models for image classification and feature extraction.",
    "Diffusion": "Diffusion model components: VAE, UNet, DiT, ControlNet, SD3, Flux denoisers.",
    "autoencoder": "Variational autoencoders for image encoding/decoding.",
    "causal-lm": "Absolute positional embedding language models (GPT-2 style).",
    "encoder": "Encoder-only transformer models (BERT family).",
}

# Display order for categories
_CATEGORY_ORDER = [
    "Text Generation",
    "Mixture of Experts",
    "Multimodal",
    "Speech-to-Text",
    "Audio",
    "encoder-only",
    "encoder",
    "encoder-decoder",
    "causal-lm",
    "vision",
    "Diffusion",
    "autoencoder",
]


def _get_category(cls: type) -> str:
    """Read the category from the model class attribute."""
    return getattr(cls, "category", "Text Generation")


def _get_task(cls: type) -> str:
    """Read the default task from the model class attribute."""
    return getattr(cls, "default_task", "text-generation")


def _source_file(cls: type) -> str:
    """Return the source path relative to the models/ package."""
    try:
        path = inspect.getfile(cls)
    except TypeError:
        return ""
    parts = path.replace("\\", "/").split("/")
    try:
        idx = parts.index("models")
        return "/".join(parts[idx:])
    except ValueError:
        return os.path.basename(path)


def _class_description(cls: type) -> str:
    """Extract the first paragraph of the class docstring."""
    doc = inspect.getdoc(cls)
    if not doc:
        return ""
    lines = []
    for line in doc.split("\n"):
        if not line.strip() and lines:
            break
        lines.append(line.strip())
    return " ".join(lines)


def _generate_model_page(model_type: str, cls: type) -> str:
    """Generate a Markdown page for a single model type."""
    task = _get_task(cls)
    source = _source_file(cls)
    description = _class_description(cls)

    lines = [
        f"# {model_type}",
        "",
        "| | |",
        "|---|---|",
        f"| **Model type** | `{model_type}` |",
        f"| **Class** | `{cls.__name__}` |",
        f"| **Task** | `{task}` |",
        f"| **Source** | `{source}` |",
        "",
    ]

    if description:
        lines += [
            "## Description",
            "",
            description,
            "",
        ]

    lines += [
        "## Usage",
        "",
        "```bash",
        "mobius build --model <MODEL_ID> output_dir/",
        "```",
        "",
        "```python",
        "from mobius import build",
        "",
        'model = build("<MODEL_ID>")',
        "```",
        "",
    ]

    return "\n".join(lines)


def _generate_index(entries: dict[str, type]) -> str:
    """Generate the models/index.md page."""
    groups: dict[str, list[tuple[str, type]]] = {}
    for model_type, cls in sorted(entries.items()):
        category = _get_category(cls)
        groups.setdefault(category, []).append((model_type, cls))

    lines = [
        "# Supported Models",
        "",
        f"mobius supports {len(entries)} registered model types.",
        "",
    ]

    for category in _CATEGORY_ORDER:
        models = groups.pop(category, [])
        if not models:
            continue
        desc = _CATEGORY_DESCRIPTIONS.get(category, "")
        lines += [f"## {category}", ""]
        if desc:
            lines += [desc, ""]
        lines += [
            "| `model_type` | Class | Task |",
            "|---|---|---|",
        ]
        for model_type, cls in models:
            task = _get_task(cls)
            lines.append(f"| {{doc}}`{model_type}` | `{cls.__name__}` | `{task}` |")
        lines.append("")

    # Any remaining categories not in _CATEGORY_ORDER
    for category, models in sorted(groups.items()):
        desc = _CATEGORY_DESCRIPTIONS.get(category, "")
        lines += [f"## {category}", ""]
        if desc:
            lines += [desc, ""]
        lines += [
            "| `model_type` | Class | Task |",
            "|---|---|---|",
        ]
        for model_type, cls in models:
            task = _get_task(cls)
            lines.append(f"| {{doc}}`{model_type}` | `{cls.__name__}` | `{task}` |")
        lines.append("")

    # Hidden toctree
    lines += [
        "```{toctree}",
        ":maxdepth: 1",
        ":hidden:",
        "",
    ]
    for model_type in sorted(entries):
        lines.append(model_type)
    lines += ["```", ""]

    return "\n".join(lines)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # registry._map stores ModelRegistration objects — extract the module class
    entries: dict[str, type] = {}
    for model_type, reg in registry._map.items():
        entries[model_type] = reg.module_class

    for model_type, cls in entries.items():
        page = _generate_model_page(model_type, cls)
        path = os.path.join(MODELS_DIR, f"{model_type}.md")
        with open(path, "w") as f:
            f.write(page)

    index = _generate_index(entries)
    with open(os.path.join(MODELS_DIR, "index.md"), "w") as f:
        f.write(index)

    print(f"Generated {len(entries)} model pages + index in {MODELS_DIR}/")


if __name__ == "__main__":
    main()
