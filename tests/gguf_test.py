# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GGUF import support.

Creates synthetic GGUF files using the ``gguf`` package's writer to
test the full pipeline without network downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

try:
    import gguf as gguf_lib

    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

pytestmark = pytest.mark.skipif(not HAS_GGUF, reason="gguf not installed")


def _create_tiny_gguf(
    path,
    arch: str = "llama",
    hidden: int = 64,
    layers: int = 2,
    heads: int = 2,
    vocab: int = 256,
    ffn_size: int | None = None,
) -> str:
    """Create a minimal GGUF file with fp32 weights for testing.

    Returns the string path to the created file.
    """
    if ffn_size is None:
        ffn_size = hidden * 4
    path = str(path)
    writer = gguf_lib.GGUFWriter(path, arch)

    # Metadata
    writer.add_block_count(layers)
    writer.add_embedding_length(hidden)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_context_length(128)
    writer.add_feed_forward_length(ffn_size)

    # Tensors — fp32 random weights
    rng = np.random.default_rng(42)

    writer.add_tensor(
        "token_embd.weight",
        rng.standard_normal((vocab, hidden), dtype=np.float32),
    )
    writer.add_tensor(
        "output_norm.weight",
        rng.standard_normal((hidden,), dtype=np.float32),
    )
    writer.add_tensor(
        "output.weight",
        rng.standard_normal((vocab, hidden), dtype=np.float32),
    )

    for i in range(layers):
        prefix = f"blk.{i}"
        writer.add_tensor(
            f"{prefix}.attn_q.weight",
            rng.standard_normal((hidden, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.attn_k.weight",
            rng.standard_normal((hidden, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.attn_v.weight",
            rng.standard_normal((hidden, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.attn_output.weight",
            rng.standard_normal((hidden, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.ffn_gate.weight",
            rng.standard_normal((ffn_size, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.ffn_up.weight",
            rng.standard_normal((ffn_size, hidden), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.ffn_down.weight",
            rng.standard_normal((hidden, ffn_size), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.attn_norm.weight",
            rng.standard_normal((hidden,), dtype=np.float32),
        )
        writer.add_tensor(
            f"{prefix}.ffn_norm.weight",
            rng.standard_normal((hidden,), dtype=np.float32),
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestGGUFReader:
    """Test GGUFModel metadata and tensor reading."""

    def test_architecture(self, tmp_path):
        """GGUFModel extracts the architecture name."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        assert model.architecture == "llama"

    def test_metadata_values(self, tmp_path):
        """GGUFModel parses metadata integer values."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        meta = model.metadata
        assert meta["llama.embedding_length"] == 64
        assert meta["llama.block_count"] == 2
        assert meta["llama.attention.head_count"] == 2
        assert meta["llama.attention.head_count_kv"] == 2
        assert meta["llama.context_length"] == 128
        assert meta["llama.feed_forward_length"] == 256

    def test_get_metadata(self, tmp_path):
        """get_metadata returns value for existing key, default otherwise."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        assert model.get_metadata("llama.block_count") == 2
        assert model.get_metadata("nonexistent.key", 42) == 42

    def test_tensor_names(self, tmp_path):
        """GGUFModel lists all tensor names."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        names = model.tensor_names
        assert "token_embd.weight" in names
        assert "output.weight" in names
        assert "blk.0.attn_q.weight" in names
        assert "blk.0.ffn_gate.weight" in names
        # 3 global + 9 per layer = 12 tensors for 1 layer
        assert len(names) == 12

    def test_tensor_items_shapes(self, tmp_path):
        """tensor_items yields arrays preserving numpy shapes."""
        from mobius.integrations.gguf._reader import GGUFModel

        hidden, vocab = 64, 256
        path = _create_tiny_gguf(
            tmp_path / "test.gguf",
            hidden=hidden,
            vocab=vocab,
            layers=1,
        )
        model = GGUFModel(path)
        tensors = dict(model.tensor_items())

        # Shapes match what was written via GGUFWriter
        assert tensors["token_embd.weight"].shape == (vocab, hidden)
        assert tensors["output.weight"].shape == (vocab, hidden)
        assert tensors["output_norm.weight"].shape == (hidden,)
        assert tensors["blk.0.attn_q.weight"].shape == (hidden, hidden)
        assert tensors["blk.0.ffn_gate.weight"].shape == (hidden * 4, hidden)
        assert tensors["blk.0.ffn_down.weight"].shape == (hidden, hidden * 4)

    def test_get_tensor(self, tmp_path):
        """get_tensor returns a single dequantized tensor."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        t = model.get_tensor("token_embd.weight")
        assert t.shape == (256, 64)  # (vocab, hidden)
        assert t.dtype == np.float32

    def test_get_tensor_missing_raises(self, tmp_path):
        """get_tensor raises KeyError for unknown tensor name."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        with pytest.raises(KeyError, match="nonexistent"):
            model.get_tensor("nonexistent")

    def test_file_not_found_raises(self, tmp_path):
        """GGUFModel raises FileNotFoundError for missing file."""
        from mobius.integrations.gguf._reader import GGUFModel

        with pytest.raises(FileNotFoundError):
            GGUFModel(tmp_path / "does_not_exist.gguf")

    def test_repr(self, tmp_path):
        """GGUFModel repr includes path, arch, and tensor count."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        r = repr(model)
        assert "llama" in r
        assert "12" in r  # tensor count


class TestGGUFConfigMapping:
    """Test GGUF metadata → ArchitectureConfig mapping."""

    def test_basic_config_fields(self, tmp_path):
        """gguf_to_config maps basic metadata to config fields."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)

        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 2
        assert config.num_key_value_heads == 2

    def test_head_dim_derived(self, tmp_path):
        """head_dim is derived from hidden_size / num_heads."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", hidden=128, heads=4)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.head_dim == 32

    def test_intermediate_size(self, tmp_path):
        """intermediate_size maps from feed_forward_length."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", hidden=64, ffn_size=512)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.intermediate_size == 512

    def test_tie_embeddings_false_when_output_present(self, tmp_path):
        """tie_word_embeddings is False when output.weight tensor exists."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is False

    def test_tie_embeddings_true_when_no_output(self, tmp_path):
        """tie_word_embeddings is True when no output.weight tensor."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        # Create GGUF without output.weight
        path = str(tmp_path / "tied.gguf")
        writer = gguf_lib.GGUFWriter(path, "llama")
        writer.add_block_count(1)
        writer.add_embedding_length(32)
        writer.add_head_count(2)
        writer.add_head_count_kv(2)
        writer.add_context_length(64)
        writer.add_feed_forward_length(128)

        rng = np.random.default_rng(0)
        writer.add_tensor(
            "token_embd.weight",
            rng.standard_normal((64, 32), dtype=np.float32),
        )
        writer.add_tensor(
            "output_norm.weight",
            rng.standard_normal((32,), dtype=np.float32),
        )
        # No output.weight — embeddings are tied
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is True

    def test_model_type_stored(self, tmp_path):
        """Config has _gguf_model_type attribute for registry lookup."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert getattr(config, "_gguf_model_type", None) == "llama"


class TestGGUFArchToModelType:
    """Test GGUF architecture → model_type mapping."""

    def test_known_architectures(self):
        """All expected GGUF architectures are mapped."""
        from mobius.integrations.gguf._config_mapping import (
            GGUF_ARCH_TO_MODEL_TYPE,
        )

        assert GGUF_ARCH_TO_MODEL_TYPE["llama"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["mistral"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["qwen2"] == "qwen2"
        assert GGUF_ARCH_TO_MODEL_TYPE["gemma2"] == "gemma2"
        assert GGUF_ARCH_TO_MODEL_TYPE["phi3"] == "phi3"
        assert GGUF_ARCH_TO_MODEL_TYPE["falcon"] == "falcon"
        assert GGUF_ARCH_TO_MODEL_TYPE["gpt2"] == "gpt2"
        assert GGUF_ARCH_TO_MODEL_TYPE["mamba"] == "mamba"


class TestGGUFTensorMapping:
    """Test GGUF → HF tensor name mapping."""

    def test_llama_global_tensors(self):
        """Global tensors map correctly for llama."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert (
            map_gguf_to_hf_names("token_embd.weight", "llama") == "model.embed_tokens.weight"
        )
        assert map_gguf_to_hf_names("output.weight", "llama") == "lm_head.weight"
        assert map_gguf_to_hf_names("output_norm.weight", "llama") == "model.norm.weight"

    def test_llama_block_tensors(self):
        """Block-indexed tensors map correctly for llama."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert (
            map_gguf_to_hf_names("blk.0.attn_q.weight", "llama")
            == "model.layers.0.self_attn.q_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.5.ffn_gate.weight", "llama")
            == "model.layers.5.mlp.gate_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.31.attn_output.weight", "llama")
            == "model.layers.31.self_attn.o_proj.weight"
        )

    def test_skip_tokenizer_tensors(self):
        """Tokenizer tensors return None (should be skipped)."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert map_gguf_to_hf_names("tokenizer.ggml.tokens", "llama") is None

    def test_skip_rope_freqs(self):
        """Rotary embedding tensors return None."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert map_gguf_to_hf_names("rope_freqs.weight", "llama") is None

    def test_unsupported_arch_raises(self):
        """Unsupported architecture raises ValueError."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        with pytest.raises(ValueError, match="Unsupported GGUF"):
            map_gguf_to_hf_names("token_embd.weight", "unknown_arch")

    def test_build_gguf_to_hf_map(self):
        """build_gguf_to_hf_map batch-maps tensor names."""
        from mobius.integrations.gguf._tensor_mapping import (
            build_gguf_to_hf_map,
        )

        gguf_names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "tokenizer.ggml.tokens",
        ]
        result = build_gguf_to_hf_map(gguf_names, "llama")
        assert "token_embd.weight" in result
        assert "blk.0.attn_q.weight" in result
        assert "tokenizer.ggml.tokens" not in result


class TestGGUFTensorProcessors:
    """Test architecture-specific tensor processors."""

    def test_no_op_for_unknown_model_type(self):
        """process_tensors returns state_dict unchanged for unknown types."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        # Config with no model_type → passthrough
        class FakeConfig:
            model_type = "unknown_model_xyz"

        sd = {"layer.weight": torch.randn(4, 4)}
        result = process_tensors(sd, FakeConfig())
        assert result is sd

    def test_no_op_when_no_model_type(self):
        """process_tensors returns state_dict when config has no model_type."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        sd = {"layer.weight": torch.randn(4, 4)}
        result = process_tensors(sd, object())
        assert result is sd

    def test_gemma_norm_offset(self):
        """Gemma processor adds 1 to norm weights."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        class FakeConfig:
            model_type = "gemma2"

        sd = {
            "model.layers.0.input_layernorm.weight": torch.zeros(8),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8),
        }
        result = process_tensors(sd, FakeConfig())
        # Norm weight should have 1 added
        assert torch.allclose(
            result["model.layers.0.input_layernorm.weight"],
            torch.ones(8),
        )
        # Non-norm weight unchanged
        assert torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            torch.ones(8, 8),
        )


class TestCLIBuildGGUF:
    """Test the build-gguf CLI subcommand."""

    def test_help_text(self, capsys):
        """build-gguf subcommand shows in help."""
        from mobius.__main__ import main

        with pytest.raises(SystemExit):
            main(["build-gguf", "--help"])
        out = capsys.readouterr().out
        assert "GGUF" in out or "gguf" in out

    def test_missing_gguf_path_errors(self):
        """build-gguf requires a gguf_path argument."""
        from mobius.__main__ import main

        with pytest.raises(SystemExit):
            main(["build-gguf"])

    def test_keep_quantized_no_quantized_tensors(self, tmp_path):
        """--keep-quantized on F32 GGUF raises ValueError."""
        from mobius.__main__ import main

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        with pytest.raises((SystemExit, ValueError)):
            main(["build-gguf", path, "--keep-quantized"])
