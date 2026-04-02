# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ResNet model graph construction."""

from __future__ import annotations

from mobius.models.resnet import (
    ResNetModel,
    _BasicBlock,
    _BottleneckBlock,
    _ConvBnRelu,
    _ResNetEncoder,
    _Shortcut,
)


class TestResNetComponents:
    """Tests for individual ResNet building blocks."""

    def test_conv_bn_relu_has_convolution_and_normalization(self):
        block = _ConvBnRelu(3, 16, kernel_size=3, stride=1, padding=1)
        assert hasattr(block, "convolution")
        assert hasattr(block, "normalization")

    def test_shortcut_has_convolution_and_normalization(self):
        shortcut = _Shortcut(64, 128, stride=2)
        assert hasattr(shortcut, "convolution")
        assert hasattr(shortcut, "normalization")

    def test_bottleneck_block_has_layer_and_shortcut(self):
        block = _BottleneckBlock(64, 256, stride=1)
        assert hasattr(block, "layer")
        assert len(block.layer) == 3
        assert block._use_shortcut

    def test_bottleneck_block_no_shortcut_when_dims_match(self):
        block = _BottleneckBlock(256, 256, stride=1)
        assert not block._use_shortcut

    def test_basic_block_has_two_conv_layers(self):
        block = _BasicBlock(64, 64, stride=1)
        assert hasattr(block, "layer")
        assert len(block.layer) == 2
        assert not block._use_shortcut

    def test_basic_block_with_shortcut(self):
        block = _BasicBlock(64, 128, stride=2)
        assert block._use_shortcut
        assert hasattr(block, "shortcut")


class TestResNetModel:
    """Tests for the full ResNet model."""

    def test_model_has_expected_attributes(self):
        from mobius._configs import ResNetConfig

        config = ResNetConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=1,
            intermediate_size=256,
            hidden_act="relu",
            num_channels=3,
            embedding_size=16,
            hidden_sizes=[32, 64, 128, 256],
            depths=[1, 1, 1, 1],
            layer_type="bottleneck",
        )
        model = ResNetModel(config)
        assert hasattr(model, "embedder")
        assert hasattr(model, "encoder")
        assert model.default_task == "image-classification"
        assert model.category == "vision"

    def test_encoder_creates_correct_number_of_stages(self):
        from mobius._configs import ResNetConfig

        config = ResNetConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=1,
            intermediate_size=256,
            hidden_act="relu",
            hidden_sizes=[32, 64, 128, 256],
            depths=[2, 3, 4, 2],
            layer_type="bottleneck",
            embedding_size=16,
        )
        encoder = _ResNetEncoder(config)
        assert len(encoder.stages) == 4
