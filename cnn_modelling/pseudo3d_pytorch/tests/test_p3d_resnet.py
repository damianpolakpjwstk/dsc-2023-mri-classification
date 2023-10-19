"""Tests for p3d_resnet.py module."""

import torch

from src.p3d_resnet import P3DResnet


def test_p3d_resnet_sanity_check():
    """Test P3DResnet forward pass."""
    in_channels = 3
    num_classes = 10
    model = P3DResnet(input_channels=in_channels, num_classes=num_classes)
    x = torch.rand(1, in_channels, 16, 224, 224)
    y = model(x)
    assert y.shape == (1, num_classes)


def test_p3d_resnet_no_dropout_sanity_check():
    """Test P3DResnet forward pass."""
    in_channels = 3
    num_classes = 10
    model = P3DResnet(input_channels=in_channels, num_classes=num_classes, dropout_value=None)
    x = torch.rand(1, in_channels, 16, 224, 224)
    y = model(x)
    assert y.shape == (1, num_classes)


def test_p3d_resnet_sequential_block_type():
    """Test if when block_type='sequential', all of 3 block types are used."""
    model = P3DResnet(input_channels=3, num_classes=10, block_type='sequential', num_blocks_per_stage=(3, 1, 1, 1))
    assert model.stem_block[0].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage1[0].__class__.__name__ == 'P3DBlockTypeB'
    assert model.stage1[1].__class__.__name__ == 'P3DBlockTypeC'
    assert model.stage1[2].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage2[0].__class__.__name__ == 'P3DBlockTypeB'
    assert model.stage3[0].__class__.__name__ == 'P3DBlockTypeC'
    assert model.stage4[0].__class__.__name__ == 'P3DBlockTypeA'


def test_p3d_resnet_same_block_type():
    """Test if when block_type='A', only P3DBlockTypeA is used."""
    model = P3DResnet(input_channels=3, num_classes=10, block_type='A', num_blocks_per_stage=(3, 1, 1, 1))
    assert model.stem_block[0].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage1[0].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage1[1].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage1[2].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage2[0].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage3[0].__class__.__name__ == 'P3DBlockTypeA'
    assert model.stage4[0].__class__.__name__ == 'P3DBlockTypeA'


def test_p3d_resnet_get_num_channels():
    """Check if get_num_channels() returns correct number of channels."""
    model = P3DResnet(input_channels=3, num_classes=10, block_type='A', num_blocks_per_stage=(3, 1, 1, 1))
    assert model.get_num_channels(1) == (64, 64, 256)
    assert model.get_num_channels(2) == (256, 128, 512)
    assert model.get_num_channels(3) == (512, 256, 1024)
    assert model.get_num_channels(4) == (1024, 512, 2048)
    model.base_channels = 32
    assert model.get_num_channels(1) == (32, 32, 128)
    assert model.get_num_channels(2) == (128, 64, 256)
    assert model.get_num_channels(3) == (256, 128, 512)
    assert model.get_num_channels(4) == (512, 256, 1024)
