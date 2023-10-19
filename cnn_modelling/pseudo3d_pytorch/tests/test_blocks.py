"""Tests for blocks.py module."""

import torch

from src.blocks import P3DBlockTypeA, P3DBlockTypeB, P3DBlockTypeC


def test_p3d_block_type_a():
    """Test P3DBlockTypeA forward pass."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeA(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)
    assert y.shape == (1, out_channels, 8, 8, 8)


def test_p3d_block_type_b():
    """Test P3DBlockTypeB forward pass."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeB(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)
    assert y.shape == (1, out_channels, 8, 8, 8)


def test_p3d_block_type_c():
    """Test P3DBlockTypeC forward pass."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeC(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)
    assert y.shape == (1, out_channels, 8, 8, 8)


def test_p3d_block_type_a_stride():
    """Test P3DBlockTypeA forward pass with strides."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeA(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=2, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 4, 4, 4)


def test_p3d_block_type_b_stride():
    """Test P3DBlockTypeB forward pass with strides."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeB(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=2, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 4, 4, 4)


def test_p3d_block_type_c_stride():
    """Test P3DBlockTypeC forward pass with strides."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeC(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=2, dilation=1, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 4, 4, 4)


def test_p3d_block_type_a_dilation():
    """Test P3DBlockTypeA forward pass with dilation."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeA(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=2, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 8, 8, 8)


def test_p3d_block_type_b_dilation():
    """Test P3DBlockTypeB forward pass with dilation."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeB(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=2, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 8, 8, 8)


def test_p3d_block_type_c_dilation():
    """Test P3DBlockTypeC forward pass with dilation."""
    in_channels = 10
    out_channels = 20
    block = P3DBlockTypeC(in_channels=in_channels, inside_channels=5, out_channels=out_channels, kernel_size=3,
                          stride=1, dilation=2, bias=False)

    x = torch.rand(1, in_channels, 8, 8, 8)
    y = block(x)

    assert y.shape == (1, out_channels, 8, 8, 8)
