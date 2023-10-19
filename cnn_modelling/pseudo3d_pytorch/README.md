# pseudo3d_pytorch
Pseudo-3D CNN networks in PyTorch.

This repository contains my own (unofficial) implementation of Pseudo-3D convolutional networks from the paper "Learning
Spatio-Temporal Representation with Pseudo-3D Residual Networks" (https://arxiv.org/abs/1711.10305). 

Precise description of the architecture can be found in the paper. Detailed description of the P3D blocks can be
found in the paper as well.

Additionally, I have implemented 3D version of (CBAM - Bottleneck Attention Module) from "Convolutional Block Attention
Module" paper. (https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html).

This implementation extends original CBAM module to 3D. In this case, the spatial attention is computed
along the temporal dimension (dim=2). Then, mean and max features are concatenated in the filter dimension (dim=1),
like in the original CBAM module. It is because MRI scans are 1-channel 3D volumes, and important features can be
extracted from the temporal dimension in this case.

## Usage
Repository contains implementation of P3D blocks (type A, B, and C) and P3D ResNet-like networks. To use them, you can
import them from `p3d_resnet.py` file and use P3DResNet class.

Model size is parametrized - you can choose between 
every ResNet size using num_blocks_per_stage parameter. Default is `(3, 4, 6, 3)`, which corresponds to the ResNet-50
architecture. For ResNet-101, use `(3, 4, 23, 3)`. For ResNet-152, use `(3, 8, 36, 3)`. 

You can also build your own P3D network using P3DBlockA, P3DBlockB, and P3DBlockC classes from `src/blocks.py` file. 
This file also contains implementation of Attention module (`AttentionBlock3D` class).

Blocks and P3D models are implemented using PyTorch's `nn.Module` class, so you can use them as any other PyTorch
module. Parameters of these classes are described in the docstrings.

Repository also contains tests in `tests` directory. You can run them using `pytest` command. 

Repository is compatible with Python 3.10 and PyTorch 2.0.1.
