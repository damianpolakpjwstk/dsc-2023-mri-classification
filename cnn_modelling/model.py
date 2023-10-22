"""CNN Network for MRI classification."""

import torch
import torchsummary
from torch import nn

from pseudo3d_pytorch.src.blocks import P3DBlockTypeA


class MRINet(nn.Module):
    """
    Network for MRI classification.
    """
    def __init__(self,  dropout_value: float | None = 0.2, base_channels: int = 32) -> None:
        """
        Initialize the network.

        :param dropout_value: dropout value for the last fully connected layer.
        :param base_channels: number of output channels of the 1x1x1 convolution in the multiscale stem block.
        """
        super().__init__()
        self.base_channels = base_channels

        self.stem = nn.Sequential(
            P3DBlockTypeA(1, 24, self.base_channels, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=5, stride=2, padding=0)
        )

        self.block1 = nn.Sequential(
            P3DBlockTypeA(self.base_channels, 16, 64, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0)
        )

        self.block2 = nn.Sequential(
            P3DBlockTypeA(64, 32, 128, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0)
        )

        self.block3 = nn.Sequential(
            P3DBlockTypeA(128, 64, 256, kernel_size=3, stride=2, dilation=1),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),
        )

        self.global_pool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.flatten_dropout = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_value) if dropout_value is not None else nn.Identity(),
        )

        self.output_0 = nn.Linear(256 * 1 * 1 * 1, 1)
        self.output_1 = nn.Linear(256 * 1 * 1 * 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.flatten_dropout(x)
        output_0 = self.output_0(x)
        output_1 = self.output_1(x)
        return output_0, output_1


if __name__ == "__main__":
    model = MRINet().to("cuda")
    print(torchsummary.summary(model, (1, 224, 224, 224)))
