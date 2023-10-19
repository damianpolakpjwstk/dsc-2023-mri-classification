"""Simple 3D CNN model for MRI classification."""
import torch
from torch import nn


class Simple3DModel(nn.Module):
    """
    Simple 3D CNN model for MRI classification.
    """
    def __init__(self, dropout_value: float = 0.2) -> None:
        """
        Initialize the network.
        :param dropout_value: dropout value for the last fully connected layer.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_age_regression = nn.Linear(32 * 1 * 1 * 1, 1)
        self.fc_sex_classification = nn.Linear(32 * 1 * 1 * 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        age = self.fc_age_regression(x)
        sex = self.fc_sex_classification(x)
        return age, sex
