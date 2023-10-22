import torch
from torch import nn
from pathlib import Path


class SaveBestModelMetrics:
    """
    Save the best model while training.

    If the current epoch's validation metrics are less than the previous metrics, then save the model state.
    """
    def __init__(self, save_path: str | Path):
        """
        Initialize the best validation loss to infinity.

        :param save_path: Path to save the best model to.
        """
        self.best_valid_metric = float('inf') * -1
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, current_valid_metric: float, epoch: int, model: nn.Module) -> None:
        """
        Save the best model if the current validation metric is higher than the previous metric.
        :param current_valid_metric: Current validation metric.
        :param epoch: Current epoch.
        :param model: Model to save.
        """
        if current_valid_metric > self.best_valid_metric:
            self.best_valid_metric = current_valid_metric
            print(f"\nBest validation metric: {self.best_valid_metric}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), str(Path(self.save_path, f'best_model.pth')))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)) + 1e-6
        return loss
