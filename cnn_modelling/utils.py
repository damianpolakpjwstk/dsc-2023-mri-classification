import torch
from torch import nn
from pathlib import Path

class SaveBestModelLoss:
    """
    Save the best model while training.

    If the current epoch's validation loss is less than the previous least less, then save the model state.
    """
    def __init__(self, save_path: str | Path):
        """
        Initialize the best validation loss to infinity.

        :param save_path: Path to save the best model to.
        """
        self.best_valid_loss = float('inf')
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, current_valid_loss: float, epoch: int, model: nn.Module) -> None:
        """
        Save the best model if the current validation loss is less than the previous least loss.
        :param current_valid_loss: Current validation loss.
        :param epoch: Current epoch.
        :param model: Model to save.
        """
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), str(Path(self.save_path, f'best_model.pth')))
