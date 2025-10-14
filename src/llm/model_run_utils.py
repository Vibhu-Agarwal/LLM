import torch
from torch.utils.tensorboard import SummaryWriter
from .model import LLMModel
from .utils import save_checkpoint
from typing import Callable


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: LLMModel,
    device: torch.device,
) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


class ModelRunUtils:

    def __init__(
        self,
        n_train: int,
        num_epochs: int,
        model: LLMModel,
        optimizer: torch.optim.Optimizer,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config_dict: dict,
        batch_loss_write_interval: int = 10,
        eval_interval: int = 30,
        tf_experiment: str = "runs/experiment",
        eval_callback: Callable[[], None] | None = None,
    ):
        self.writer = SummaryWriter(tf_experiment)
        self.n_train = n_train
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.device = device
        self.batch_loss_write_interval = batch_loss_write_interval
        self.eval_interval = eval_interval
        self.eval_callback = eval_callback
        self.config_dict = config_dict

    def handle_logs_and_evaluations(
        self,
        epoch: int,
        batch_idx: int,
        batch_loss: float,
        epoch_total_loss: float,
    ):
        step = epoch * self.n_train + batch_idx + 1
        if step == 0:
            return

        epoch_avg_loss = epoch_total_loss / (batch_idx + 1)
        print(
            f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{self.n_train}], Batch Loss: {batch_loss:.4f}, Epoch Avg Loss: {epoch_avg_loss:.4f}"
        )
        if batch_idx % self.batch_loss_write_interval == 0:
            self.writer.add_scalar("Train/BatchLoss", batch_loss, step)

        if step % self.eval_interval == 0:
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input_batch, target_batch in self.val_loader:
                    val_loss += calc_loss_batch(
                        input_batch, target_batch, self.model, self.device
                    ).item()
            val_loss /= len(self.val_loader)
            epoch_avg_loss = epoch_total_loss / (batch_idx + 1)
            self.writer.add_scalar("Validation/Loss", val_loss, step)
            self.writer.add_scalar("Train/EpochAvgLoss", epoch_avg_loss, step)
            print(
                f"GlobalSteps: {step}, Validation Loss: {val_loss:.4f}, Train (Epoch) Avg Loss: {epoch_avg_loss:.4f}"
            )

            self.save_model(epoch, batch_idx)

            if self.eval_callback is not None:
                self.eval_callback()

            self.model.train()

    def save_model(self, epoch: int, batch_idx: int | None = None):
        if batch_idx is not None:
            filename = f"checkpoint_ep{epoch}_step{batch_idx}.pth.tar"
        else:
            filename = f"checkpoint_ep{epoch}.pth.tar"
        save_checkpoint(
            epoch + 1,
            self.model,
            self.optimizer,
            filename=filename,
            config_dict=self.config_dict,
        )
