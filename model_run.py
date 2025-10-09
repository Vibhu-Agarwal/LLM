import torch
from model import LLMModel
from torch.utils.tensorboard import SummaryWriter
from utils import (
    save_checkpoint,
    get_device,
)


def get_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer


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


def train_model(
    train_loader: torch.utils.data.DataLoader,
    model: LLMModel,
    optimizer: torch.optim.Optimizer,
    start_epoch: int = 0,
    num_epochs: int = 1,
    tf_experiment: str = "runs/my_first_llm_exp",
):
    device = get_device()
    writer = SummaryWriter(tf_experiment)

    model.train()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch/Loss", loss.item(), step)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                save_checkpoint(
                    epoch + 1,
                    model,
                    optimizer,
                    avg_loss,
                    filename=f"checkpoint_ep{epoch+1}_step{batch_idx+1}.pth.tar",
                )

        avg_loss = total_loss / len(train_loader)
        epoch_num = epoch + 1
        print(
            f"Epoch [{epoch_num}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}"
        )

        save_checkpoint(
            epoch_num,
            model,
            optimizer,
            avg_loss,
            filename=f"checkpoint_ep{epoch_num}.pth.tar",
        )

        writer.add_scalar("Epoch/Avg_Loss", avg_loss, epoch)

    print("Training finished.")
    writer.close()
