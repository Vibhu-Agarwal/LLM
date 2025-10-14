import torch
from model import LLMModel
from utils import get_device
from model_run_utils import calc_loss_batch, ModelRunUtils
from typing import Callable


def get_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer


def train_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: LLMModel,
    optimizer: torch.optim.Optimizer,
    config_dict: dict,
    start_epoch: int = 0,
    num_epochs: int = 1,
    on_eval: Callable[[], None] | None = None,
):
    device = get_device()
    util_handler = ModelRunUtils(
        n_train=len(train_loader),
        num_epochs=num_epochs,
        model=model,
        optimizer=optimizer,
        val_loader=val_loader,
        device=device,
        config_dict=config_dict,
        eval_callback=on_eval,
    )

    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_total_loss = 0.0
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_total_loss += batch_loss

            util_handler.handle_logs_and_evaluations(
                epoch,
                batch_idx,
                batch_loss,
                epoch_total_loss,
            )

        util_handler.save_model(epoch)

    print("Training finished.")
    util_handler.writer.close()
