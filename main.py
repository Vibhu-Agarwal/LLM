from config import Config
from data import get_train_and_val_dataloaders, get_tokenizer
from model import LLMModel
from model_run import get_optimizer, train_model
from utils import load_checkpoint, ensure_checkpoints_dir_exists, get_device
from playground import playground_inference

device = get_device()
print(f"Using device: {device}")

tokenizer = get_tokenizer()
config = Config(
    context_length=256,
    vocab_size=tokenizer.n_vocab,
    emb_dim=128,
    n_blocks=2,
    n_heads=32,
    ffn_dim=512,
)
model = LLMModel(config)
txt = open("data/adventures_of_sherlock_holmes.txt", "r").read()


train_loader, val_loader = get_train_and_val_dataloaders(
    txt,
    tokenizer,
    batch_size=32,
    max_length=config.context_length,
    stride=128,
    train_ratio=0.9,
)
optimizer = get_optimizer(model)


if __name__ == "__main__":
    ensure_checkpoints_dir_exists()
    checkpoint_file = "saved_checkpoints/filename.pth.tar"

    start_epoch = 0
    if checkpoint_file:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer, device)
        if start_epoch > 0:
            playground_inference(model, tokenizer, device, config)

    train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        start_epoch=start_epoch,
        num_epochs=1,
        on_eval=lambda: playground_inference(model, tokenizer, device, config),
    )
