from llm.config import Config
from llm.data import get_train_and_val_dataloaders, get_tokenizer, DATA_FETCHERS
from llm.model import LLMModel
from llm.model_run import get_optimizer, train_model
from llm.utils import load_checkpoint, ensure_checkpoints_dir_exists, get_device
from llm.playground import playground_inference

device = get_device()
print(f"Using device: {device}")

tokenizer = get_tokenizer()
config = Config(
    context_length=512,
    vocab_size=tokenizer.n_vocab,
    emb_dim=128,
    n_blocks=4,
    n_heads=16,
)
model = LLMModel(config).to(device)


train_loader, val_loader = get_train_and_val_dataloaders(
    DATA_FETCHERS["llm_data"],
    tokenizer,
    batch_size=32,
    max_length=config.context_length,
    stride=128,
    train_ratio=0.9,
)
optimizer = get_optimizer(model, lr=0.001)


if __name__ == "__main__":
    ensure_checkpoints_dir_exists()
    checkpoint_file = "saved_checkpoints/filename.pth.tar"

    start_epoch = 0
    if checkpoint_file:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer, device)

    playground_inference(model, tokenizer, device, config)

    train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        start_epoch=start_epoch,
        num_epochs=10,
        config_dict=config.dict(),
        on_eval=lambda: playground_inference(model, tokenizer, device, config),
    )
