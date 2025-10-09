from config import Config
from data import create_dataloader, get_tokenizer
from model import LLMModel
from model_run import get_optimizer, train_model
from utils import load_checkpoint, ensure_checkpoints_dir_exists, get_device

device = get_device()
print(f"Using device: {device}")

tokenizer = get_tokenizer()
config = Config(context_length=256, vocab_size=tokenizer.n_vocab)
model = LLMModel(config)
txt = open("data/adventures_of_sherlock_holmes.txt", "r").read()


train_loader = create_dataloader(
    txt=txt,
    tokenizer=tokenizer,
    batch_size=32,
    max_length=config.context_length,
    stride=128,
    shuffle=True,
    drop_last=True,
)
optimizer = get_optimizer(model)


if __name__ == "__main__":
    ensure_checkpoints_dir_exists()
    checkpoint_file = "saved_checkpoints/filename.pth.tar"

    start_epoch = 0
    if checkpoint_file:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer, device)

    train_model(
        train_loader,
        model,
        optimizer,
        start_epoch=start_epoch,
        num_epochs=1,
        tf_experiment="runs/single_layer_rnn_greedy_search",
    )
