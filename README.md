# vibhu-llm

`vibhu-llm` is a PyTorch-based library providing a from-scratch implementation of a Large Language Model (LLM). The primary goal of this library is to offer a clear and understandable decoder-only transformer model that can be trained on custom datasets and used for text generation.

## Installation

You can install `vibhu-llm` directly from PyPI:

```bash
pip install vibhu-llm
```

## Usage

Here's a quick example of how to use `vibhu-llm` to train a model and generate text:

```python
import torch
from llm.config import Config
from llm.data import get_train_and_val_dataloaders, get_tokenizer, DATA_FETCHERS
from llm.model import LLMModel
from llm.model_run import get_optimizer, train_model
from llm.utils import get_device
from llm.inference import generate_and_print_sample

# 1. Setup device and tokenizer
device = get_device()
print(f"Using device: {device}")
tokenizer = get_tokenizer()

# 2. Configure the model
config = Config(
    context_length=256,
    vocab_size=tokenizer.n_vocab,
    emb_dim=128,
    n_blocks=2,
    n_heads=8,
    dropout=0.1,
)
model = LLMModel(config).to(device)
print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Load data
# You can choose from available data fetchers
train_loader, val_loader = get_train_and_val_dataloaders(
    DATA_FETCHERS["harry_potter"],
    tokenizer,
    batch_size=32,
    max_length=config.context_length,
    stride=128,
    train_ratio=0.9,
)

# 4. Setup optimizer
optimizer = get_optimizer(model, lr=0.001)

# 5. Train the model
train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    num_epochs=10,
    config_dict=config.dict(),
    experiment_name="runs/my_experiment",
    eval_interval=100,
    save_interval=100,
)

# 6. Generate text
generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context="The magic of this world",
    context_size=config.context_length,
    max_new_tokens=50,
)
```

## Configuration

The `llm.config.Config` class allows you to customize the model architecture. Here are the available parameters:

| Parameter      | Type  | Description                               | Default |
| -------------- | ----- | ----------------------------------------- | ------- |
| `vocab_size`   | `int` | Vocabulary size                           | 50257   |
| `context_length`| `int` | Context length for the model              | 1024    |
| `emb_dim`      | `int` | Embedding dimension                       | 768     |
| `n_blocks`     | `int` | Number of transformer blocks              | 12      |
| `n_heads`      | `int` | Number of attention heads                 | 12      |
| `dropout`      | `float`| Dropout rate                             | 0.1     |


## Contributing

If you're interested in improving the library, here's how you can get started:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Vibhu-Agarwal/LLM
    cd LLM
    ```

2.  **Install dependencies:**
    It is recommended to use [Poetry](https://python-poetry.org/) for managing dependencies.
    ```bash
    poetry install
    ```

3.  **Project Structure:**

| File                | Description                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `src/llm/config.py`         | Contains the configuration class for the model architecture.                                                     |
| `src/llm/data.py`           | Handles data loading, tokenization, and creating `DataLoader` instances for training and validation.             |
| `src/llm/model.py`          | Defines the LLM architecture.                                                                                    |
| `src/llm/model_run.py`      | Contains the main training loop and optimizer setup.                                                             |
| `src/llm/model_run_utils.py`| Provides utility functions for the training loop, such as loss calculation and TensorBoard logging.              |
| `src/llm/inference.py`      | Includes functions for generating text with a trained model.                                                     |
| `src/llm/utils.py`          | Contains utility functions for saving/loading checkpoints and selecting the correct device for training.         |
| `src/main.py`           | An example of how to use the library. This file is not part of the packaged library. |

