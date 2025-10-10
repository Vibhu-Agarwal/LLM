# LLM From Scratch

This project is a from-scratch implementation of a Large Language Model (LLM) in PyTorch. The primary goal is to build a decoder-only transformer model, train it on a custom dataset, and use it for text generation.

## Features

*   **Data Loading:** Custom data loader for text datasets.
*   **Tokenizer:** Utilizes `tiktoken` for efficient tokenization.
*   **Training:** Implements a standard training loop with validation and checkpointing.
*   **Inference:** Generate new text from a starting context.
*   **Utilities:** Helper functions for saving/loading checkpoints and device selection.

## Installation

It is recommended to use [Poetry](https://python-poetry.org/) for managing dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd LLM
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

## Usage

The primary entry point for training the model is `main.py`.

```bash
poetry run python main.py
```

You can modify `main.py` to experiment with different configurations and data. The `playground.py` file contains functions for simple inference and data inspection, which can be integrated into `main.py` for quick tests.

## File Descriptions

| File                | Description                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `main.py`           | The main entry point for training the model.                                                                     |
| `config.py`         | Contains the configuration class for the model architecture.                                                     |
| `data.py`           | Handles data loading, tokenization, and creating `DataLoader` instances for training and validation.             |
| `model.py`          | Defines the LLM architecture.                                                                                    |
| `model_run.py`      | Contains the main training loop and optimizer setup.                                                             |
| `model_run_utils.py`| Provides utility functions for the training loop, such as loss calculation and TensorBoard logging.              |
| `inference.py`      | Includes functions for generating text with a trained model.                                                     |
| `playground.py`     | A collection of functions for experimenting with the model and data.                                             |
| `utils.py`          | Contains utility functions for saving/loading checkpoints and selecting the correct device for training.         |