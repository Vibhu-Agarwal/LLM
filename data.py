import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable

DataFetcher = Callable[[], str]


def _get_adventures_of_sherlock_holmes_text():
    with open("data/adventures_of_sherlock_holmes.txt", "r") as file:
        txt = file.read()
    txt = txt.split(
        "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***\n\n\n\n\n"
    )[1]
    txt = txt.split(
        "\n\n\n\n\n\n*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
    )[0]
    return txt


DATA_FETCHERS: dict[str, DataFetcher] = {
    "adventures_of_sherlock_holmes": _get_adventures_of_sherlock_holmes_text
}


class CustomDataset(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def get_train_and_val_dataloaders(
    data_fetcher: DataFetcher,
    tokenizer: tiktoken.Encoding,
    batch_size=4,
    max_length=256,
    stride=128,
    train_ratio=0.9,
) -> tuple[DataLoader, DataLoader]:
    txt = data_fetcher()
    split_idx = int(len(txt) * train_ratio)

    train_dataset = CustomDataset(txt[:split_idx], tokenizer, max_length, stride)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = CustomDataset(txt[split_idx:], tokenizer, max_length, stride)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader
