import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


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
    txt,
    tokenizer: tiktoken.Encoding,
    batch_size=4,
    max_length=256,
    stride=128,
    train_ratio=0.9,
) -> tuple[DataLoader, DataLoader]:
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
