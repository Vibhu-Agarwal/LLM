import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable

DataFetcher = Callable[[], str]


def _get_sherlock_holmes_text(filename: str, start_marker: str, end_marker: str):
    with open(f"data/sherlock_holmes/{filename}", "r") as file:
        txt = file.read()
    txt = txt.split(start_marker)[1]
    txt = txt.split(end_marker)[0]
    return txt


def _get_adventures_of_sherlock_holmes_text():
    return _get_sherlock_holmes_text(
        "adventures_of_sherlock_holmes.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***",
    )


def _get_valley_of_fear_sherlock_holmes_text():
    return _get_sherlock_holmes_text(
        "valley_of_fear.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE VALLEY OF FEAR ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK THE VALLEY OF FEAR ***",
    )


def _get_sign_of_the_four():
    return _get_sherlock_holmes_text(
        "sign_of_the_four.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE SIGN OF THE FOUR ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK THE SIGN OF THE FOUR ***",
    )


def _get_study_in_scarlet():
    return _get_sherlock_holmes_text(
        "study_in_scarlet.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK A STUDY IN SCARLET ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK A STUDY IN SCARLET ***",
    )


def _get_return_of_sherlock_holmes():
    return _get_sherlock_holmes_text(
        "return_of_sherlock_holmes.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE RETURN OF SHERLOCK HOLMES ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK THE RETURN OF SHERLOCK HOLMES ***",
    )


def _get_hound_of_the_baskervilles():
    return _get_sherlock_holmes_text(
        "hound_of_baskervilles.txt",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE HOUND OF THE BASKERVILLES ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK THE HOUND OF THE BASKERVILLES ***",
    )


def _get_llm_data_fetcher():
    t1 = _get_adventures_of_sherlock_holmes_text()
    t2 = _get_valley_of_fear_sherlock_holmes_text()
    t3 = _get_sign_of_the_four()
    t4 = _get_study_in_scarlet()
    t5 = _get_return_of_sherlock_holmes()
    t6 = _get_hound_of_the_baskervilles()
    return f"{t1}\n\n{t2}\n\n{t3}\n\n{t4}\n\n{t5}\n\n{t6}"


DATA_FETCHERS: dict[str, DataFetcher] = {
    "adventures_of_sherlock_holmes": _get_adventures_of_sherlock_holmes_text,
    "llm_data": _get_llm_data_fetcher,
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
