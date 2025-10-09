import torch
import torch.nn as nn
from config import Config


class LLMModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

    def forward(self, in_idx):
        return in_idx
