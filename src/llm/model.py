import torch
import torch.nn as nn
from .config import Config


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert (
            cfg.emb_dim % cfg.n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.cfg = cfg
        self.ln_for_qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        qkv_merged = self.ln_for_qkv(x)  # B x T x 3E
        q, k, v = qkv_merged.chunk(3, dim=-1)  # B x T x E each
        nh, hs = self.cfg.n_heads, self.cfg.emb_dim // self.cfg.n_heads
        q = q.view(*q.shape[:-1], nh, hs).transpose(1, 2)  # B x nh x T x hs
        k = k.view(*k.shape[:-1], nh, hs).transpose(1, 2)  # B x nh x T x hs
        v = v.view(*v.shape[:-1], nh, hs).transpose(1, 2)  # B x nh x T x hs

        pre_att = q @ k.transpose(-2, -1)  # B x nh x T x T
        attn_scores = pre_att / (hs**0.5)  # B x nh x T x T

        T = attn_scores.size(-1)
        mask = torch.tril(torch.ones((T, T), device=x.device)).view(1, 1, T, T)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.nn.functional.softmax(
            attn_scores, dim=-1
        )  # B x nh x T x T

        x = attn_weights @ v  # B x nh x T x hs
        out = x.transpose(1, 2).contiguous().view(*identity.shape)  # B x T x E
        return out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.emb_dim)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.emb_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        identity = x
        x = self.ln1(x)  # B x T x E
        atten_out = self.attn(x)  # B x T x E
        x = self.ln2(identity + atten_out)  # B x T x E
        x = self.mlp(x)  # B x T x E
        return x + atten_out  # B x T x E


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.n_layers)))
        self.ln = nn.LayerNorm(cfg.emb_dim)

    def forward(self, x):
        x = self.blocks(x)  # B x T x E
        out = self.ln(x)  # B x T x E
        return out


class TokenPostionEmbedding(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)

    def forward(self, in_idx):
        _, seq_length = in_idx.size()
        positions = torch.arange(seq_length, device=in_idx.device).unsqueeze(0)
        token_embeddings = self.token_emb(in_idx)  # B x T x E
        position_embeddings = self.pos_emb(positions)  # 1 x T x E
        return token_embeddings + position_embeddings


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.emb = TokenPostionEmbedding(cfg)
        self.transformer = Transformer(cfg)
        self.ln_f = nn.Linear(cfg.emb_dim, cfg.vocab_size)

    def forward(self, in_idx):
        emb = self.emb(in_idx)  # B x T x E
        x = self.transformer(emb)  # B x T x E
        out = self.ln_f(x)  # B x T x V
        return out


class LLMModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gpt = GPT(cfg)

    def forward(self, in_idx):
        out = self.gpt(in_idx)  # B x T x V
        return out
