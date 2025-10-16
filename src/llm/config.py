class Config:
    def __init__(
        self,
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_blocks=12,
        n_heads=12,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_layers = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout

    def dict(self):
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
        }
