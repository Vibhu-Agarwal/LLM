import torch
import tiktoken
from model import LLMModel


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(
    model: LLMModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
    context_size: int,
    max_new_tokens: int,
):
    print(
        f"Generating text with context: '{start_context}' [max_new_tokens={max_new_tokens} context_size={context_size}]"
    )
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.encode("unicode-escape").decode())
        print("-" * 20)
        print(decoded_text)


def generate_text_simple(
    model: LLMModel, idx: torch.Tensor, max_new_tokens: int, context_size: int
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    print("Final token IDs shape:", idx.shape)
    return idx
