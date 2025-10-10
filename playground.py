import torch
import tiktoken
from inference import generate_and_print_sample, token_ids_to_text
from model import LLMModel, Config


def playground_data(
    data_loader: torch.utils.data.DataLoader, tokenizer: tiktoken.Encoding
):
    token_ids: torch.Tensor = data_loader.dataset[0][0]
    print("Token IDs shape:", token_ids.shape)
    print("Token IDs:", token_ids)
    print("Decoded text:", token_ids_to_text(token_ids.unsqueeze(0), tokenizer))


def playground_inference(
    model: LLMModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    config: Config,
    max_new_tokens: int = 200,
):
    start_context = "Sherlock Holmes looked me in the eye and said"

    generate_and_print_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_context=start_context,
        context_size=config.context_length,
        max_new_tokens=max_new_tokens,
    )
