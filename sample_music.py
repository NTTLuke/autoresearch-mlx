from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from prepare import Tokenizer, MAX_SEQ_LEN
from train import (
    GPT,
    GPTConfig,
    DEPTH,
    ASPECT_RATIO,
    HEAD_DIM,
    WINDOW_PATTERN,
)


def build_inference_config(vocab_size: int) -> GPTConfig:
    """
    Rebuild the exact model config used during training.
    This must match the training architecture so the saved weights load correctly.
    """
    model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=model_dim // HEAD_DIM,
        n_kv_head=model_dim // HEAD_DIM,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


def _set_path_value(model, path: str, value):
    """Set a parameter by dot-separated path, resolving lists vs dicts from the model."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def load_trained_model(weights_path: str | Path):
    """
    Load tokenizer + trained model weights.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find weights file: {weights_path}\n"
            "Run training first and save weights, e.g.\n"
            'model.save_weights("music_model.safetensors")'
        )

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = build_inference_config(vocab_size)
    model = GPT(config)

    raw_weights = mx.load(str(weights_path))
    for path, value in raw_weights.items():
        _set_path_value(model, path, value)
    model.eval()

    return model, tokenizer


def encode_prompt(tokenizer: Tokenizer, prompt: str) -> list[int]:
    """
    Normalize tokenizer output to a flat list of token ids.
    """
    ids = tokenizer.encode(prompt)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def sample_next_token(logits, temperature: float) -> int:
    """
    Sample one token from the final-step logits.
    """
    if temperature <= 0:
        return int(mx.argmax(logits).item())

    next_logits = logits / temperature
    return int(mx.random.categorical(next_logits).item())


def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
) -> str:
    """
    Generate a continuation from a text prompt.
    """
    ids = encode_prompt(tokenizer, prompt)

    for _ in range(max_new_tokens):
        x = mx.array([ids[-MAX_SEQ_LEN:]], dtype=mx.int32)
        logits = model(x)  # expected shape: [1, T, vocab]
        next_token_logits = logits[0, -1]
        next_id = sample_next_token(next_token_logits, temperature)
        ids.append(next_id)

    return tokenizer.decode(ids)


def read_prompt(prompt_arg: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text()

    if prompt_arg:
        return prompt_arg

    return """L:1/8
M:4/4
K:D
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate music text from saved autoresearch weights.")
    parser.add_argument(
        "--weights",
        default="music_model.safetensors",
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt string to continue",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; lower is more conservative",
    )
    args = parser.parse_args()

    prompt = read_prompt(args.prompt, args.prompt_file)
    model, tokenizer = load_trained_model(args.weights)

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("=== PROMPT ===")
    print(prompt)
    print()
    print("=== SAMPLE ===")
    print(output)


if __name__ == "__main__":
    main()