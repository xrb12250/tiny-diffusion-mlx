"""Sampling script for the MLX diffusion transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_model import (
    DiffusionConfig,
    DiffusionTransformer,
    MaskedDiffusionSchedule,
    decode_tokens,
    encode_text,
)
from mlx_training import get_random_context


def load_dataset_tokens(path: Path) -> mx.array | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return encode_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from MLX diffusion model")
    parser.add_argument("--weights", type=Path, default=Path("weights/mlx_diffusion_model.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/tiny_goethe.txt"))
    parser.add_argument("--num-blocks", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    config = DiffusionConfig()
    print(f"Using MLX device: {mx.default_device()}\n")

    model = DiffusionTransformer(config)
    model.init_weights()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    model.load_weights(str(args.weights))
    print(f"Loaded weights from {args.weights}\n")

    dataset_tokens = load_dataset_tokens(args.data)
    if config.context_len > 0 and dataset_tokens is None:
        raise FileNotFoundError(
            "Context length > 0 requires dataset file for context sampling"
        )

    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps,
        mask_token_id=config.mask_token_id,
        context_len=config.context_len,
    )

    if config.context_len > 0 and dataset_tokens is not None:
        print(
            f"Generating continuous blocks (context_len={config.context_len}) with"
            f" temperature {args.temperature}\n"
        )
        prev_context = None
        output_fragments = []
        for block_idx in range(args.num_blocks):
            if block_idx == 0:
                context = get_random_context(dataset_tokens, config.context_len, batch_size=1)
            else:
                context = prev_context
            sample = model.sample(
                batch_size=1,
                seq_len=config.sequence_len,
                mask_schedule=mask_schedule,
                num_steps=None,
                temperature=args.temperature,
                context_tokens=context,
            )
            block_tokens = sample[0]
            prev_context = block_tokens[-config.context_len :].reshape(1, -1)
            text = decode_tokens(block_tokens)
            if block_idx == 0:
                output_fragments.append(text)
            else:
                output_fragments.append(text[config.context_len :])
        combined = "".join(output_fragments)
        print(combined)
    else:
        print(
            f"Generating independent samples (temperature={args.temperature})"
            f" using {config.diffusion_steps} denoising steps\n"
        )
        for i in range(args.num_blocks):
            sample = model.sample(
                batch_size=1,
                seq_len=config.sequence_len,
                mask_schedule=mask_schedule,
                num_steps=None,
                temperature=args.temperature,
            )
            text = decode_tokens(sample[0])
            print(f"--- Sample {i + 1} ---")
            print(text)
            print()


if __name__ == "__main__":
    main()
