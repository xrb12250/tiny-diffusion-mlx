"""
Sample/inference script for the trained diffusion model
"""

import torch
from model import (
    DiffusionTransformer,
    DiffusionConfig,
    decode_tokens,
    encode_text,
    MaskedDiffusionSchedule,
)


def load_dataset_text(data_path="data/tiny_shakespeare.txt"):
    """Load dataset text for random context sampling"""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return encode_text(text)


def get_random_context(dataset_tokens, context_len, batch_size=1):
    """Get random context tokens from dataset"""
    max_start = len(dataset_tokens) - context_len
    start_indices = torch.randint(0, max_start, (batch_size,))
    context_tokens = torch.stack(
        [dataset_tokens[start : start + context_len] for start in start_indices]
    )
    return context_tokens


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Create model with same config as training
    config = DiffusionConfig()

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def generate_samples(model, num_samples=5, temperature=1.0, dataset_tokens=None):
    """Generate text samples from the model"""
    device = model.get_device()

    # Create mask schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=model.config.diffusion_steps,
        mask_token_id=model.config.mask_token_id,
        context_len=model.config.context_len,
    )

    print(f"Generating {num_samples} samples with {model.config.diffusion_steps} denoising steps...\n")

    for i in range(num_samples):
        with torch.no_grad():
            # Get random context if context_len > 0
            context_tokens = None
            if model.config.context_len > 0 and dataset_tokens is not None:
                context_tokens = get_random_context(
                    dataset_tokens, model.config.context_len, batch_size=1
                )

            # Generate a sample
            tokens = model.sample(
                batch_size=1,
                seq_len=model.config.sequence_len,
                mask_schedule=mask_schedule,
                num_steps=None,
                temperature=temperature,
                device=device,
                context_tokens=context_tokens,
            )

            # Decode tokens to text
            text = decode_tokens(tokens[0])

            print(f"--- Sample {i + 1} ---")
            print(text)
            print()


def generate_continuous_blocks(
    model, num_blocks=30, temperature=1.0, dataset_tokens=None
):
    """
    Generate multiple blocks sequentially, where each block is conditioned on the
    last context_len characters of the previous block.

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        temperature: Sampling temperature
    """
    device = model.get_device()
    context_len = model.config.context_len

    # Create mask schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=model.config.diffusion_steps,
        mask_token_id=model.config.mask_token_id,
        context_len=model.config.context_len,
    )

    print(
        f"Generating {num_blocks} continuous blocks with {model.config.diffusion_steps} denoising steps..."
    )
    print(
        f"Each block conditions on the last {context_len} characters of the previous block\n"
    )

    all_text = ""

    for block_idx in range(num_blocks):
        with torch.no_grad():
            # Get context tokens for this block
            context_tokens = None
            if block_idx == 0 and dataset_tokens is not None:
                # First block: use random context from dataset
                context_tokens = get_random_context(dataset_tokens, context_len, batch_size=1)
            elif block_idx > 0:
                # Subsequent blocks: use last context_len tokens from previous block
                context_tokens = prev_context.unsqueeze(0)

            # Generate block
            tokens = model.sample(
                batch_size=1,
                seq_len=model.config.sequence_len,
                mask_schedule=mask_schedule,
                num_steps=None,
                temperature=temperature,
                device=device,
                context_tokens=context_tokens,
            )

            # Store the last context_len tokens for next iteration
            prev_context = tokens[0, -context_len:]

            # Decode tokens to text
            text = decode_tokens(tokens[0])

            # Print in real-time
            if block_idx == 0:
                print(text, end="", flush=True)
                all_text += text
            else:
                # Don't repeat the context that was already printed
                new_text = text[context_len:]
                print(new_text, end="", flush=True)
                all_text += new_text

    print("\n")
    return all_text


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Load model
    checkpoint_path = "weights/diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("Model loaded!\n")

    # Load dataset for random context sampling
    dataset_tokens = None
    if model.config.context_len > 0:
        print("Loading dataset for context sampling...")
        dataset_tokens = load_dataset_text("data/tiny_shakespeare.txt")
        print(f"Loaded {len(dataset_tokens)} tokens from dataset\n")

    # Choose generation mode based on context_len
    if model.config.context_len > 0:
        print(f"Using continuous block generation (context_len={model.config.context_len})\n")
        generate_continuous_blocks(
            model,
            num_blocks=30,
            temperature=1.0,
            dataset_tokens=dataset_tokens,
        )
    else:
        print("Using independent sample generation (no context)\n")
        generate_samples(
            model,
            num_samples=5,
            temperature=1.0,
            dataset_tokens=dataset_tokens,
        )


if __name__ == "__main__":
    main()
