"""
Visualize the diffusion sampling process step by step
Shows each denoising step with masks and intermediate results
Supports both single-shot and continuous block generation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import (
    DiffusionTransformer,
    DiffusionConfig,
    encode_text,
    decode_tokens,
    MaskedDiffusionSchedule,
)


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    config = DiffusionConfig()
    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_initial_context(data_path, context_len):
    """Load the first context_len characters from dataset"""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()[:context_len]
    tokens = encode_text(text)
    return tokens


def generate_with_visualization(
    model, num_blocks=5, temperature=1.0, dataset_tokens=None
):
    """
    Generate samples and visualize each denoising step

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        temperature: Sampling temperature
        dataset_tokens: Dataset tokens for initial context (if context_len > 0)
    """
    device = model.get_device()
    seq_len = model.config.sequence_len
    num_steps = model.config.diffusion_steps
    context_len = model.config.context_len

    # Create mask schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=num_steps,
        mask_token_id=model.config.mask_token_id,
        context_len=context_len,
    )

    print(f"Generating {num_blocks} blocks with {num_steps} denoising steps each...")
    print(f"Context length: {context_len}")
    print()

    # Pre-calculate all frames for all blocks
    all_frames = []
    all_masks = []
    all_block_indices = []
    completed_blocks_text = []  # Cache decoded text for completed blocks

    prev_context = None

    for block_idx in range(num_blocks):
        print(f"Pre-calculating block {block_idx + 1}/{num_blocks}...")

        # Get context tokens for this block
        context_tokens = None
        if context_len > 0:
            if block_idx == 0 and dataset_tokens is not None:
                # First block: use first context_len tokens from dataset
                context_tokens = dataset_tokens[:context_len].unsqueeze(0).to(device)
            elif block_idx > 0 and prev_context is not None:
                # Subsequent blocks: use last context_len tokens from previous block
                context_tokens = prev_context.unsqueeze(0)

        # Start from all mask tokens
        x = torch.full(
            (1, seq_len),
            model.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            x[:, :context_len] = context_tokens

        # Store initial state
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[context_len:] = True  # Everything except context is masked
        all_frames.append(x[0].cpu().clone())
        all_masks.append(mask.cpu())
        all_block_indices.append(block_idx)

        # Denoise step by step
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                # Apply masking for this timestep
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                x_masked = mask_schedule.add_masks(x, t_batch)

                # Predict clean tokens
                logits = model.forward(x_masked, t_batch)

                # Sample from predicted distribution
                probs = F.softmax(logits / temperature, dim=-1)
                x_new = torch.multinomial(
                    probs.view(-1, model.config.vocab_size), num_samples=1
                ).view(1, seq_len)

                # Only update positions that were masked
                mask_positions = x_masked == model.config.mask_token_id
                x = torch.where(mask_positions, x_new, x)

                # Store frame (every frame)
                all_frames.append(x[0].cpu().clone())
                all_masks.append(mask_positions[0].cpu())
                all_block_indices.append(block_idx)

        # Store the last context_len tokens for next iteration
        if context_len > 0:
            prev_context = x[0, -context_len:]

        # Cache the decoded text for this completed block
        chars_per_row = 64
        num_rows = (seq_len + chars_per_row - 1) // chars_per_row
        block_lines = []
        for row_idx in range(num_rows):
            row_start = row_idx * chars_per_row
            row_end = min(row_start + chars_per_row, seq_len)
            row_text = ""
            for idx in range(row_start, row_end):
                char = decode_tokens([x[0, idx].item()])
                if char == "\n":
                    char = "↵"
                row_text += char
            block_lines.append(row_text)
        completed_blocks_text.append(block_lines)

    print("Done! Now showing animation...\n")

    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Calculate number of rows based on sequence length
    chars_per_row = 64
    num_rows = (seq_len + chars_per_row - 1) // chars_per_row

    # Create text object
    text_obj = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        fontweight="normal",
        linespacing=1.2,
    )

    title = fig.suptitle("", fontsize=12)

    def init():
        """Initialize animation"""
        return [text_obj, title]

    def update(frame_idx):
        """Update function for animation"""
        if frame_idx >= len(all_frames):
            return [text_obj, title]

        frame_tokens = all_frames[frame_idx]
        mask = all_masks[frame_idx]
        block_idx = all_block_indices[frame_idx]

        # Calculate which step within the block we're at
        frames_per_block = num_steps + 1
        step_in_block = frame_idx % frames_per_block

        # Build accumulated text from all completed blocks + current block
        # Show only the last 5 blocks (sliding window)
        max_visible_blocks = 6
        all_lines = []

        # Calculate which blocks to show
        start_block = max(0, block_idx - max_visible_blocks + 1)

        # Add completed blocks in the visible window (use cached decoded text)
        for prev_block_idx in range(start_block, block_idx):
            # Use pre-decoded text for this block
            block_text_lines = completed_blocks_text[prev_block_idx]

            # For the first visible block, show all rows
            # For subsequent blocks, skip the first row (context that overlaps with previous block's last row)
            start_row = 0 if prev_block_idx == start_block else 1
            for row_idx in range(start_row, num_rows):
                all_lines.append(block_text_lines[row_idx])

        # Add current block (skip first row if not the first visible block)
        start_row = 0 if block_idx == start_block else 1
        for row_idx in range(start_row, num_rows):
            start_idx = row_idx * chars_per_row
            end_idx = min(start_idx + chars_per_row, seq_len)
            row_text = ""

            for idx in range(start_idx, end_idx):
                if mask[idx]:
                    row_text += "█"
                else:
                    char = decode_tokens([frame_tokens[idx]])
                    if char == "\n":
                        char = "↵"
                    row_text += char

            all_lines.append(row_text)

        text_obj.set_text("\n".join(all_lines))
        text_obj.set_color("black")

        if step_in_block == 0:
            title_text = f"Block {block_idx + 1}/{num_blocks} - Initial (all masked except context)"
        else:
            title_text = (
                f"Block {block_idx + 1}/{num_blocks} - Step {step_in_block}/{num_steps}"
            )

        title.set_text(title_text)

        return [text_obj, title]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(all_frames),
        interval=15,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


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

    # Load dataset tokens for initial context if context_len > 0
    dataset_tokens = None
    if model.config.context_len > 0:
        print("Loading initial context from dataset...")
        dataset_tokens = load_initial_context(
            "data/tiny_shakespeare.txt", model.config.sequence_len
        )
        print(f"Loaded {len(dataset_tokens)} tokens\n")

    # Generate with visualization
    generate_with_visualization(
        model,
        num_blocks=10,
        temperature=1.0,
        dataset_tokens=dataset_tokens,
    )


if __name__ == "__main__":
    main()
