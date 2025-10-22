"""
Sample from diffusion model using Conway's Game of Life rules for masking
Displays 256 tokens as a 16x16 grid and applies Game of Life dynamics
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from model import DiffusionTransformer, DiffusionConfig


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Create model with same config as training
    config = DiffusionConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        max_timesteps=32,
    )

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def apply_game_of_life_rules(grid):
    """
    Apply Conway's Game of Life rules to a binary grid
    Args:
        grid: Binary tensor of shape (16, 16)
    Returns:
        next_grid: Next state according to Game of Life rules
    """
    # Pad the grid to handle edges
    padded = F.pad(grid.float(), (1, 1, 1, 1), mode='constant', value=0)

    # Count neighbors for each cell
    neighbors = torch.zeros_like(grid, dtype=torch.float32)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            neighbors += padded[1+i:17+i, 1+j:17+j]

    # Apply Game of Life rules:
    # 1. Any live cell with 2-3 neighbors survives
    # 2. Any dead cell with exactly 3 neighbors becomes alive
    # 3. All other cells die or stay dead
    next_grid = torch.zeros_like(grid)
    next_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    next_grid[(grid == 0) & (neighbors == 3)] = 1

    return next_grid


def tokens_to_grid(tokens):
    """Convert 256 tokens to 16x16 binary grid based on character value"""
    # Use odd/even or threshold to create binary grid
    grid = (tokens % 2).reshape(16, 16)
    return grid


def display_grid(grid, tokens, mask=None):
    """Display the 16x16 grid with characters"""
    print("=" * 18)
    for i in range(16):
        row = ""
        for j in range(16):
            idx = i * 16 + j
            # Show █ for masked positions (alive cells)
            if mask is not None and mask[idx]:
                row += "█"
            else:
                char = chr(min(int(tokens[idx]), 127))
                # Replace newline with space
                if char == '\n':
                    char = ' '
                row += char
        print("|" + row + "|")
    print("=" * 18)


def generate_with_game_of_life(
    model, num_iterations=10, seq_len=256, num_steps=32, temperature=1.0
):
    """
    Generate samples using Game of Life rules to determine masking
    Uses matplotlib animation for live visualization

    Args:
        model: The trained diffusion model
        num_iterations: Number of Game of Life iterations to run
        seq_len: Sequence length (must be 256 for 16x16 grid)
        num_steps: Number of diffusion denoising steps
        temperature: Sampling temperature
    """
    assert seq_len == 256, "seq_len must be 256 for 16x16 grid"

    device = model.get_device()

    print(f"Generating with Game of Life dynamics for {num_iterations} iterations\n")

    # Initial generation: create first sample from scratch
    with torch.no_grad():
        tokens = model.sample(
            batch_size=1,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temperature,
            device=device,
        )[0]  # Shape: (256,)

    # Convert to grid
    grid = tokens_to_grid(tokens)

    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Create a SINGLE text object for all 16 rows (centered)
    # This ensures zero gap between lines
    text_obj = ax.text(0.5, 0.5, '', ha='center', va='center',
                     fontsize=14, family='monospace',
                     fontweight='normal', linespacing=1.0)

    title = fig.suptitle('Initial state - Masked cells: 0/256', fontsize=14)

    # State variables for animation
    state = {
        'tokens': tokens,
        'grid': grid,
        'iteration': 0
    }

    def init():
        """Initialize animation"""
        # Show initial state
        mask = (state['grid'] == 1).flatten()

        # Build all rows as a single multi-line string
        lines = []
        for row_idx in range(16):
            row_text = ""
            for col_idx in range(16):
                idx = row_idx * 16 + col_idx
                if mask[idx]:
                    row_text += '█'
                else:
                    char = chr(min(int(state['tokens'][idx]), 127))
                    if char == '\n':
                        char = ' '
                    row_text += char
            lines.append(row_text)

        text_obj.set_text('\n'.join(lines))
        text_obj.set_color('black')

        title.set_text(f'Initial state - Masked cells: {mask.sum().item()}/256')
        return [text_obj, title]

    def update(frame):
        """Update function for animation"""
        if state['iteration'] >= num_iterations:
            return [text_obj, title]

        # Apply Game of Life rules to get next state
        next_grid = apply_game_of_life_rules(state['grid'])

        # Create mask: alive cells will be resampled
        mask = (next_grid == 1).flatten()  # Shape: (256,)

        # Start from current tokens
        x = state['tokens'].clone().unsqueeze(0)  # Shape: (1, 256)

        # Denoise, but only update masked positions (alive cells)
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                logits = model.forward(x, t_batch)

                if t > 0:
                    # Sample from predicted distribution
                    probs = F.softmax(logits / temperature, dim=-1)
                    x_new = torch.multinomial(
                        probs.view(-1, model.config.vocab_size), num_samples=1
                    ).view(1, seq_len)

                    # Only update masked positions (alive cells)
                    x[0, mask] = x_new[0, mask]
                else:
                    # Final step: take argmax
                    x_new = torch.argmax(logits, dim=-1)
                    x[0, mask] = x_new[0, mask]

        # Update state
        state['tokens'] = x[0]
        state['grid'] = next_grid
        state['iteration'] += 1

        # Update display - build all rows as a single multi-line string
        lines = []
        for row_idx in range(16):
            row_text = ""
            for col_idx in range(16):
                idx = row_idx * 16 + col_idx
                if mask[idx]:
                    row_text += '█'
                else:
                    char = chr(min(int(state['tokens'][idx]), 127))
                    if char == '\n':
                        char = ' '
                    row_text += char
            lines.append(row_text)

        text_obj.set_text('\n'.join(lines))
        text_obj.set_color('black')

        title.set_text(f'Iteration {state["iteration"]} - Masked cells: {mask.sum().item()}/256')

        return [text_obj, title]

    # Create animation (faster, with looping)
    anim = FuncAnimation(fig, update, init_func=init, frames=num_iterations,
                        interval=50, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    return state['tokens']


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
    checkpoint_path = "diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("Model loaded!\n")

    # Generate with Game of Life dynamics
    generate_with_game_of_life(
        model,
        num_iterations=200,
        seq_len=256,
        num_steps=32,
        temperature=1.0,
    )


if __name__ == "__main__":
    main()
