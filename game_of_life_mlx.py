"""Animate diffusion sampling using Conway's Game of Life masking driven by MLX."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mlx.core as mx
import numpy as np

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlx_model import DiffusionConfig, DiffusionTransformer, decode_tokens, encode_text


def load_model(weights_path: Path) -> DiffusionTransformer:
    config = DiffusionConfig()
    model = DiffusionTransformer(config)
    model.init_weights()
    model.load_weights(str(weights_path))
    return model


def apply_game_of_life_rules(grid: mx.array) -> mx.array:
    shifts = [-1, 0, 1]
    neighbor_sum = mx.zeros_like(grid)
    for dx in shifts:
        for dy in shifts:
            if dx == 0 and dy == 0:
                continue
            neighbor_sum += mx.roll(mx.roll(grid, dx, axis=0), dy, axis=1)
    survive = (grid == 1) & ((neighbor_sum == 2) | (neighbor_sum == 3))
    born = (grid == 0) & (neighbor_sum == 3)
    return mx.where(survive | born, mx.ones_like(grid), mx.zeros_like(grid))


def load_initial_tokens(path: Path, num_chars: int = 1024) -> np.ndarray:
    text = path.read_text(encoding="utf-8")[:num_chars]
    if len(text) < num_chars:
        text = text + " " * (num_chars - len(text))
    tokens = encode_text(text)
    mx.eval(tokens)
    return np.array(tokens.tolist(), dtype=np.int32)


def run_iteration(
    model: DiffusionTransformer,
    tokens_np: np.ndarray,
    mask_np: np.ndarray,
    temperature: float,
) -> np.ndarray:
    config = model.config
    seq_len = config.sequence_len
    num_steps = config.diffusion_steps
    updated = tokens_np.copy()
    if config.context_len > 0:
        mask_np = mask_np.copy()
        mask_np[: config.context_len] = False
    mask_token = mx.array(config.mask_token_id, dtype=mx.int32)
    for chunk_idx in range(len(tokens_np) // seq_len):
        start = chunk_idx * seq_len
        end = start + seq_len
        chunk_mask = mask_np[start:end]
        if not np.any(chunk_mask):
            continue
        x = mx.array(updated[start:end], dtype=mx.int32)[None, :]
        mx.eval(x)
        bool_mask = mx.array(chunk_mask, dtype=mx.bool_)[None, :]
        for t in range(num_steps - 1, -1, -1):
            t_batch = mx.full((1,), t, dtype=mx.int32)
            x_masked = mx.where(bool_mask, mask_token, x)
            logits = model(x_masked, t_batch) / temperature
            sampled = mx.random.categorical(logits, axis=-1).astype(mx.int32)
            x = mx.where(bool_mask, sampled, x)
        mx.eval(x)
        updated[start:end] = np.array(x.tolist()[0], dtype=np.int32)
    return updated


def build_animation(
    frames: list[np.ndarray], masks: list[np.ndarray], interval_ms: int = 50
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text_obj = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        linespacing=1.0,
    )
    title = fig.suptitle("", fontsize=14)

    def render(frame_tokens: np.ndarray, mask: np.ndarray, iteration: int) -> None:
        rows = []
        for row_idx in range(32):
            start = row_idx * 32
            end = start + 32
            row_mask = mask[start:end]
            if np.any(row_mask):
                chars = list(" " * 32)
                for col, masked in enumerate(row_mask):
                    if masked:
                        chars[col] = "■"
                    else:
                        token = int(frame_tokens[start + col])
                        char = chr(token)
                        chars[col] = " " if char == "\n" else char
                rows.append("".join(chars))
            else:
                token_row = mx.array(frame_tokens[start:end], dtype=mx.int32)
                text_row = decode_tokens(token_row)
                rows.append(text_row.replace("\n", " "))
        text_obj.set_text("\n".join(rows))
        title.set_text(f"Iteration {iteration} - Masked cells: {int(mask.sum())}/1024")

    def init():
        render(frames[0], masks[0], 0)
        return [text_obj, title]

    def update(idx: int):
        render(frames[idx], masks[idx], idx)
        return [text_obj, title]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Game of Life sampler")
    parser.add_argument("--weights", type=Path, default=Path("weights/mlx_diffusion_model.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/tiny_shakespeare.txt"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=50, help="Animation frame interval in ms")
    args = parser.parse_args()

    print(f"Using MLX device: {mx.default_device()}")
    print(f"Loading weights from {args.weights}...")
    model = load_model(args.weights)
    print("Weights loaded.\n")

    print(f"Loading seed text from {args.data}...")
    tokens_np = load_initial_tokens(args.data, num_chars=1024)
    print("Seed ready.\n")

    random_offset = mx.random.randint(0, 256, shape=(1024,), dtype=mx.int32)
    mx.eval(random_offset)
    initial_grid = ((mx.array(tokens_np, dtype=mx.int32) + random_offset) % 2).reshape(32, 32)
    mx.eval(initial_grid)

    frames = [tokens_np.copy()]
    masks = [np.array((initial_grid == 1).flatten().tolist(), dtype=bool)]
    grid = initial_grid
    tokens_current = tokens_np

    for iteration in range(1, args.iterations + 1):
        print(f"Iteration {iteration}/{args.iterations}")
        grid = apply_game_of_life_rules(grid)
        mx.eval(grid)
        mask = np.array(grid.flatten().tolist(), dtype=bool)
        tokens_current = run_iteration(model, tokens_current, mask, args.temperature)
        frames.append(tokens_current.copy())
        masks.append(mask.copy())

    print("Game of Life sampling complete. Launching animation...")
    anim = build_animation(frames, masks, interval_ms=args.interval)
    return anim


if __name__ == "__main__":
    main()
