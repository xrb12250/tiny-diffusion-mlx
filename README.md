# Tiny Diffusion — MLX Port

This repository contains an Apple MLX implementation of the Tiny Diffusion character-level transformer. It keeps the same diffusion-based denoising objective as the original PyTorch project while taking advantage of MLX kernels for fast training and sampling on Apple Silicon devices.

The original PyTorch version of the project now lives in [`legacy/`](legacy/) to keep the focus of the root package on the MLX port.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- macOS with Apple Silicon and the [MLX](https://github.com/ml-explore/mlx) runtime (tested on MLX ≥ 0.14)

## Installation

```bash
# Clone and enter the repository
git clone <repository-url>
cd tiny-diffusion-mlx

# Install dependencies
uv sync
```

> **Tip:** `uv` automatically creates and manages the virtual environment. Use `uv run …` in the commands below to execute scripts within that environment.

## Training with MLX

`mlx_training.py` mirrors the training loop of the original project while running entirely on MLX arrays.

```bash
# Train from scratch on Tiny Shakespeare (data/tiny_shakespeare.txt)
uv run mlx_training.py --max-iters 20000 --weights weights/mlx_diffusion_model.npz
```

Key arguments:

- `--data`: path to the training corpus (defaults to `data/tiny_shakespeare.txt`).
- `--batch-size`: number of sequences per step (default `64`).
- `--max-iters`: number of optimization steps.
- `--eval-interval`: interval (in steps) for printing diffusion samples.
- `--weights`: output path for the `.npz` checkpoint (also used when resuming).
- `--resume`: resume from a previous run (expects the weights file and optional optimizer state).

## Sampling

`mlx_sample.py` loads a saved MLX checkpoint and generates character blocks using the MLX sampler.

```bash
# Generate 30 connected blocks with overlap for continuous text
uv run mlx_sample.py --weights weights/mlx_diffusion_model.npz --num-blocks 30 --temperature 1.1
```

Set `--data` to a reference corpus if you want the sampler to draw contextual seeds. If `context_len` is zero in the model configuration the script falls back to unconditional sampling.

## Visualising the Diffusion Process

`game_of_life_mlx.py` animates sampling where the mask pattern is driven by Conway's Game of Life.

```bash
uv run game_of_life_mlx.py --weights weights/mlx_diffusion_model.npz --iterations 200 --temperature 0.9
```

This script requires `matplotlib` and will open an interactive window showing how masked cells evolve while the model denoises the text.

## Project Structure

```
tiny-diffusion-mlx/
├── data/                     # Tiny Shakespeare and auxiliary corpora
├── mlx_model.py              # Diffusion transformer implemented with MLX
├── mlx_training.py           # MLX training loop and utilities
├── mlx_sample.py             # Sampling CLI for MLX checkpoints
├── game_of_life_mlx.py       # Game-of-Life-driven diffusion visualiser
├── weights/                  # Default output location for MLX checkpoints
└── legacy/                   # Original PyTorch implementation (kept for reference)
```

## Legacy Code

The previous PyTorch codebase—including its training, sampling and animation scripts, plus the original `.pt` weights—has been moved into [`legacy/`](legacy/). None of that code is imported by the MLX pipeline, so feel free to remove it entirely if you no longer need the historical implementation.

