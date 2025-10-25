"""Training script for the MLX diffusion transformer."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import trange

from mlx_model import (
    DiffusionConfig,
    DiffusionTransformer,
    MaskedDiffusionSchedule,
    decode_tokens,
    encode_text,
)


def load_dataset_tokens(path: Path) -> mx.array:
    text = path.read_text(encoding="utf-8")
    return encode_text(text)


def make_data_iterator(
    tokens: mx.array, batch_size: int, seq_len: int
):
    total = int(tokens.shape[0]) // (batch_size * seq_len)
    total *= batch_size * seq_len
    if total == 0:
        raise ValueError("Dataset too small for the chosen batch size and sequence length")
    trimmed = tokens[:total]
    trimmed = trimmed.reshape(batch_size, -1)
    position = 0
    inner_len = int(trimmed.shape[1])
    while True:
        if position + seq_len > inner_len:
            position = 0
        batch = trimmed[:, position : position + seq_len]
        position += seq_len
        yield batch


def get_random_context(
    dataset_tokens: mx.array, context_len: int, batch_size: int = 1
) -> mx.array:
    max_start = int(dataset_tokens.shape[0]) - context_len
    if max_start <= 0:
        raise ValueError("Dataset shorter than context length")
    starts = mx.random.randint(0, max_start, shape=(batch_size,), dtype=mx.int32)
    mx.eval(starts)
    ctx = [dataset_tokens[int(start) : int(start) + context_len] for start in starts.tolist()]
    return mx.stack(ctx)


def build_loss_fn(mask_schedule: MaskedDiffusionSchedule):
    def loss_fn(model: DiffusionTransformer, x0: mx.array) -> mx.array:
        batch_size = int(x0.shape[0])
        timesteps = mx.random.randint(
            0, mask_schedule.num_timesteps, shape=(batch_size,), dtype=mx.int32
        )
        x_t, mask = mask_schedule.add_masks(x0, timesteps, return_mask=True)
        logits = model(x_t, timesteps)
        losses = nn.losses.cross_entropy(logits, x0, reduction="none")
        mask_f = mask.astype(losses.dtype)
        masked_loss = mx.sum(losses * mask_f)
        denom = mx.maximum(mx.sum(mask_f), mx.array(1.0, dtype=losses.dtype))
        return masked_loss / denom

    return loss_fn


def train(
    model: DiffusionTransformer,
    data_iter,
    mask_schedule: MaskedDiffusionSchedule,
    optimizer: optim.Optimizer,
    num_steps: int,
    sample_interval: int,
    dataset_tokens: mx.array | None,
    start_step: int = 0,
) -> int:
    loss_fn = build_loss_fn(mask_schedule)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    end_step = start_step + num_steps
    progress = trange(start_step, end_step, desc="Training", leave=True)
    last_step = start_step
    for global_step in progress:
        batch = next(data_iter)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_value = float(loss.item())
        progress.set_postfix({"loss": f"{loss_value:.4f}"})
        if (global_step + 1) % sample_interval == 0:
            sample_and_print(model, mask_schedule, dataset_tokens, global_step + 1)
        last_step = global_step + 1
    return last_step


def sample_and_print(
    model: DiffusionTransformer,
    mask_schedule: MaskedDiffusionSchedule,
    dataset_tokens: mx.array | None,
    step: int,
) -> None:
    context = None
    if model.config.context_len > 0 and dataset_tokens is not None:
        context = get_random_context(dataset_tokens, model.config.context_len, batch_size=1)
    sample = model.sample(
        batch_size=1,
        seq_len=model.config.sequence_len,
        mask_schedule=mask_schedule,
        num_steps=None,
        temperature=1.0,
        context_tokens=context,
    )
    text = decode_tokens(sample[0])
    print(f"\n--- Sample at step {step} ---")
    print(text)
    print("--- End sample ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diffusion model with MLX")
    parser.add_argument("--data", type=Path, default=Path("data/tiny_shakespeare.txt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/mlx_diffusion_model.npz"),
        help="Path to save model weights (also used as resume source if --resume is set)",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Path to store optimizer/training state (defaults to weights path with .state.pkl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from saved weights and optimizer state if available",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Override starting global step when resuming without a saved optimizer state",
    )
    args = parser.parse_args()

    config = DiffusionConfig()
    print(f"Sequence len: {config.sequence_len}")
    print(f"Diffusion steps: {config.diffusion_steps}")
    print(f"Context len: {config.context_len}")
    print(f"Using MLX device: {mx.default_device()}\n")

    model = DiffusionTransformer(config)
    model.init_weights()
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps,
        mask_token_id=config.mask_token_id,
        context_len=config.context_len,
    )

    dataset_tokens = load_dataset_tokens(args.data)
    data_iter = make_data_iterator(dataset_tokens, args.batch_size, config.sequence_len)

    weights_path = args.weights
    state_path = args.state_path or weights_path.with_suffix(".state.pkl")

    start_step = 0
    if args.resume:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found for resume: {weights_path}")
        model.load_weights(str(weights_path))
        mx.eval(model.parameters())
        print(f"Loaded weights from {weights_path}")
        if state_path.exists():
            with state_path.open("rb") as f:
                state_payload = pickle.load(f)
            optimizer_state = state_payload.get("optimizer_state")
            if optimizer_state is not None:
                optimizer.state.update(optimizer_state)
            start_step = int(state_payload.get("step", 0))
            print(
                f"Loaded optimizer state from {state_path}; resuming from step {start_step}"
            )
        elif args.resume_step is not None:
            start_step = args.resume_step
            print(
                "Optimizer state file missing; resuming with reinitialized optimizer "
                f"from provided step {start_step}"
            )
        else:
            print(
                "Optimizer state file missing; resuming with reinitialized optimizer "
                "from step 0"
            )

    if config.context_len > 0:
        print("Context conditioning enabled. Random contexts will be drawn during sampling.\n")

    print("Starting training...\n")
    final_step = train(
        model=model,
        data_iter=data_iter,
        mask_schedule=mask_schedule,
        optimizer=optimizer,
        num_steps=args.max_iters,
        sample_interval=args.eval_interval,
        dataset_tokens=dataset_tokens,
        start_step=start_step,
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("wb") as f:
        pickle.dump({"optimizer_state": optimizer.state, "step": final_step}, f)
    print(f"Model parameters saved to {weights_path}")
    print(f"Training state saved to {state_path} (step={final_step})")


if __name__ == "__main__":
    main()
