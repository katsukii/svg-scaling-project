"""Coordinate check: verify µP implementation correctness.

Trains models of different widths for a few steps on the same fixed batch,
then plots activation norms at each layer.

Expected behavior:
  - SP: activation norms grow with width (lines slope upward)
  - µP: activation norms stay constant across widths (flat lines)

Usage:
    python scripts/coord_check.py
    python scripts/coord_check.py --device mps --steps 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import ModelConfig, GPT


def get_fixed_batch(
    data_path: str, batch_size: int, block_size: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a deterministic batch from tokenized data."""
    data = np.fromfile(data_path, dtype=np.uint16)
    # Always use the first batch_size * block_size tokens for reproducibility
    x_list, y_list = [], []
    for i in range(batch_size):
        start = i * block_size
        x_list.append(torch.from_numpy(data[start:start + block_size].astype(np.int64)))
        y_list.append(torch.from_numpy(data[start + 1:start + 1 + block_size].astype(np.int64)))
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


def collect_activation_norms(
    model: GPT, x: torch.Tensor, y: torch.Tensor,
) -> dict[str, float]:
    """Run forward pass and collect L2 norm of activations at each layer."""
    norms = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            norms[name] = output.detach().float().norm().item()
        return hook_fn

    # Register hooks on key layers
    hooks.append(model.transformer.wte.register_forward_hook(make_hook("wte")))
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.c_proj.register_forward_hook(make_hook(f"block{i}.attn")))
        hooks.append(block.mlp.c_proj.register_forward_hook(make_hook(f"block{i}.mlp")))
    hooks.append(model.transformer.ln_f.register_forward_hook(make_hook("ln_f")))

    model.train()
    _, loss = model(x, y)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return norms


def run_coord_check(
    widths: list[int],
    n_steps: int,
    mup: bool,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    lr: float = 3e-3,
) -> dict[str, list[float]]:
    """Train models of different widths and collect activation norms after n_steps.

    Returns: {layer_name: [norm_for_width0, norm_for_width1, ...]}
    """
    all_norms: dict[str, list[float]] = {}

    for width in widths:
        # Use fixed n_head that divides width (min 2 heads, head_dim >= 16)
        n_head = max(2, width // 32)
        while width % n_head != 0:
            n_head -= 1

        config = ModelConfig(
            vocab_size=4096,
            block_size=1024,
            n_layer=4,
            n_head=n_head,
            n_embd=width,
            d_ff=width * 4,
            dropout=0.0,
            bias=False,
            mup=mup,
            mup_base_width=128,
        )

        torch.manual_seed(42)
        model = GPT(config)

        if mup:
            # Use mup package for proper base shape setup
            from train import setup_mup
            setup_mup(model, config, device)

        model = model.to(device)

        if mup:
            from mup import MuAdamW
            decay_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() >= 2]
            nodecay_params = [p for n, p in model.named_parameters()
                              if p.requires_grad and p.dim() < 2]
            optimizer = MuAdamW([
                {'params': decay_params, 'weight_decay': 0.1},
                {'params': nodecay_params, 'weight_decay': 0.0},
            ], lr=lr, betas=(0.9, 0.95))
        else:
            optimizer = model.configure_optimizers(
                weight_decay=0.1, learning_rate=lr,
                betas=(0.9, 0.95), device_type=device.type,
            )

        # Train for n_steps on the fixed batch
        model.train()
        for _ in range(n_steps):
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Collect activation norms
        norms = collect_activation_norms(model, x, y)
        mode = "µP" if mup else "SP"
        print(f"  {mode} width={width:4d}: loss={loss.item():.4f}, "
              f"norms={{{', '.join(f'{k}: {v:.2f}' for k, v in sorted(norms.items()))}}}")

        for name, val in norms.items():
            if name not in all_norms:
                all_norms[name] = []
            all_norms[name].append(val)

        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return all_norms


def plot_coord_check(
    widths: list[int],
    sp_norms: dict[str, list[float]],
    mup_norms: dict[str, list[float]],
    output_path: str,
) -> None:
    """Plot SP vs µP activation norms across widths."""
    layer_names = sorted(sp_norms.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, (norms, title) in zip(axes, [(sp_norms, "SP (Standard)"), (mup_norms, "µP")]):
        for name in layer_names:
            ax.plot(widths, norms[name], 'o-', label=name, markersize=5)
        ax.set_xlabel("Width (n_embd)")
        ax.set_ylabel("Activation L2 Norm")
        ax.set_title(f"{title}: Activation Norms vs Width")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Coordinate check for µP verification")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=3, help="Training steps per model")
    parser.add_argument("--lr", type=float, default=3e-3, help="Base learning rate")
    parser.add_argument("--data-path", type=str, default="data/tokenized/train.bin")
    parser.add_argument("--output", type=str, default="results/plots/coord_check.png")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    widths = [64, 128, 256, 512, 768]
    batch_size = 4
    block_size = 1024

    print("Loading fixed batch...")
    x, y = get_fixed_batch(args.data_path, batch_size, block_size, device)

    print(f"\n--- SP (Standard Parameterization) ---")
    sp_norms = run_coord_check(widths, args.steps, mup=False, x=x, y=y,
                               device=device, lr=args.lr)

    print(f"\n--- µP (Maximal Update Parameterization) ---")
    mup_norms = run_coord_check(widths, args.steps, mup=True, x=x, y=y,
                                device=device, lr=args.lr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_coord_check(widths, sp_norms, mup_norms, args.output)


if __name__ == "__main__":
    main()
