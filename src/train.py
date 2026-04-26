"""Training script for SVG language model.

Usage:
    python src/train.py --config configs/tiny.yaml
    python src/train.py --config configs/tiny.yaml --max-steps 1000  # override
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from model import ModelConfig, GPT


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of sequences from the data.

    The data is a flat array of token IDs. We pick random starting
    positions and extract sequences of length block_size.

    Returns:
        (x, y) where x is input and y is target (shifted by 1)
    """
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    block_size: int,
    eval_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimate train and val loss over eval_steps batches."""
    model.eval()
    losses = {}
    for split_name, data in [('train', train_data), ('val', val_data)]:
        total_loss = 0.0
        for _ in range(eval_steps):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split_name] = total_loss / eval_steps
    model.train()
    return losses


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description='Train SVG language model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--max-steps', type=int, default=None, help='Override max steps')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/mps)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--mup', action='store_true',
                        help='Enable µP (Maximal Update Parameterization)')
    parser.add_argument('--mup-base-width', type=int, default=128,
                        help='Base width for µP (default: 128 = Tiny n_embd)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    data_cfg = cfg['data']

    if args.learning_rate is not None:
        train_cfg['learning_rate'] = args.learning_rate

    # Resolve paths relative to the config file's parent directory
    config_dir = Path(args.config).resolve().parent.parent
    for key in ['train_path', 'val_path']:
        p = Path(data_cfg[key])
        if not p.is_absolute():
            data_cfg[key] = str(config_dir / p)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading tokenized data...")
    train_data = np.fromfile(data_cfg['train_path'], dtype=np.uint16)
    val_data = np.fromfile(data_cfg['val_path'], dtype=np.uint16)
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Create model
    mc = ModelConfig(
        vocab_size=model_cfg['vocab_size'],
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout'],
        bias=model_cfg['bias'],
        mup=args.mup,
        mup_base_width=args.mup_base_width,
    )
    model = GPT(mc).to(device)
    n_params = model.get_num_params()
    param_mode = "µP" if args.mup else "SP"
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M) [{param_mode}]")
    if args.mup:
        print(f"  µP width_mult={model.width_mult:.3f}, base_width={args.mup_base_width}")

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=train_cfg['weight_decay'],
        learning_rate=train_cfg['learning_rate'],
        betas=tuple(train_cfg['betas']),
        device_type=device.type,
    )

    # Compute training steps
    block_size = model_cfg['block_size']
    batch_size = train_cfg['batch_size']
    tokens_per_step = batch_size * block_size

    if args.max_steps:
        max_steps = args.max_steps
    elif train_cfg.get('max_steps'):
        max_steps = train_cfg['max_steps']
    else:
        # 1 epoch
        max_steps = len(train_data) // tokens_per_step

    warmup_steps = int(max_steps * train_cfg['warmup_frac'])
    max_lr = train_cfg['learning_rate']
    min_lr = max_lr * train_cfg['min_lr_frac']
    print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")
    print(f"Tokens per step: {tokens_per_step:,}")

    # Output directory
    config_name = Path(args.config).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f'results/runs/{config_name}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Training loop
    log_entries = []
    best_val_loss = float('inf')
    t0 = time.time()

    model.train()
    for step in range(max_steps):
        # Update learning rate (µP: apply per-group lr_scale)
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

        # Forward + backward
        x, y = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(x, y)
        loss.backward()

        # Gradient clipping
        if train_cfg['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % train_cfg['log_interval'] == 0:
            dt = time.time() - t0
            tokens_seen = (step + 1) * tokens_per_step
            # Show per-group LR for µP verification (only on first log)
            if args.mup and step == 0:
                for i, pg in enumerate(optimizer.param_groups):
                    print(f"  param_group[{i}]: lr={pg['lr']:.2e}, "
                          f"lr_scale={pg.get('lr_scale', 1.0):.4f}, "
                          f"n_params={sum(p.numel() for p in pg['params']):,}")
            print(f"step {step:6d}/{max_steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {dt:.1f}s | {tokens_seen:,} tokens")
            log_entries.append({
                'step': step,
                'train_loss': loss.item(),
                'lr': lr,
                'time': dt,
                'tokens_seen': tokens_seen,
            })

        # Evaluation
        if step > 0 and step % train_cfg['eval_interval'] == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                batch_size, block_size, train_cfg['eval_steps'], device,
            )
            print(f"  → eval train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}, "
                  f"val_ppl={math.exp(losses['val']):.2f}")
            log_entries.append({
                'step': step,
                'eval_train_loss': losses['train'],
                'eval_val_loss': losses['val'],
                'eval_val_ppl': math.exp(losses['val']),
            })
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
                print(f"  → New best val_loss={best_val_loss:.4f}, saved checkpoint")

        # Periodic checkpoint
        if step > 0 and train_cfg['checkpoint_interval'] and step % train_cfg['checkpoint_interval'] == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg,
            }, output_dir / f'checkpoint_{step}.pt')

    # Final evaluation
    losses = estimate_loss(
        model, train_data, val_data,
        batch_size, block_size, train_cfg['eval_steps'], device,
    )
    total_time = time.time() - t0

    # Update best checkpoint if final eval is better
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        torch.save(model.state_dict(), output_dir / 'best_model.pt')
        print(f"  → Final eval is new best val_loss={best_val_loss:.4f}, saved checkpoint")

    print(f"\n--- Training complete ---")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Final train_loss: {losses['train']:.4f}")
    print(f"  Final val_loss: {losses['val']:.4f}")
    print(f"  Final val_ppl: {math.exp(losses['val']):.2f}")
    print(f"  Best val_loss: {best_val_loss:.4f}")

    # Save final model + logs
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    torch.save({
        'step': max_steps,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': cfg,
    }, output_dir / 'final_checkpoint.pt')

    # Save config + training summary
    summary = {
        'config': cfg,
        'config_name': config_name,
        'n_params': n_params,
        'max_steps': max_steps,
        'total_time_s': total_time,
        'final_train_loss': losses['train'],
        'final_val_loss': losses['val'],
        'final_val_ppl': math.exp(losses['val']),
        'best_val_loss': best_val_loss,
        'device': str(device),
        'mup': args.mup,
        'mup_base_width': args.mup_base_width if args.mup else None,
        'width_mult': model.width_mult if args.mup else None,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save training log
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
