"""Training script for SVG language model.

Usage:
    python src/train.py --config configs/tiny.yaml
    python src/train.py --config configs/tiny.yaml --max-steps 1000  # override
    python src/train.py --config configs/tiny.yaml --mup             # µP mode
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


def setup_mup(model: GPT, config: ModelConfig, device: torch.device) -> None:
    """Set up µP base shapes on the model.

    Creates a base model (with base_width) and a delta model (base_width + 1)
    to compute the base shapes for µP scaling. This enables MuAdamW to
    automatically scale learning rates per layer.
    """
    from mup import set_base_shapes, make_base_shapes

    base_config = ModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.mup_base_width,
        d_ff=config.mup_base_width * (config.d_ff // config.n_embd),
        dropout=config.dropout,
        bias=config.bias,
        mup=True,
        mup_base_width=config.mup_base_width,
    )
    # n_head must divide base width — use same n_head if divisible, else 1
    if config.mup_base_width % config.n_head == 0:
        base_config.n_head = config.n_head
    else:
        base_config.n_head = 1

    delta_config = ModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=base_config.n_head,
        n_embd=config.mup_base_width + base_config.n_head,  # delta = base + 1 per head
        d_ff=base_config.d_ff + (config.d_ff // config.n_embd) * base_config.n_head,
        dropout=config.dropout,
        bias=config.bias,
        mup=True,
        mup_base_width=config.mup_base_width,
    )

    base_model = GPT(base_config)
    delta_model = GPT(delta_config)
    base_shapes = make_base_shapes(base_model, delta_model)
    set_base_shapes(model, base_shapes)
    del base_model, delta_model
    print("  µP base shapes set successfully")


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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
        dropout=model_cfg.get('dropout', 0.0),
        bias=model_cfg.get('bias', False),
        mup=args.mup,
        mup_base_width=args.mup_base_width,
    )
    model = GPT(mc)

    # µP setup: set base shapes before moving to device
    if args.mup:
        setup_mup(model, mc, device)

    model = model.to(device)
    n_params = model.get_num_params()
    param_mode = "µP" if args.mup else "SP"
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M) [{param_mode}]")
    if args.mup:
        width_mult = mc.n_embd / mc.mup_base_width
        print(f"  µP width_mult={width_mult:.3f}, base_width={args.mup_base_width}")

    # Optimizer
    if args.mup:
        from mup import MuAdamW
        # MuAdamW automatically scales LR per layer based on base shapes
        decay_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': train_cfg['weight_decay']},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = MuAdamW(optim_groups,
                            lr=train_cfg['learning_rate'],
                            betas=tuple(train_cfg['betas']))
    else:
        optimizer = model.configure_optimizers(
            weight_decay=train_cfg['weight_decay'],
            learning_rate=train_cfg['learning_rate'],
            betas=tuple(train_cfg['betas']),
            device_type=device.type,
        )

    # Gradient accumulation and batch size
    block_size = model_cfg['block_size']
    batch_size = train_cfg['batch_size']
    grad_accum_steps = train_cfg.get('grad_accum_steps', 1)
    effective_batch_tokens = batch_size * block_size * grad_accum_steps

    # Compute training steps (optimizer steps, not micro-batches)
    if args.max_steps:
        max_steps = args.max_steps
    elif train_cfg.get('max_steps'):
        max_steps = train_cfg['max_steps']
    else:
        # 1 epoch = process all training tokens once
        max_steps = len(train_data) // effective_batch_tokens

    warmup_steps = int(max_steps * train_cfg['warmup_frac'])
    max_lr = train_cfg['learning_rate']
    min_lr = max_lr * train_cfg['min_lr_frac']
    print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")
    print(f"Batch size: {batch_size}, Block size: {block_size}, "
          f"Grad accum: {grad_accum_steps}")
    print(f"Effective tokens/step: {effective_batch_tokens:,}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        print(f"  Resumed at step {start_step}")

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
    throughput_samples = []  # for averaging
    peak_gpu_mem_mb = 0.0

    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for step in range(start_step, max_steps):
        step_start = time.time()

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward + backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = get_batch(train_data, batch_size, block_size, device)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps  # scale for accumulation
            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        if train_cfg['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])

        optimizer.step()

        # Track throughput and GPU memory
        step_time = time.time() - step_start
        tokens_per_sec = effective_batch_tokens / step_time
        throughput_samples.append(tokens_per_sec)

        if device.type == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if gpu_mem > peak_gpu_mem_mb:
                peak_gpu_mem_mb = gpu_mem

        # Logging
        if step % train_cfg['log_interval'] == 0:
            dt = time.time() - t0
            tokens_seen = (step + 1) * effective_batch_tokens
            gpu_str = f" | gpu_mem {peak_gpu_mem_mb:.0f}MB" if device.type == 'cuda' else ""
            print(f"step {step:6d}/{max_steps} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {dt:.1f}s | {tokens_seen:,} tok | "
                  f"{tokens_per_sec:.0f} tok/s{gpu_str}")
            log_entry = {
                'step': step,
                'train_loss': accum_loss,
                'lr': lr,
                'time': dt,
                'tokens_seen': tokens_seen,
                'tokens_per_sec': tokens_per_sec,
            }
            if device.type == 'cuda':
                log_entry['gpu_mem_mb'] = peak_gpu_mem_mb
            log_entries.append(log_entry)

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

    avg_throughput = sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0.0

    print(f"\n--- Training complete ---")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"  Final train_loss: {losses['train']:.4f}")
    print(f"  Final val_loss: {losses['val']:.4f}")
    print(f"  Final val_ppl: {math.exp(losses['val']):.2f}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Avg throughput: {avg_throughput:,.0f} tok/s")
    if device.type == 'cuda':
        print(f"  Peak GPU memory: {peak_gpu_mem_mb:.0f} MB")

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
        'width_mult': mc.n_embd / mc.mup_base_width if args.mup else None,
        'effective_batch_tokens': effective_batch_tokens,
        'grad_accum_steps': grad_accum_steps,
        'avg_tokens_per_second': avg_throughput,
        'peak_gpu_memory_mb': peak_gpu_mem_mb if device.type == 'cuda' else None,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save training log
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
