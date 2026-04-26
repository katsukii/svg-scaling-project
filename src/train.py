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


class EpochIterator:
    """Sequential, shuffled-window iterator for true epoch training.

    Divides the data into non-overlapping windows of size block_size+1,
    shuffles them, and yields batches without replacement.  Every window
    is consumed before the next epoch starts—including a final partial
    batch smaller than batch_size (if any).

    One full pass through all windows = one epoch.
    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        block_size: int,
        device: torch.device,
        seed: int = 42,
    ):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # Non-overlapping window start positions (each window is block_size+1 tokens)
        window_size = block_size + 1
        n_windows = len(data) // window_size
        self.all_starts = np.arange(n_windows) * window_size
        self.n_windows = n_windows
        self.rng = np.random.RandomState(seed)
        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle window order for a new epoch."""
        self.perm = self.rng.permutation(self.n_windows)
        self.cursor = 0

    @property
    def batches_per_epoch(self) -> int:
        """Number of batches per epoch (including final partial batch)."""
        return math.ceil(self.n_windows / self.batch_size)

    @property
    def remaining_in_epoch(self) -> int:
        """Number of batches left in the current epoch."""
        remaining_windows = len(self.perm) - self.cursor
        if remaining_windows <= 0:
            return 0
        return math.ceil(remaining_windows / self.batch_size)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch. Includes partial batches at epoch end.

        When all windows are exhausted, reshuffles for the next epoch.
        """
        if self.cursor >= len(self.perm):
            self._shuffle()

        end = min(self.cursor + self.batch_size, len(self.perm))
        idx = self.perm[self.cursor:end]
        self.cursor = end

        starts = self.all_starts[idx]
        bs = self.block_size
        x = torch.stack([
            torch.from_numpy(self.data[s:s + bs].astype(np.int64))
            for s in starts
        ])
        y = torch.stack([
            torch.from_numpy(self.data[s + 1:s + 1 + bs].astype(np.int64))
            for s in starts
        ])
        return x.to(self.device), y.to(self.device)

    def state_dict(self) -> dict:
        """Serialize iterator state for checkpointing."""
        return {
            'cursor': self.cursor,
            'perm': self.perm.tolist(),
            'rng_state': self.rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore iterator state from a checkpoint."""
        self.cursor = state['cursor']
        self.perm = np.array(state['perm'])
        self.rng.set_state(state['rng_state'])


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch (used for evaluation only).

    For training, use EpochIterator instead to ensure true epoch semantics.
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


def get_lr_factor(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_frac: float,
) -> float:
    """Compute LR as a fraction of peak LR (0..1 range).

    Used for µP: MuAdamW stores per-group width-scaled LRs at init time,
    so the scheduler must multiply by a *relative* factor rather than
    overwriting with an absolute value.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr_frac
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + math.cos(math.pi * progress))


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
        # Save initial per-group LRs so the scheduler can scale relatively.
        # MuAdamW sets width-dependent LRs; absolute overwrites would erase them.
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
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

    # Create epoch iterator for sequential, without-replacement training
    train_iter = EpochIterator(train_data, batch_size, block_size, device)

    # Compute training steps (optimizer steps, not micro-batches)
    # batches_per_epoch includes the final partial batch (ceil division),
    # so every window is consumed.  The training loop respects epoch
    # boundaries: it never pulls batches from the next epoch into the
    # current optimizer step's accumulation.  Each optimizer step uses
    # min(grad_accum_steps, remaining_in_epoch) micro-batches.
    batches_per_epoch = train_iter.batches_per_epoch
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)

    if args.max_steps:
        max_steps = args.max_steps
    elif train_cfg.get('max_steps'):
        max_steps = train_cfg['max_steps']
    else:
        # 1 epoch = every non-overlapping window seen once
        max_steps = steps_per_epoch

    warmup_steps = int(max_steps * train_cfg['warmup_frac'])
    max_lr = train_cfg['learning_rate']
    min_lr = max_lr * train_cfg['min_lr_frac']
    n_windows = train_iter.n_windows
    print(f"Windows: {n_windows} (each {block_size+1} tok, "
          f"covers {n_windows * (block_size+1):,}/{len(train_data):,} tokens)")
    print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")
    print(f"Steps per epoch: {steps_per_epoch} (batches_per_epoch={batches_per_epoch})")
    print(f"Batch size: {batch_size}, Block size: {block_size}, "
          f"Grad accum: {grad_accum_steps}")
    print(f"Effective tokens/step: {effective_batch_tokens:,}")

    # Resume from checkpoint
    start_step = 0
    resume_tokens_seen = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        resume_tokens_seen = ckpt.get('total_tokens_seen', 0)
        if 'train_iter' in ckpt:
            train_iter.load_state_dict(ckpt['train_iter'])
            print(f"  Resumed at step {start_step}, tokens_seen={resume_tokens_seen:,} "
                  f"(iterator state restored)")
        else:
            # Legacy checkpoint without iterator state: fast-forward
            # the iterator so it doesn't replay the same data order.
            # Advance cursor by the number of batches already consumed.
            batches_consumed = start_step * grad_accum_steps
            for _ in range(batches_consumed):
                train_iter.get_batch()
            print(f"  Resumed at step {start_step} (iterator fast-forwarded {batches_consumed} batches)")

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
    total_tokens_seen = resume_tokens_seen  # exact cumulative count

    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for step in range(start_step, max_steps):
        step_start = time.time()

        # Update learning rate
        if args.mup:
            # µP: scale each group's width-dependent LR by a relative factor
            factor = get_lr_factor(step, warmup_steps, max_steps,
                                   train_cfg['min_lr_frac'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * factor
            lr = max_lr * factor  # for logging
        else:
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Forward + backward with gradient accumulation.
        # Clamp micro-steps to remaining batches in epoch so that
        # accumulation never crosses an epoch boundary.
        actual_accum = min(grad_accum_steps, max(1, train_iter.remaining_in_epoch))
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        step_tokens = 0
        for micro_step in range(actual_accum):
            x, y = train_iter.get_batch()
            step_tokens += x.shape[0] * block_size
            _, loss = model(x, y)
            loss = loss / actual_accum  # scale for accumulation
            loss.backward()
            accum_loss += loss.item()
        total_tokens_seen += step_tokens

        # Gradient clipping
        if train_cfg['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])

        optimizer.step()

        # Track throughput and GPU memory
        step_time = time.time() - step_start
        tokens_per_sec = step_tokens / step_time
        throughput_samples.append(tokens_per_sec)

        if device.type == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if gpu_mem > peak_gpu_mem_mb:
                peak_gpu_mem_mb = gpu_mem

        # Logging
        if step % train_cfg['log_interval'] == 0:
            dt = time.time() - t0
            gpu_str = f" | gpu_mem {peak_gpu_mem_mb:.0f}MB" if device.type == 'cuda' else ""
            print(f"step {step:6d}/{max_steps} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {dt:.1f}s | {total_tokens_seen:,} tok | "
                  f"{tokens_per_sec:.0f} tok/s{gpu_str}")
            log_entry = {
                'step': step,
                'train_loss': accum_loss,
                'lr': lr,
                'time': dt,
                'tokens_seen': total_tokens_seen,
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
                'train_iter': train_iter.state_dict(),
                'total_tokens_seen': total_tokens_seen,
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
    # The last completed step index is max_steps - 1 (loop runs
    # range(start_step, max_steps)).  Storing the actual last step
    # ensures resume with start_step = ckpt['step'] + 1 is correct.
    last_step = max_steps - 1
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    torch.save({
        'step': last_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_iter': train_iter.state_dict(),
        'total_tokens_seen': total_tokens_seen,
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
        'total_tokens_seen': total_tokens_seen,
        'effective_batch_tokens': effective_batch_tokens,
        'grad_accum_steps': grad_accum_steps,
        'steps_per_epoch': steps_per_epoch,
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
