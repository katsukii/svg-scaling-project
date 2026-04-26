"""Plot token sequence length distribution histogram.

Reads tokenized binary data and plots a histogram of per-SVG token lengths,
with vertical lines at p50, p90, p95, p99, and block_size=1024.

Usage:
    python scripts/plot_token_histogram.py
    python scripts/plot_token_histogram.py --data data/tokenized/train.bin --tokenizer tokenizer/bpe_svg.json
    python scripts/plot_token_histogram.py --stats-output analysis/token_length_stats.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer


def get_sequence_lengths(data: np.ndarray, bos_id: int, eos_id: int) -> list[int]:
    """Split concatenated token array by BOS/EOS and compute sequence lengths."""
    lengths = []
    current_len = 0
    in_sequence = False

    for token in data:
        if token == bos_id:
            in_sequence = True
            current_len = 1  # count BOS
        elif token == eos_id and in_sequence:
            current_len += 1  # count EOS
            lengths.append(current_len)
            in_sequence = False
            current_len = 0
        elif in_sequence:
            current_len += 1

    return lengths


def main():
    parser = argparse.ArgumentParser(description='Plot token length histogram')
    parser.add_argument('--data', type=str, default='data/tokenized/train.bin',
                        help='Path to tokenized .bin file')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/bpe_svg.json',
                        help='Path to tokenizer JSON')
    parser.add_argument('--output', type=str, default='results/plots/token_length_histogram.png',
                        help='Output path for histogram image')
    parser.add_argument('--block-size', type=int, default=1024,
                        help='Model block size for reference line')
    parser.add_argument('--stats-output', type=str, default=None,
                        help='Path to save JSON statistics (p50, p90, p95, p99, etc.)')
    args = parser.parse_args()

    # Load tokenizer for special token IDs
    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Load data
    print(f"Loading {args.data}...")
    data = np.fromfile(args.data, dtype=np.uint16)
    print(f"  Total tokens: {len(data):,}")

    # Compute per-sequence lengths
    lengths = get_sequence_lengths(data, bos_id, eos_id)
    lengths = np.array(lengths)
    print(f"  Sequences: {len(lengths):,}")
    print(f"  Min: {lengths.min()}, Max: {lengths.max()}, "
          f"Median: {int(np.median(lengths))}, Mean: {lengths.mean():.1f}")

    p50 = int(np.percentile(lengths, 50))
    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    pct_above_2048 = float((lengths > 2048).sum() / len(lengths) * 100)
    print(f"  P50: {p50}, P90: {p90}, P95: {p95}, P99: {p99}")
    print(f"  % above 2048: {pct_above_2048:.1f}%")

    # Save JSON statistics
    if args.stats_output:
        stats = {
            'total_sequences': int(len(lengths)),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'pct_above_2048': pct_above_2048,
        }
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(p50, color='green', linestyle='--', linewidth=1.5, label=f'P50 = {p50}')
    ax.axvline(p90, color='purple', linestyle='--', linewidth=1.5, label=f'P90 = {p90}')
    ax.axvline(p95, color='orange', linestyle='--', linewidth=1.5, label=f'P95 = {p95}')
    ax.axvline(p99, color='magenta', linestyle='--', linewidth=1.5, label=f'P99 = {p99}')
    ax.axvline(args.block_size, color='red', linestyle='-', linewidth=1.5,
               label=f'block_size = {args.block_size}')

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Count')
    ax.set_title('Token Sequence Length Distribution (Training Set)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
