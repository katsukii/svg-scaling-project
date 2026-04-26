"""BPE tokenizer training and data tokenization for SVG dataset.

Trains a byte-level BPE tokenizer on cleaned SVG data, then tokenizes
train/val/test splits into binary files for efficient training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def train_tokenizer(train_path: Path, vocab_size: int, output_path: Path) -> Tokenizer:
    """Train a BPE tokenizer on the training data.

    Args:
        train_path: path to train.jsonl
        vocab_size: target vocabulary size
        output_path: path to save the trained tokenizer

    Returns:
        Trained tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=True,
    )

    # tokenizers.train() reads files line-by-line, so we use an iterator instead
    def svg_iterator():
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    yield record['svg']

    tokenizer.train_from_iterator(svg_iterator(), trainer, length=None)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def tokenize_split(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    max_token_len: int | None = 2048,
) -> dict:
    """Tokenize a data split and save as numpy binary.

    Each SVG is wrapped with BOS/EOS tokens, then all token IDs are
    concatenated into a single flat array (GPT-style training data).
    This gives the model explicit document boundaries:
        ... <eos> <bos> <svg ...> </svg> <eos> <bos> <svg ...> ...

    Args:
        tokenizer: trained tokenizer
        input_path: path to split .jsonl file
        output_path: path to save .bin file
        max_token_len: max sequence length (incl. BOS/EOS); None = no filter

    Returns:
        dict with tokenization statistics
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    all_ids: list[int] = []
    seq_lengths: list[int] = []
    filtered_too_long = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            svg_text = record['svg']
            encoded = tokenizer.encode(svg_text)
            ids = [bos_id] + encoded.ids + [eos_id]

            if max_token_len is not None and len(ids) > max_token_len:
                filtered_too_long += 1
                continue

            all_ids.extend(ids)
            seq_lengths.append(len(ids))

    # Save as uint16 numpy array (vocab_size <= 65535)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(str(output_path))

    seq_lengths_sorted = sorted(seq_lengths)
    n = len(seq_lengths)
    stats = {
        'num_sequences': n,
        'total_tokens': len(all_ids),
        'filtered_too_long': filtered_too_long,
        'min_len': int(min(seq_lengths)) if n else 0,
        'max_len': int(max(seq_lengths)) if n else 0,
        'median_len': int(seq_lengths_sorted[n // 2]) if n else 0,
        'mean_len': round(sum(seq_lengths) / n, 1) if n else 0,
        'p90_len': int(seq_lengths_sorted[int(n * 0.9)]) if n else 0,
        'p95_len': int(seq_lengths_sorted[int(n * 0.95)]) if n else 0,
        'p99_len': int(seq_lengths_sorted[int(n * 0.99)]) if n else 0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer and tokenize data')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                        help='Directory with train.jsonl, val.jsonl, test.jsonl')
    parser.add_argument('--tokenizer-dir', type=str, default='tokenizer',
                        help='Directory to save tokenizer')
    parser.add_argument('--output-dir', type=str, default='data/tokenized',
                        help='Directory to save tokenized .bin files')
    parser.add_argument('--vocab-size', type=int, default=4096,
                        help='BPE vocabulary size')
    parser.add_argument('--max-token-len', type=int, default=2048,
                        help='Max token sequence length filter (default: 2048, 0 = no filter)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer on training data
    print("Training BPE tokenizer...")
    tokenizer_path = tokenizer_dir / 'bpe_svg.json'
    tokenizer = train_tokenizer(input_dir / 'train.jsonl', args.vocab_size, tokenizer_path)

    # Tokenize each split
    max_token_len = args.max_token_len if args.max_token_len > 0 else None
    print(f"Max token length filter: {max_token_len}")

    all_stats = {}
    for split in ['train', 'val', 'test']:
        print(f"\nTokenizing {split}...")
        stats = tokenize_split(
            tokenizer,
            input_dir / f'{split}.jsonl',
            output_dir / f'{split}.bin',
            max_token_len=max_token_len,
        )
        all_stats[split] = stats
        print(f"  Sequences: {stats['num_sequences']}")
        print(f"  Filtered (too long): {stats['filtered_too_long']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Token lengths: min={stats['min_len']}, median={stats['median_len']}, "
              f"mean={stats['mean_len']}, max={stats['max_len']}")
        print(f"  Percentiles: p90={stats['p90_len']}, p95={stats['p95_len']}, p99={stats['p99_len']}")

    # Check 100M token minimum (project spec requirement)
    train_tokens = all_stats['train']['total_tokens']
    if train_tokens < 100_000_000:
        print(f"\n  [WARN] Training set has {train_tokens:,} tokens, below the 100M minimum.")
        print(f"         Consider raising --max-token-len or adding supplementary datasets.")

    # Save stats
    stats_path = output_dir / 'tokenize_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Quick decode test
    print(f"\n--- Decode test ---")
    with open(input_dir / 'train.jsonl', 'r') as f:
        sample = json.loads(f.readline().strip())['svg']
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    match = sample == decoded
    print(f"  Round-trip match: {match}")
    if not match:
        print(f"  Original[:100]: {sample[:100]}")
        print(f"  Decoded[:100]:  {decoded[:100]}")


if __name__ == '__main__':
    main()
