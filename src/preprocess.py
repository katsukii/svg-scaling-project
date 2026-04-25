"""SVG data preprocessing pipeline.

Loads raw SVG data from HuggingFace dataset, cleans it, and saves
processed data for tokenization.

Reference: nanoGPT data preparation approach, adapted for SVG domain.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from lxml import etree
from datasets import load_from_disk


def clean_svg(svg_text: str) -> str | None:
    """Clean a single SVG string.

    Steps:
    1. Remove HTML/XML comments
    2. Strip leading/trailing whitespace
    3. Validate as well-formed XML via lxml

    Returns cleaned SVG string, or None if invalid.
    """
    # Remove comments
    svg_text = re.sub(r'<!--.*?-->', '', svg_text, flags=re.DOTALL)
    svg_text = svg_text.strip()

    if not svg_text:
        return None

    # Validate XML
    try:
        etree.fromstring(svg_text.encode('utf-8'))
    except etree.XMLSyntaxError:
        return None

    return svg_text


def process_split(dataset, split_name: str, output_dir: Path, max_len: int | None = None) -> dict:
    """Process a single dataset split.

    Args:
        dataset: HuggingFace dataset split
        split_name: 'train', 'val', or 'test'
        output_dir: directory to write output files
        max_len: optional max character length filter

    Returns:
        dict with processing statistics
    """
    stats = {
        'total': len(dataset),
        'valid': 0,
        'invalid_xml': 0,
        'too_long': 0,
        'char_lengths': [],
    }

    output_path = output_dir / f'{split_name}.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(dataset)):
            svg_text = dataset[i]['Svg']
            cleaned = clean_svg(svg_text)

            if cleaned is None:
                stats['invalid_xml'] += 1
                continue

            if max_len is not None and len(cleaned) > max_len:
                stats['too_long'] += 1
                continue

            stats['valid'] += 1
            stats['char_lengths'].append(len(cleaned))
            f.write(json.dumps({'svg': cleaned}) + '\n')

    return stats


def print_stats(stats: dict, split_name: str) -> None:
    """Print processing statistics for a split."""
    lengths = stats['char_lengths']
    print(f"\n--- {split_name} ---")
    print(f"  Total:       {stats['total']}")
    print(f"  Valid:       {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"  Invalid XML: {stats['invalid_xml']}")
    print(f"  Too long:    {stats['too_long']}")
    if lengths:
        lengths_sorted = sorted(lengths)
        print(f"  Char length: min={min(lengths)}, max={max(lengths)}, "
              f"median={lengths_sorted[len(lengths)//2]}, mean={sum(lengths)/len(lengths):.0f}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SVG dataset')
    parser.add_argument('--input-dir', type=str, default='data/raw/svg-icons-simple',
                        help='Path to HuggingFace dataset on disk')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for cleaned data')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Max character length filter (None = no filter)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_dir}...")
    ds = load_from_disk(str(input_dir))

    all_stats = {}
    for split_name in ['train', 'val', 'test']:
        print(f"Processing {split_name}...")
        stats = process_split(ds[split_name], split_name, output_dir, args.max_len)
        print_stats(stats, split_name)
        # Save lengths for later analysis, but not in the summary JSON
        all_stats[split_name] = {k: v for k, v in stats.items() if k != 'char_lengths'}

    # Save summary stats
    stats_path = output_dir / 'preprocess_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == '__main__':
    main()
