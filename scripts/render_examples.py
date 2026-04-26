"""Render example SVGs from the dataset at different complexity levels.

Selects simple, medium, and complex SVGs from the training data,
renders them via CairoSVG, and creates a grid image.

Usage:
    python scripts/render_examples.py
    python scripts/render_examples.py --input data/processed/train.jsonl --output results/plots/dataset_examples.png
"""

import argparse
import io
import json
import random
from pathlib import Path

import cairosvg
from PIL import Image


def categorize_svgs(input_path: Path, seed: int = 42) -> dict[str, list[str]]:
    """Categorize SVGs by character length into simple/medium/complex."""
    categories = {'simple': [], 'medium': [], 'complex': []}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            svg = record['svg']
            char_len = len(svg)

            if char_len < 500:
                categories['simple'].append(svg)
            elif char_len < 2000:
                categories['medium'].append(svg)
            else:
                categories['complex'].append(svg)

    rng = random.Random(seed)
    for cat in categories:
        rng.shuffle(categories[cat])

    return categories


def render_svg(svg_text: str, size: int = 200) -> Image.Image | None:
    """Render an SVG to a PIL Image."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode('utf-8'),
            output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert('RGB')
    except Exception:
        return None


def create_grid(
    categories: dict[str, list[str]],
    samples_per_cat: int = 3,
    cell_size: int = 200,
    output_path: Path = Path('results/plots/dataset_examples.png'),
) -> None:
    """Create a grid of rendered SVG examples."""
    cat_names = ['simple', 'medium', 'complex']
    images = []
    labels = []

    for cat in cat_names:
        svgs = categories[cat][:samples_per_cat * 3]  # try extra in case some fail
        count = 0
        for svg in svgs:
            if count >= samples_per_cat:
                break
            img = render_svg(svg, cell_size)
            if img is not None:
                images.append(img)
                labels.append(f"{cat} ({len(svg)} chars)")
                count += 1
        # Pad with blanks if needed
        while count < samples_per_cat:
            images.append(Image.new('RGB', (cell_size, cell_size), (240, 240, 240)))
            labels.append(f"{cat} (no render)")
            count += 1

    # Create grid: rows=categories, cols=samples_per_cat
    grid_cols = samples_per_cat
    grid_rows = len(cat_names)
    margin = 30  # top margin for labels
    grid_w = grid_cols * cell_size
    grid_h = grid_rows * (cell_size + margin)
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)

    for row_idx, cat in enumerate(cat_names):
        y_offset = row_idx * (cell_size + margin)
        draw.text((5, y_offset + 5), f"{cat.upper()}", fill='black')
        for col_idx in range(samples_per_cat):
            img_idx = row_idx * samples_per_cat + col_idx
            grid.paste(images[img_idx], (col_idx * cell_size, y_offset + margin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    print(f"Saved: {output_path}")
    print(f"  Categories: " + ", ".join(
        f"{cat}={len(categories[cat])}" for cat in cat_names))


def main():
    parser = argparse.ArgumentParser(description='Render example SVGs from dataset')
    parser.add_argument('--input', type=str, default='data/processed/train.jsonl',
                        help='Path to train.jsonl')
    parser.add_argument('--output', type=str, default='results/plots/dataset_examples.png',
                        help='Output path for grid image')
    parser.add_argument('--samples-per-cat', type=int, default=3,
                        help='Samples per complexity category')
    args = parser.parse_args()

    print(f"Loading SVGs from {args.input}...")
    categories = categorize_svgs(Path(args.input))

    print(f"Creating example grid...")
    create_grid(categories, args.samples_per_cat, output_path=Path(args.output))


if __name__ == '__main__':
    main()
