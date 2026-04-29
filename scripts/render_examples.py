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
    cell_size: int = 150,
    output_path: Path = Path('report/figures/dataset_examples.png'),
    dpi: int = 300,
) -> None:
    """Create a 1-row × 9-column horizontal grid of rendered SVG examples.

    Layout: [Simple (3 cells)] | [Medium (3 cells)] | [Complex (3 cells)]
    Each group has a centered label above its cells.
    """
    cat_names = ['simple', 'medium', 'complex']
    all_images: list[Image.Image] = []

    for cat in cat_names:
        svgs = categories[cat][:samples_per_cat * 3]  # try extra in case some fail
        count = 0
        for svg in svgs:
            if count >= samples_per_cat:
                break
            img = render_svg(svg, cell_size)
            if img is not None:
                all_images.append(img)
                count += 1
        # Pad with blanks if needed
        while count < samples_per_cat:
            all_images.append(Image.new('RGB', (cell_size, cell_size), (240, 240, 240)))
            count += 1

    from PIL import ImageDraw

    label_h = 22        # pixels for group label row at top
    group_gap = 8       # pixels between groups (visual separator)
    n_groups = len(cat_names)
    total_width = samples_per_cat * n_groups * cell_size + (n_groups - 1) * group_gap
    total_height = label_h + cell_size

    grid = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for g_idx, cat in enumerate(cat_names):
        x_start = g_idx * (samples_per_cat * cell_size + group_gap)
        group_w = samples_per_cat * cell_size

        # Draw vertical separator before each group (except first)
        if g_idx > 0:
            sep_x = x_start - group_gap // 2
            draw.line([(sep_x, 0), (sep_x, total_height)], fill=(160, 160, 160), width=1)

        # Draw group label centered above the group cells
        label = cat.capitalize()
        try:
            tw = draw.textlength(label)
        except AttributeError:
            tw = len(label) * 6  # rough estimate for older Pillow
        draw.text((x_start + (group_w - tw) // 2, 3), label, fill='black')

        # Paste cell images in the row below the label
        for c_idx in range(samples_per_cat):
            img = all_images[g_idx * samples_per_cat + c_idx]
            grid.paste(img, (x_start + c_idx * cell_size, label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), dpi=(dpi, dpi))
    print(f"Saved: {output_path}  ({total_width}x{total_height} px, {dpi} dpi)")
    print(f"  Categories: " + ", ".join(
        f"{cat}={len(categories[cat])}" for cat in cat_names))


def main():
    parser = argparse.ArgumentParser(description='Render example SVGs from dataset')
    parser.add_argument('--input', type=str, default='data/processed/train.jsonl',
                        help='Path to train.jsonl')
    parser.add_argument('--output', type=str, default='report/figures/dataset_examples.png',
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
