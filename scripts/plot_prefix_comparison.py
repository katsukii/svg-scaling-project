"""Create a prefix completion comparison grid.

For each prefix type, renders the prefix SVG and up to 2 model completions
side by side. Layout: rows = prefix types, cols = [prefix, completion1, completion2].

Usage:
    python scripts/plot_prefix_comparison.py
"""

import argparse
import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw


PREFIXES = ['face_partial', 'open_path', 'single_shape_group']
CELL_SIZE = 200


def render_svg(svg_text: str, size: int = CELL_SIZE) -> Image.Image | None:
    """Render SVG text to a PIL Image."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode('utf-8'),
            output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert('RGB')
    except Exception as e:
        print(f'  Render failed: {e}')
        return None


def blank(label: str = '') -> Image.Image:
    """Create a blank placeholder cell."""
    img = Image.new('RGB', (CELL_SIZE, CELL_SIZE), (240, 240, 240))
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, CELL_SIZE // 2 - 5), label, fill=(150, 150, 150))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix-dir', default='results/prefixes_v2')
    parser.add_argument('--samples-dir', default='results/samples_v2/prefix_conditioned')
    parser.add_argument('--output', default='results/plots/prefix_completion.png')
    parser.add_argument('--completions', type=int, default=2,
                        help='Number of completions to show per prefix')
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir)
    samples_dir = Path(args.samples_dir)
    n_comp = args.completions
    n_cols = 1 + n_comp  # prefix + completions
    margin_top = 25

    rows = []
    for prefix_name in PREFIXES:
        prefix_path = prefix_dir / f'{prefix_name}.svg'
        if not prefix_path.exists():
            print(f'SKIP {prefix_name}: prefix not found')
            continue

        prefix_svg = prefix_path.read_text()
        prefix_img = render_svg(prefix_svg) or blank('no render')

        completion_imgs = []
        sample_dir = samples_dir / prefix_name
        svg_files = sorted(sample_dir.glob('*.svg'))[:n_comp]
        for svg_file in svg_files:
            svg_text = svg_file.read_text()
            img = render_svg(svg_text)
            completion_imgs.append(img or blank('render fail'))
            print(f'  {svg_file.name}: rendered')

        while len(completion_imgs) < n_comp:
            completion_imgs.append(blank('incomplete'))

        rows.append((prefix_name, prefix_img, completion_imgs))

    if not rows:
        print('No prefixes rendered.')
        return

    # Build grid
    grid_w = n_cols * CELL_SIZE
    grid_h = len(rows) * (CELL_SIZE + margin_top)
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Column headers
    col_labels = ['Prefix'] + [f'Completion {i+1}' for i in range(n_comp)]

    for row_idx, (name, prefix_img, comp_imgs) in enumerate(rows):
        y = row_idx * (CELL_SIZE + margin_top)
        label = name.replace('_', ' ')
        draw.text((5, y + 3), label, fill='black')

        # Prefix
        grid.paste(prefix_img, (0, y + margin_top))
        # Completions
        for c_idx, c_img in enumerate(comp_imgs):
            grid.paste(c_img, ((1 + c_idx) * CELL_SIZE, y + margin_top))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    print(f'\nSaved: {output_path}')


if __name__ == '__main__':
    main()
