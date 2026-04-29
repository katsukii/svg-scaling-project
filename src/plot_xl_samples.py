"""Generate representative XL-only sample figures for the report main text.

Produces two figures:
1. xl_unconditional_samples.png — 5 unconditional samples in a single row
2. xl_prefix_samples.png — 3 prefix completions (one per prefix type),
   each shown as a prefix→completion→rendered triplet

These replace the full cross-size grids (moved to Appendix) in the main text.
"""

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 200
DPI = 300
ROOT = Path(__file__).resolve().parent.parent


def render_svg(svg_text: str, size: int = CELL_SIZE) -> Image.Image | None:
    """Render SVG text to a PIL Image, with lxml recovery for malformed XML."""
    def _render(data: bytes) -> Image.Image:
        png_data = cairosvg.svg2png(
            bytestring=data, output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert('RGB')

    # Try direct render first
    try:
        return _render(svg_text.encode('utf-8'))
    except Exception:
        pass

    # Fallback: use lxml recovery to fix malformed XML
    try:
        from lxml import etree
        parser = etree.XMLParser(recover=True)
        tree = etree.fromstring(svg_text.encode('utf-8'), parser)
        fixed = etree.tostring(tree, encoding='unicode')
        return _render(fixed.encode('utf-8'))
    except Exception as e:
        print(f'  Render failed: {e}')
        return None


def blank(size: int = CELL_SIZE, label: str = '') -> Image.Image:
    img = Image.new('RGB', (size, size), (240, 240, 240))
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, size // 2 - 5), label, fill=(150, 150, 150))
    return img


def get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)
    except (OSError, IOError):
        return ImageFont.load_default()


def generate_unconditional_figure() -> None:
    """Generate XL unconditional 5-sample single-row figure."""
    samples_dir = ROOT / 'results' / 'samples_v2' / 'unconditional' / 'topp_t0.8'
    svg_files = sorted(samples_dir.glob('sample_*.svg'))[:5]

    if len(svg_files) < 5:
        print(f'WARNING: Only {len(svg_files)} SVG files found, expected 5')

    padding = 4
    cols = len(svg_files)
    grid_w = cols * CELL_SIZE + (cols + 1) * padding
    grid_h = CELL_SIZE + 2 * padding

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for i, svg_path in enumerate(svg_files):
        svg_text = svg_path.read_text()
        img = render_svg(svg_text) or blank(label='render fail')
        x = padding + i * (CELL_SIZE + padding)
        grid.paste(img, (x, padding))
        print(f'  Unconditional sample {i}: {svg_path.name}')

    output = ROOT / 'report' / 'figures' / 'xl_unconditional_samples.png'
    grid.save(str(output), dpi=(DPI, DPI))
    print(f'Saved: {output}')


def generate_prefix_figure() -> None:
    """Generate XL prefix completion figure: 3 rendered completions in a row.

    Each column shows the rendered result of one prefix type completion.
    Labels indicate the prefix type.
    """
    import re

    samples_dir = ROOT / 'results' / 'samples_v2' / 'prefix_conditioned'

    prefix_types = ['face_partial', 'open_path', 'single_shape_group']
    display_names = ['Face (partial)', 'Open path', 'Shape group']

    padding = 4
    label_h = 22
    n_types = len(prefix_types)

    grid_w = n_types * CELL_SIZE + (n_types + 1) * padding
    grid_h = CELL_SIZE + label_h + 2 * padding

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = get_font(13)

    for i, (ptype, dname) in enumerate(zip(prefix_types, display_names)):
        x = padding + i * (CELL_SIZE + padding)

        # Label
        draw.text((x + 4, padding), dname, fill='black', font=font)

        # Find first renderable completion
        sample_dir = samples_dir / ptype
        svg_files = sorted(sample_dir.glob('sample_*.svg'))
        comp_img = None
        for svg_file in svg_files:
            svg_text = svg_file.read_text()
            # Fix duplicate attributes by removing duplicates
            svg_text = re.sub(
                r'(\s)(stroke-opacity|fill-opacity|opacity)="[^"]*"\s+\2="([^"]*)"',
                r'\1\2="\3"',
                svg_text,
            )
            img = render_svg(svg_text)
            if img is not None:
                comp_img = img
                print(f'  Prefix {ptype}: using {svg_file.name}')
                break

        if comp_img is None:
            comp_img = blank(label='no render')
            print(f'  Prefix {ptype}: no renderable samples')

        grid.paste(comp_img, (x, padding + label_h))

    output = ROOT / 'report' / 'figures' / 'xl_prefix_samples.png'
    grid.save(str(output), dpi=(DPI, DPI))
    print(f'Saved: {output}')


if __name__ == '__main__':
    print('Generating XL unconditional samples figure...')
    generate_unconditional_figure()
    print('\nGenerating XL prefix completion figure...')
    generate_prefix_figure()
