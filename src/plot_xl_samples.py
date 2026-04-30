"""Generate representative XL-only sample figures for the report main text.

Produces two figures:
1. xl_unconditional_samples.png — 5 unconditional samples in a single row
2. xl_prefix_samples.png — 3 prefix completions (one per prefix type),
   each shown as a [rendered prefix | completion text | rendered result] triplet

These replace the full cross-size grids (moved to Appendix) in the main text.
"""

import io
import re
import textwrap
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 200
DPI = 300
ROOT = Path(__file__).resolve().parent.parent


def render_svg_strict(svg_text: str, size: int = CELL_SIZE) -> Image.Image | None:
    """Render SVG text to a PIL Image using CairoSVG only (no lxml recovery)."""
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


def _render_partial_svg(svg_text: str, size: int = CELL_SIZE) -> Image.Image | None:
    """Render a partial SVG (prefix) by using lxml recovery to close open tags."""
    try:
        from lxml import etree
        parser = etree.XMLParser(recover=True)
        tree = etree.fromstring(svg_text.encode('utf-8'), parser)
        fixed = etree.tostring(tree, encoding='unicode')
        return render_svg_strict(fixed, size)
    except Exception:
        return None


def render_failed_placeholder(size: int = CELL_SIZE) -> Image.Image:
    """Gray placeholder with 'render failed' label."""
    img = Image.new('RGB', (size, size), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    draw.text((size // 2, size // 2), 'render failed',
              fill=(120, 120, 120), font=font, anchor='mm')
    return img


def blank(size: int = CELL_SIZE, label: str = '') -> Image.Image:
    img = Image.new('RGB', (size, size), (240, 240, 240))
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, size // 2 - 5), label, fill=(150, 150, 150))
    return img


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/Courier.dfont',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _get_label_font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _render_text_cell(text: str, width: int = CELL_SIZE,
                      height: int = CELL_SIZE) -> Image.Image:
    """Render a text snippet onto a white-background image with monospace font."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(9)

    # Wrap text to fit cell width (~35 chars per line at font size 9)
    wrapped = textwrap.fill(text, width=35)
    # Truncate if too many lines
    lines = wrapped.split('\n')[:18]
    truncated = '\n'.join(lines)

    draw.multiline_text((6, 6), truncated, fill=(30, 30, 30), font=font,
                        spacing=2)
    return img


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
        img = render_svg_strict(svg_text) or blank(label='render fail')
        x = padding + i * (CELL_SIZE + padding)
        grid.paste(img, (x, padding))
        print(f'  Unconditional sample {i}: {svg_path.name}')

    output = ROOT / 'report' / 'figures' / 'xl_unconditional_samples.png'
    grid.save(str(output), dpi=(DPI, DPI))
    print(f'Saved: {output}')


def generate_prefix_figure() -> None:
    """Generate XL prefix completion figure: 3 rows × 3 columns.

    Columns: [rendered prefix | model completion text | rendered result]
    Rows: face_partial, open_path, single_shape_group
    """
    prefix_dir = ROOT / 'results' / 'prefixes_v2'
    samples_dir = ROOT / 'results' / 'samples_v2' / 'prefix_conditioned'

    prefix_types = ['face_partial', 'open_path', 'single_shape_group']
    display_names = ['Face (partial)', 'Open path', 'Shape group']
    col_labels = ['Prefix', 'Model completion', 'Rendered result']

    padding = 4
    row_label_h = 20
    col_label_h = 18
    n_cols = 3
    n_rows = len(prefix_types)

    grid_w = n_cols * CELL_SIZE + (n_cols + 1) * padding
    grid_h = col_label_h + n_rows * (CELL_SIZE + row_label_h + padding) + padding

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    label_font = _get_label_font(11)

    # Column headers
    for c, clabel in enumerate(col_labels):
        x = padding + c * (CELL_SIZE + padding) + CELL_SIZE // 2
        draw.text((x, 2), clabel, fill=(80, 80, 80), font=label_font, anchor='mt')

    for row_idx, (ptype, dname) in enumerate(zip(prefix_types, display_names)):
        y_base = col_label_h + row_idx * (CELL_SIZE + row_label_h + padding)

        # Row label
        draw.text((padding + 4, y_base), dname, fill='black', font=label_font)
        y_cells = y_base + row_label_h

        # --- Col 1: Rendered prefix ---
        prefix_path = prefix_dir / f'{ptype}.svg'
        if prefix_path.exists():
            prefix_text = prefix_path.read_text()
            # Prefix files are partial SVGs (unclosed tags); use lxml recovery
            # to produce a well-formed document for rendering the prefix only
            prefix_img = _render_partial_svg(prefix_text) or blank(label='no render')
        else:
            prefix_text = ''
            prefix_img = blank(label='no prefix')

        x1 = padding
        grid.paste(prefix_img, (x1, y_cells))

        # --- Find first sample file ---
        sample_dir = samples_dir / ptype
        svg_files = sorted(sample_dir.glob('sample_*.svg'))
        sample_text = None
        for svg_file in svg_files:
            sample_text = svg_file.read_text()
            print(f'  Prefix {ptype}: using {svg_file.name}')
            break

        if sample_text is None:
            print(f'  Prefix {ptype}: no samples found')
            # Col 2 and 3: placeholders
            x2 = padding + 1 * (CELL_SIZE + padding)
            x3 = padding + 2 * (CELL_SIZE + padding)
            grid.paste(blank(label='no sample'), (x2, y_cells))
            grid.paste(blank(label='no sample'), (x3, y_cells))
            continue

        # --- Col 2: Model completion text ---
        if prefix_text:
            completion_part = sample_text[len(prefix_text):][:200]
        else:
            completion_part = sample_text[:200]

        text_img = _render_text_cell(completion_part)
        x2 = padding + 1 * (CELL_SIZE + padding)
        grid.paste(text_img, (x2, y_cells))

        # --- Col 3: Rendered result (no lxml recovery) ---
        # Fix duplicate attributes before rendering
        cleaned = re.sub(
            r'(\s)(stroke-opacity|fill-opacity|opacity)="[^"]*"\s+\2="([^"]*)"',
            r'\1\2="\3"',
            sample_text,
        )
        rendered = render_svg_strict(cleaned)
        if rendered is not None:
            result_img = rendered
        else:
            result_img = render_failed_placeholder()
            print(f'    → render failed (no recovery)')

        x3 = padding + 2 * (CELL_SIZE + padding)
        grid.paste(result_img, (x3, y_cells))

    output = ROOT / 'report' / 'figures' / 'xl_prefix_samples.png'
    grid.save(str(output), dpi=(DPI, DPI))
    print(f'Saved: {output}')


if __name__ == '__main__':
    print('Generating XL unconditional samples figure...')
    generate_unconditional_figure()
    print('\nGenerating XL prefix completion figure...')
    generate_prefix_figure()
