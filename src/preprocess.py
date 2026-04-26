"""SVG data preprocessing pipeline.

Loads raw SVG data from HuggingFace dataset, cleans it, and saves
processed data for tokenization.

Cleaning steps:
  1. Remove HTML/XML comments
  2. Strip metadata elements (<metadata>, <title>, <desc>)
  3. Normalize coordinate precision (round to 1 decimal place)
  4. Compress unnecessary whitespace
  5. Validate as well-formed XML
  6. Validate rendering via CairoSVG
  7. Filter by character length (min/max)

Reference: nanoGPT data preparation approach, adapted for SVG domain.
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import cairosvg
from lxml import etree
from datasets import load_from_disk


# SVG namespace
SVG_NS = 'http://www.w3.org/2000/svg'
NSMAP = {'svg': SVG_NS}

# Elements to strip (metadata / non-visual)
STRIP_TAGS = [
    f'{{{SVG_NS}}}metadata',
    f'{{{SVG_NS}}}title',
    f'{{{SVG_NS}}}desc',
    'metadata', 'title', 'desc',  # without namespace
]

# Numeric attributes whose values should be rounded
NUMERIC_ATTRS = {
    'x', 'y', 'x1', 'y1', 'x2', 'y2',
    'cx', 'cy', 'r', 'rx', 'ry',
    'width', 'height',
    'dx', 'dy',
    'stroke-width', 'stroke-dashoffset',
    'font-size', 'letter-spacing',
    'opacity', 'fill-opacity', 'stroke-opacity',
}

# Regex for floating-point numbers with decimals in path/transform/viewBox data
_FLOAT_RE = re.compile(r'(-?\d+\.\d{2,})')


def _round_float_match(m: re.Match) -> str:
    """Round a matched float to 1 decimal place."""
    return f'{float(m.group()):.1f}'


def _round_floats_in_string(s: str) -> str:
    """Round all floats with 2+ decimal places to 1 decimal in a string."""
    return _FLOAT_RE.sub(_round_float_match, s)


def _round_single_value(val: str) -> str:
    """Round a single numeric attribute value to 1 decimal."""
    try:
        f = float(val)
        if '.' in val:
            return f'{f:.1f}'
        return val  # integer value, leave as-is
    except ValueError:
        return val


def normalize_coordinates(root: etree._Element) -> None:
    """Round coordinate precision to 1 decimal place in-place.

    Handles:
    - Numeric attributes (x, y, width, height, etc.)
    - viewBox attribute
    - d attribute (path data)
    - transform attribute
    - points attribute (polygon/polyline)
    """
    for elem in root.iter():
        for attr_name, attr_val in list(elem.attrib.items()):
            local_name = attr_name.split('}')[-1] if '}' in attr_name else attr_name

            if local_name in NUMERIC_ATTRS:
                elem.attrib[attr_name] = _round_single_value(attr_val)
            elif local_name in ('d', 'transform', 'viewBox', 'points'):
                elem.attrib[attr_name] = _round_floats_in_string(attr_val)


def strip_metadata_elements(root: etree._Element) -> None:
    """Remove metadata, title, desc elements from SVG."""
    for tag in STRIP_TAGS:
        for elem in root.findall(f'.//{tag}'):
            elem.getparent().remove(elem)


def validate_render(svg_bytes: bytes) -> bool:
    """Check that the SVG renders without errors via CairoSVG."""
    try:
        cairosvg.svg2png(bytestring=svg_bytes, output_width=64, output_height=64)
        return True
    except Exception:
        return False


def clean_svg(svg_text: str, normalize_coords: bool = True,
              check_render: bool = True) -> tuple[str | None, str | None]:
    """Clean a single SVG string.

    Returns:
        (cleaned_svg, rejection_reason) — one of the two is None.
    """
    # 1. Remove comments
    svg_text = re.sub(r'<!--.*?-->', '', svg_text, flags=re.DOTALL)
    svg_text = svg_text.strip()

    if not svg_text:
        return None, 'empty'

    # 2. Parse XML
    try:
        root = etree.fromstring(svg_text.encode('utf-8'))
    except etree.XMLSyntaxError:
        return None, 'invalid_xml'

    # 3. Strip metadata elements
    strip_metadata_elements(root)

    # 4. Normalize coordinates
    if normalize_coords:
        normalize_coordinates(root)

    # 5. Re-serialize — strip_text removes inter-element whitespace
    cleaned_bytes = etree.tostring(root, encoding='unicode')

    # 6. Compress whitespace: collapse runs of whitespace to single space,
    #    but preserve single spaces in attribute values
    cleaned = re.sub(r'\s+', ' ', cleaned_bytes).strip()

    # 7. Validate render
    if check_render:
        if not validate_render(cleaned.encode('utf-8')):
            return None, 'render_failed'

    return cleaned, None


def process_split(dataset, split_name: str, output_dir: Path,
                  min_len: int = 50, max_len: int | None = None,
                  normalize_coords: bool = True,
                  check_render: bool = True) -> dict:
    """Process a single dataset split.

    Args:
        dataset: HuggingFace dataset split
        split_name: 'train', 'val', or 'test'
        output_dir: directory to write output files
        min_len: minimum character length (default: 50, per spec)
        max_len: optional max character length filter
        normalize_coords: whether to round coordinates to 1 decimal
        check_render: whether to validate rendering via CairoSVG

    Returns:
        dict with processing statistics
    """
    stats = {
        'total': len(dataset),
        'valid': 0,
        'invalid_xml': 0,
        'too_short': 0,
        'too_long': 0,
        'render_failed': 0,
        'empty': 0,
    }

    output_path = output_dir / f'{split_name}.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(dataset)):
            svg_text = dataset[i]['Svg']
            cleaned, reason = clean_svg(svg_text, normalize_coords, check_render)

            if reason is not None:
                stats[reason] = stats.get(reason, 0) + 1
                continue

            if len(cleaned) < min_len:
                stats['too_short'] += 1
                continue

            if max_len is not None and len(cleaned) > max_len:
                stats['too_long'] += 1
                continue

            stats['valid'] += 1
            f.write(json.dumps({'svg': cleaned}) + '\n')

    return stats


def print_stats(stats: dict, split_name: str) -> None:
    """Print processing statistics for a split."""
    print(f"\n--- {split_name} ---")
    print(f"  Total:          {stats['total']}")
    print(f"  Valid:          {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"  Invalid XML:    {stats['invalid_xml']}")
    print(f"  Too short:      {stats['too_short']}")
    print(f"  Too long:       {stats['too_long']}")
    print(f"  Render failed:  {stats['render_failed']}")
    print(f"  Empty:          {stats.get('empty', 0)}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SVG dataset')
    parser.add_argument('--input-dir', type=str, default='data/raw/svg-icons-simple',
                        help='Path to HuggingFace dataset on disk')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for cleaned data')
    parser.add_argument('--min-len', type=int, default=50,
                        help='Min character length filter (default: 50)')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Max character length filter (None = no filter)')
    parser.add_argument('--no-coord-norm', action='store_true',
                        help='Skip coordinate normalization')
    parser.add_argument('--no-render-check', action='store_true',
                        help='Skip CairoSVG render validation')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_dir}...")
    ds = load_from_disk(str(input_dir))

    normalize_coords = not args.no_coord_norm
    check_render = not args.no_render_check
    print(f"Settings: min_len={args.min_len}, max_len={args.max_len}, "
          f"coord_norm={normalize_coords}, render_check={check_render}")

    all_stats = {}
    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing {split_name}...")
        stats = process_split(
            ds[split_name], split_name, output_dir,
            min_len=args.min_len,
            max_len=args.max_len,
            normalize_coords=normalize_coords,
            check_render=check_render,
        )
        print_stats(stats, split_name)
        all_stats[split_name] = stats

    # Save summary stats
    stats_path = output_dir / 'preprocess_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == '__main__':
    main()
