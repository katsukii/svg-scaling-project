"""Evaluation script for SVG language model.

Computes quantitative metrics on generated SVG samples and test set perplexity.

Metrics:
  - Test perplexity (cross-entropy loss on test.bin)
  - XML validity rate (lxml parsing)
  - SVG render rate (CairoSVG)
  - Structural validity (correct <svg> root, closed tags, valid attributes)

Usage:
    python src/evaluate.py \
        --config configs/xl.yaml \
        --checkpoint results/runs/mup_xl/best_model.pt \
        --samples-dir results/samples/ \
        --test-data data/tokenized/test.bin \
        --output-dir results/evaluation/ \
        --mup
"""

import argparse
import io
import json
import math
import re
from pathlib import Path

import cairosvg
import numpy as np
import torch
from lxml import etree
from PIL import Image

from model import ModelConfig, GPT


def compute_test_perplexity(
    model: GPT,
    test_data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict:
    """Compute perplexity over the full test set using non-overlapping windows.

    Iterates over all non-overlapping windows in test_data so the result
    is deterministic and covers the entire test set, not a random sample.
    """
    model.eval()
    window_size = block_size + 1
    n_windows = len(test_data) // window_size

    if n_windows == 0:
        return {'test_loss': float('nan'), 'test_perplexity': float('nan'),
                'n_windows': 0}

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            actual_bs = batch_end - batch_start

            x = torch.stack([
                torch.from_numpy(
                    test_data[i * window_size:i * window_size + block_size]
                    .astype(np.int64))
                for i in range(batch_start, batch_end)
            ])
            y = torch.stack([
                torch.from_numpy(
                    test_data[i * window_size + 1:i * window_size + 1 + block_size]
                    .astype(np.int64))
                for i in range(batch_start, batch_end)
            ])
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            batch_tokens = actual_bs * block_size
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    return {
        'test_loss': avg_loss,
        'test_perplexity': math.exp(avg_loss),
        'n_windows': n_windows,
        'total_tokens': total_tokens,
    }


def check_xml_validity(svg_text: str) -> bool:
    """Check if SVG parses as valid XML."""
    try:
        etree.fromstring(svg_text.encode('utf-8'))
        return True
    except (etree.XMLSyntaxError, ValueError):
        return False


def check_svg_render(svg_text: str) -> bool:
    """Check if SVG renders successfully via CairoSVG."""
    try:
        cairosvg.svg2png(bytestring=svg_text.encode('utf-8'),
                         output_width=128, output_height=128)
        return True
    except Exception:
        return False


# Regex for CSS numeric values with optional units: "10", "1.5px", "50%", "2em"
_CSS_NUMERIC_RE = re.compile(
    r'^[+-]?(\d+\.?\d*|\.\d+)\s*(%|px|pt|pc|em|ex|rem|in|cm|mm|vw|vh)?$'
)

# CSS keyword values that are valid for numeric-like attributes
_CSS_KEYWORDS = {'none', 'inherit', 'initial', 'auto', 'unset'}


def check_structural_validity(svg_text: str) -> dict:
    """Check structural SVG validity.

    Checks:
      - <svg> root element
      - Properly closed tags (implied by lxml parse success)
      - Valid viewBox format (4 numeric values)
      - Valid numeric attribute values (no NaN, no garbled text in
        attributes that should be numeric like x, y, width, height, r, etc.)
    """
    checks = {
        'has_svg_root': False,
        'properly_closed': False,
        'valid_viewbox': False,
        'valid_attributes': False,
    }

    try:
        root = etree.fromstring(svg_text.encode('utf-8'))
        tag = root.tag
        if '}' in tag:
            tag = tag.split('}')[1]
        checks['has_svg_root'] = tag == 'svg'
        checks['properly_closed'] = True  # lxml parse success means closed tags

        # Check for viewBox attribute
        viewbox = root.get('viewBox') or root.get('viewbox')
        if viewbox:
            parts = viewbox.strip().replace(',', ' ').split()
            if len(parts) == 4:
                try:
                    [float(p) for p in parts]
                    checks['valid_viewbox'] = True
                except ValueError:
                    pass

        # Check numeric attributes across all elements
        numeric_attrs = {
            'x', 'y', 'cx', 'cy', 'r', 'rx', 'ry',
            'width', 'height', 'x1', 'y1', 'x2', 'y2',
            'dx', 'dy', 'fx', 'fy', 'fr',
            'stroke-width', 'stroke-dashoffset', 'opacity',
            'fill-opacity', 'stroke-opacity', 'font-size',
        }
        attr_ok = True
        for elem in root.iter():
            for attr_name, attr_val in elem.attrib.items():
                # Strip namespace prefix
                local_name = attr_name.split('}')[1] if '}' in attr_name else attr_name
                if local_name in numeric_attrs:
                    val = attr_val.strip()
                    if not val or val in _CSS_KEYWORDS:
                        continue  # valid CSS keyword values
                    if not _CSS_NUMERIC_RE.match(val):
                        attr_ok = False
                        break
            if not attr_ok:
                break
        checks['valid_attributes'] = attr_ok

    except (etree.XMLSyntaxError, ValueError):
        pass

    return checks


def _evaluate_single(file_path: Path, samples_dir: Path, is_complete: bool) -> dict:
    """Evaluate a single sample file and return per-sample metrics."""
    rel_path = str(file_path.relative_to(samples_dir))
    if not is_complete:
        return {
            'file': file_path.name,
            'relative_path': rel_path,
            'length': file_path.stat().st_size,
            'complete': False,
            'xml_valid': False,
            'renders': False,
            'has_svg_root': False,
            'properly_closed': False,
            'valid_viewbox': False,
            'valid_attributes': False,
        }

    svg_text = file_path.read_text(encoding='utf-8')
    sample: dict = {
        'file': file_path.name,
        'relative_path': rel_path,
        'length': len(svg_text),
        'complete': True,
    }
    sample['xml_valid'] = check_xml_validity(svg_text)
    sample['renders'] = check_svg_render(svg_text)
    structural = check_structural_validity(svg_text)
    sample.update(structural)
    return sample


def _aggregate(per_sample: list[dict]) -> dict:
    """Compute aggregate metrics from a list of per-sample dicts.

    Output schema matches v1 eval_metrics.json['samples'] so that
    v1/v2 results can be compared mechanically.
    """
    total = len(per_sample)
    if total == 0:
        return {}
    complete = sum(1 for s in per_sample if s['complete'])
    xml_valid = sum(1 for s in per_sample if s['xml_valid'])
    render_ok = sum(1 for s in per_sample if s['renders'])
    has_svg_root = sum(1 for s in per_sample if s.get('has_svg_root'))
    valid_viewbox = sum(1 for s in per_sample if s.get('valid_viewbox'))
    valid_attributes = sum(1 for s in per_sample if s.get('valid_attributes'))
    structural = sum(1 for s in per_sample
                     if s.get('has_svg_root') and s.get('properly_closed')
                     and s.get('valid_attributes'))
    return {
        'total_samples': total,
        'complete_samples': complete,
        'incomplete_samples': total - complete,
        'xml_valid': xml_valid,
        'render_success': render_ok,
        'structural_valid': structural,
        'has_svg_root': has_svg_root,
        'valid_viewbox': valid_viewbox,
        'valid_attributes': valid_attributes,
        'completion_rate': complete / total,
        'xml_validity_rate': xml_valid / total,
        'render_rate': render_ok / total,
        'structural_validity_rate': structural / total,
    }


def _categorize(rel_path: str) -> tuple[str, str]:
    """Return (level1_category, level2_category) from relative path.

    E.g. 'unconditional/topp_t0.8/sample_0.svg' -> ('unconditional', 'topp_t0.8')
         'prefix_conditioned/face_partial/sample_0.svg' -> ('prefix_conditioned', 'face_partial')
    """
    parts = rel_path.replace('\\', '/').split('/')
    level1 = parts[0] if len(parts) > 1 else 'root'
    level2 = parts[1] if len(parts) > 2 else 'root'
    return level1, level2


def evaluate_samples(samples_dir: Path) -> dict:
    """Evaluate all generated outputs in a directory (recursively).

    Searches samples_dir and all subdirectories for complete SVGs (*.svg)
    and incomplete outputs (*_incomplete.txt) so that the denominator
    reflects the total number of generation attempts, not just successes.

    Returns a dict with:
      - aggregate_full: metrics over all samples
      - aggregate_old_subset: metrics over unconditional/topp_* only
        (closest v1 comparator — same sampling method and temperatures,
         but excludes v1 prefix samples since prefix content changed)
      - categories: per-category breakdowns (level1 and level2)
      - per_sample: list of per-sample metrics with relative_path
    """
    svg_files = sorted(samples_dir.rglob('*.svg'))
    incomplete_files = sorted(samples_dir.rglob('*_incomplete.txt'))
    total = len(svg_files) + len(incomplete_files)

    if total == 0:
        print(f"  [WARN] No samples found in {samples_dir}")
        return {}

    # Evaluate all samples
    per_sample = []
    for svg_path in svg_files:
        per_sample.append(_evaluate_single(svg_path, samples_dir, is_complete=True))
    for inc_path in incomplete_files:
        per_sample.append(_evaluate_single(inc_path, samples_dir, is_complete=False))

    # Sort by relative_path for deterministic output
    per_sample.sort(key=lambda s: s['relative_path'])

    # --- 3-tier aggregation ---

    # 1) aggregate_full: all samples
    aggregate_full = _aggregate(per_sample)

    # 2) aggregate_old_subset: unconditional/topp_* only (30 samples)
    #    Closest v1 comparator: same method (top-p 0.95) and temperatures.
    #    Excludes v1 prefix samples (prefix content changed between v1/v2).
    old_subset = [s for s in per_sample
                  if s['relative_path'].startswith('unconditional/topp_')]
    aggregate_old_subset = _aggregate(old_subset)

    # 3) categories: per-directory breakdown
    from collections import defaultdict
    by_level1: dict[str, list[dict]] = defaultdict(list)
    by_level2: dict[str, list[dict]] = defaultdict(list)
    for s in per_sample:
        l1, l2 = _categorize(s['relative_path'])
        by_level1[l1].append(s)
        by_level2[f"{l1}/{l2}"].append(s)

    categories = {}
    for key, samples in sorted(by_level1.items()):
        categories[key] = _aggregate(samples)
    for key, samples in sorted(by_level2.items()):
        categories[key] = _aggregate(samples)

    return {
        'aggregate_full': aggregate_full,
        'aggregate_old_subset': aggregate_old_subset,
        'categories': categories,
        'per_sample': per_sample,
    }


def render_sample_grid(
    samples_dir: Path,
    output_path: Path,
    grid_cols: int = 5,
    cell_size: int = 128,
) -> None:
    """Render SVG samples into a grid image (searches recursively)."""
    svg_files = sorted(samples_dir.rglob('*.svg'))
    if not svg_files:
        return

    images = []
    for svg_path in svg_files:
        svg_text = svg_path.read_text(encoding='utf-8')
        try:
            png_data = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=cell_size, output_height=cell_size,
            )
            img = Image.open(io.BytesIO(png_data)).convert('RGBA')
            # Add white background
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            images.append(bg.convert('RGB'))
        except Exception:
            # Create a red placeholder for failed renders
            img = Image.new('RGB', (cell_size, cell_size), (255, 200, 200))
            images.append(img)

    if not images:
        return

    grid_rows = (len(images) + grid_cols - 1) // grid_cols
    grid_w = grid_cols * cell_size
    grid_h = grid_rows * cell_size
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for i, img in enumerate(images):
        r, c = divmod(i, grid_cols)
        grid.paste(img, (c * cell_size, r * cell_size))

    grid.save(str(output_path))
    print(f"  Saved grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SVG language model')
    parser.add_argument('--config', type=str, help='Model config YAML')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--samples-dir', type=str, required=True,
                        help='Directory containing generated .svg files')
    parser.add_argument('--test-data', type=str, default='data/tokenized/test.bin',
                        help='Path to test.bin for perplexity computation')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Output directory for metrics and grid')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--mup', action='store_true')
    parser.add_argument('--mup-base-width', type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    all_metrics = {}

    # 1. Test perplexity (if model provided)
    if args.config and args.checkpoint:
        print("Computing test perplexity...")
        from generate import load_model
        model = load_model(args.config, args.checkpoint, device,
                           mup=args.mup, mup_base_width=args.mup_base_width)
        test_data = np.fromfile(args.test_data, dtype=np.uint16)
        block_size = model.config.block_size
        ppl_metrics = compute_test_perplexity(
            model, test_data, args.batch_size, block_size, device,
        )
        all_metrics['perplexity'] = ppl_metrics
        print(f"  Test loss: {ppl_metrics['test_loss']:.4f}")
        print(f"  Test perplexity: {ppl_metrics['test_perplexity']:.2f}")
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # 2. Sample evaluation
    print(f"\nEvaluating samples from {samples_dir}...")
    sample_metrics = evaluate_samples(samples_dir)
    if sample_metrics:
        all_metrics['samples'] = sample_metrics

        def _print_agg(label: str, agg: dict) -> None:
            if not agg:
                return
            n = agg['total_samples']
            nc = agg['complete_samples']
            print(f"\n  [{label}] ({n} samples)")
            print(f"    Completion:  {nc}/{n} ({agg['completion_rate']:.1%})")
            print(f"    XML valid:   {agg['xml_valid']}/{n} ({agg['xml_validity_rate']:.1%})")
            print(f"    Renders OK:  {agg['render_success']}/{n} ({agg['render_rate']:.1%})")
            print(f"    Structural:  {agg['structural_valid']}/{n} ({agg['structural_validity_rate']:.1%})")

        _print_agg('ALL', sample_metrics['aggregate_full'])
        _print_agg('OLD SUBSET (uncond top-p)', sample_metrics['aggregate_old_subset'])
        for cat_key, cat_agg in sorted(sample_metrics.get('categories', {}).items()):
            _print_agg(cat_key, cat_agg)

    # 3. Render grid
    print("\nRendering sample grid...")
    render_sample_grid(samples_dir, output_dir / 'sample_grid.png')

    # 4. Save metrics
    metrics_path = output_dir / 'eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
