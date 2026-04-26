"""Evaluation script for SVG language model.

Computes quantitative metrics on generated SVG samples and test set perplexity.

Metrics:
  - Test perplexity (cross-entropy loss on test.bin)
  - XML validity rate (lxml parsing)
  - SVG render rate (CairoSVG)
  - Structural validity (correct <svg> root, closed tags, valid attributes)

Usage:
    python -m src.evaluate \
        --config configs/xl.yaml \
        --checkpoint results/runs/mup_scaling_study/xl/best_model.pt \
        --samples-dir results/samples/xl_mup/ \
        --test-data data/tokenized/test.bin \
        --output-dir results/evaluation/ \
        --mup --mup-base-width 128
"""

import argparse
import io
import json
import math
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
    eval_steps: int,
    device: torch.device,
) -> dict:
    """Compute perplexity on test data."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            max_start = len(test_data) - block_size - 1
            ix = torch.randint(max_start, (batch_size,))
            x = torch.stack([torch.from_numpy(
                test_data[i:i + block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(
                test_data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / eval_steps
    return {
        'test_loss': avg_loss,
        'test_perplexity': math.exp(avg_loss),
        'eval_steps': eval_steps,
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


def check_structural_validity(svg_text: str) -> dict:
    """Check structural SVG validity."""
    checks = {
        'has_svg_root': False,
        'properly_closed': False,
        'valid_viewbox': False,
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
            parts = viewbox.strip().split()
            checks['valid_viewbox'] = len(parts) == 4
    except (etree.XMLSyntaxError, ValueError):
        pass

    return checks


def evaluate_samples(samples_dir: Path) -> dict:
    """Evaluate all SVG files in a directory."""
    svg_files = sorted(samples_dir.glob('*.svg'))
    if not svg_files:
        print(f"  [WARN] No .svg files found in {samples_dir}")
        return {}

    results = {
        'total_samples': len(svg_files),
        'xml_valid': 0,
        'render_success': 0,
        'structural_valid': 0,
        'has_svg_root': 0,
        'valid_viewbox': 0,
        'per_sample': [],
    }

    for svg_path in svg_files:
        svg_text = svg_path.read_text(encoding='utf-8')
        sample = {
            'file': svg_path.name,
            'length': len(svg_text),
        }

        sample['xml_valid'] = check_xml_validity(svg_text)
        if sample['xml_valid']:
            results['xml_valid'] += 1

        sample['renders'] = check_svg_render(svg_text)
        if sample['renders']:
            results['render_success'] += 1

        structural = check_structural_validity(svg_text)
        sample.update(structural)
        if structural['has_svg_root'] and structural['properly_closed']:
            results['structural_valid'] += 1
        if structural['has_svg_root']:
            results['has_svg_root'] += 1
        if structural['valid_viewbox']:
            results['valid_viewbox'] += 1

        results['per_sample'].append(sample)

    n = results['total_samples']
    results['xml_validity_rate'] = results['xml_valid'] / n
    results['render_rate'] = results['render_success'] / n
    results['structural_validity_rate'] = results['structural_valid'] / n

    return results


def render_sample_grid(
    samples_dir: Path,
    output_path: Path,
    grid_cols: int = 5,
    cell_size: int = 128,
) -> None:
    """Render SVG samples into a grid image."""
    svg_files = sorted(samples_dir.glob('*.svg'))
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
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of batches for perplexity estimation')
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
            model, test_data, args.batch_size, block_size,
            args.eval_steps, device,
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
        n = sample_metrics['total_samples']
        print(f"  Total samples: {n}")
        print(f"  XML valid: {sample_metrics['xml_valid']}/{n} "
              f"({sample_metrics['xml_validity_rate']:.1%})")
        print(f"  Renders OK: {sample_metrics['render_success']}/{n} "
              f"({sample_metrics['render_rate']:.1%})")
        print(f"  Structural valid: {sample_metrics['structural_valid']}/{n} "
              f"({sample_metrics['structural_validity_rate']:.1%})")

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
