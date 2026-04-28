"""Generate SVG samples from each model size for qualitative comparison.

Produces unconditional and prefix-conditioned samples for Tiny→XL,
renders them with CairoSVG, and creates a summary grid image.

Usage:
    python scripts/generate_size_comparison.py [--device mps]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-unconditional', type=int, default=10)
    parser.add_argument('--num-prefix', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--output-dir', type=str,
                        default='results/samples_size_comparison')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    sizes = ['tiny', 'small', 'medium', 'large', 'xl']
    prefixes = {
        'single_shape_group': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><rect x="3" y="3" width="18" height="18" fill="none" stroke="black"/>',
        'face_partial': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="black"/><circle cx="9" cy="10" r="1.5" fill="black"/>',
        'open_path': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M4 4 L20 4 L20 20',
    }

    device_arg = []
    if args.device:
        device_arg = ['--device', args.device]

    for size in sizes:
        config = f'configs/{size}.yaml'
        ckpt = f'results/runs/mup/{size}/best_model.pt'

        if not Path(ckpt).exists():
            print(f'SKIP {size}: checkpoint not found at {ckpt}')
            continue

        # --- Unconditional ---
        uncond_dir = f'{args.output_dir}/{size}/unconditional'
        print(f'\n{"="*60}')
        print(f'  {size.upper()} — unconditional ({args.num_unconditional} samples)')
        print(f'{"="*60}')

        cmd = [
            sys.executable, 'src/generate.py',
            '--config', config,
            '--checkpoint', ckpt,
            '--tokenizer', 'tokenizer/bpe_svg.json',
            '--num-samples', str(args.num_unconditional),
            '--temperature', str(args.temperature),
            '--top-k', str(args.top_k),
            '--max-tokens', str(args.max_tokens),
            '--output-dir', uncond_dir,
            '--mup',
            *device_arg,
        ]
        subprocess.run(cmd, check=True)

        # --- Prefix-conditioned ---
        for pname, prefix in prefixes.items():
            prefix_dir = f'{args.output_dir}/{size}/prefix/{pname}'
            print(f'\n  {size.upper()} — prefix: {pname} ({args.num_prefix} samples)')

            cmd = [
                sys.executable, 'src/generate.py',
                '--config', config,
                '--checkpoint', ckpt,
                '--tokenizer', 'tokenizer/bpe_svg.json',
                '--num-samples', str(args.num_prefix),
                '--prefix', prefix,
                '--temperature', str(args.temperature),
                '--top-k', str(args.top_k),
                '--max-tokens', str(args.max_tokens),
                '--output-dir', prefix_dir,
                '--mup',
                *device_arg,
            ]
            subprocess.run(cmd, check=True)

    # --- Post-processing ---
    print(f'\n{"="*60}')
    print('  Post-processing: render + stats')
    print(f'{"="*60}')

    render_and_summarize(Path(args.output_dir), sizes)


def render_and_summarize(output_dir: Path, sizes: list[str]):
    """Render SVGs with CairoSVG and compute per-size stats."""
    import xml.etree.ElementTree as ET

    try:
        import cairosvg
        has_cairo = True
    except ImportError:
        print('WARNING: cairosvg not installed, skipping render')
        has_cairo = False

    stats = {}

    for size in sizes:
        size_dir = output_dir / size
        if not size_dir.exists():
            continue

        svg_files = list(size_dir.rglob('*.svg'))
        incomplete_files = list(size_dir.rglob('*_incomplete.txt'))
        total = len(svg_files) + len(incomplete_files)

        xml_valid = 0
        render_ok = 0

        for svg_path in svg_files:
            # XML validity check
            try:
                ET.parse(svg_path)
                xml_valid += 1
            except ET.ParseError:
                continue

            # Render to PNG
            if has_cairo:
                png_path = svg_path.with_suffix('.png')
                try:
                    cairosvg.svg2png(
                        url=str(svg_path),
                        write_to=str(png_path),
                        output_width=200,
                        output_height=200,
                    )
                    render_ok += 1
                except Exception:
                    pass

        stats[size] = {
            'total': total,
            'complete': len(svg_files),
            'incomplete': len(incomplete_files),
            'xml_valid': xml_valid,
            'render_success': render_ok,
            'completion_rate': len(svg_files) / total if total > 0 else 0,
            'xml_validity_rate': xml_valid / total if total > 0 else 0,
            'render_rate': render_ok / total if total > 0 else 0,
        }

        print(f'  {size:>8s}: {len(svg_files)}/{total} complete, '
              f'{xml_valid} xml-valid, {render_ok} rendered')

    # Save stats
    stats_path = output_dir / 'size_comparison_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\n  Stats saved to {stats_path}')

    # Create grid image
    if has_cairo:
        create_grid(output_dir, sizes)


def create_grid(output_dir: Path, sizes: list[str]):
    """Create a grid image: 5 sizes × 5 unconditional samples."""
    try:
        from PIL import Image
    except ImportError:
        print('WARNING: Pillow not installed, skipping grid')
        return

    cell_size = 200
    cols = 5  # samples per row
    rows = len(sizes)
    padding = 2
    label_height = 30

    grid_w = cols * (cell_size + padding) + padding
    grid_h = rows * (cell_size + padding + label_height) + padding
    grid = Image.new('RGB', (grid_w, grid_h), 'white')

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for row, size in enumerate(sizes):
        uncond_dir = output_dir / size / 'unconditional'
        pngs = sorted(uncond_dir.glob('*.png'))[:cols]

        y_offset = row * (cell_size + padding + label_height) + padding

        # Label
        if draw and font:
            summary_path = Path(f'results/runs/mup/{size}/summary.json')
            label = size.upper()
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f)
                label += f'  ({s["n_params"]/1e6:.1f}M, val_loss={s["best_val_loss"]:.3f})'
            draw.text((padding + 4, y_offset + 2), label, fill='black', font=font)

        for col, png_path in enumerate(pngs):
            x = col * (cell_size + padding) + padding
            y = y_offset + label_height
            try:
                img = Image.open(png_path).resize((cell_size, cell_size))
                grid.paste(img, (x, y))
            except Exception:
                pass

    grid_path = output_dir / 'size_comparison_grid.png'
    grid.save(grid_path)
    print(f'  Grid saved to {grid_path}')


if __name__ == '__main__':
    main()
