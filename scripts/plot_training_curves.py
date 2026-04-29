"""Plot training loss curves for all model sizes.

Reads training_log.json from each model's run directory and creates
a multi-line plot showing loss vs. training step.

Usage:
    python scripts/plot_training_curves.py --run-dir results/runs/sp
    python scripts/plot_training_curves.py --run-dir results/runs/mup --label "µP"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_ORDER = ['tiny', 'small', 'medium', 'large', 'xl']
COLORS = {
    'tiny': '#1f77b4',
    'small': '#ff7f0e',
    'medium': '#2ca02c',
    'large': '#d62728',
    'xl': '#9467bd',
}


def load_training_log(log_path: Path) -> tuple[list[int], list[float]]:
    """Load steps and losses from a training_log.json."""
    with open(log_path) as f:
        entries = json.load(f)
    steps = [e['step'] for e in entries]
    losses = [e['train_loss'] for e in entries]
    return steps, losses


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Directory containing model subdirs with training_log.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: results/plots/training_curves_{label}.png)')
    parser.add_argument('--label', type=str, default='SP',
                        help='Label for the plot title (SP or µP)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        safe_label = args.label.lower().replace('µ', 'mu')
        output_path = Path(f'results/plots/training_curves_{safe_label}.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    found = 0

    for model_name in MODEL_ORDER:
        log_path = run_dir / model_name / 'training_log.json'
        if not log_path.exists():
            print(f'  SKIP {model_name}: {log_path} not found')
            continue
        steps, losses = load_training_log(log_path)
        ax.plot(steps, losses, label=model_name.capitalize(),
                color=COLORS[model_name], linewidth=1.5, alpha=0.8)
        print(f'  {model_name}: {len(steps)} points, '
              f'loss {losses[0]:.3f} → {losses[-1]:.3f}')
        found += 1

    if found == 0:
        print(f'No training logs found in {run_dir}/')
        print('Download training_log.json files from Colab first.')
        return

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'Training Curves ({args.label}, {found} models)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f'\nSaved: {output_path}')


if __name__ == '__main__':
    main()
