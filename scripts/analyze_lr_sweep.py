"""Analyze LR sweep results: summary table + plots.

Usage:
    python scripts/analyze_lr_sweep.py
    python scripts/analyze_lr_sweep.py --sweep-dir results/runs/lr_sweep
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_sweep_results(sweep_dir: Path) -> list[dict]:
    """Load summary.json from each completed run."""
    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"  [WARN] {run_dir.name}: no summary.json (skipped)")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        summary["run_dir"] = str(run_dir)
        summary["run_name"] = run_dir.name
        results.append(summary)
    return results


def load_training_logs(sweep_dir: Path) -> dict[str, list[dict]]:
    """Load training_log.json from each completed run."""
    logs = {}
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log_path = run_dir / "training_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            logs[run_dir.name] = json.load(f)
    return logs


def print_summary_table(results: list[dict]) -> None:
    """Print a comparison table of LR sweep results."""
    print("\n=== LR Sweep Results ===\n")
    print(f"{'LR':>10s}  {'Best Val Loss':>13s}  {'Final Val Loss':>14s}  "
          f"{'Final PPL':>10s}  {'Time (s)':>9s}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        lr = r["config"]["training"]["learning_rate"]
        print(f"{lr:>10.1e}  {r['best_val_loss']:>13.4f}  {r['final_val_loss']:>14.4f}  "
              f"{r['final_val_ppl']:>10.2f}  {r['total_time_s']:>9.1f}")

    best = min(results, key=lambda x: x["best_val_loss"])
    best_lr = best["config"]["training"]["learning_rate"]
    print(f"\n→ Optimal LR: {best_lr:.1e} (best_val_loss={best['best_val_loss']:.4f})")


def plot_lr_comparison(results: list[dict], output_path: Path) -> None:
    """Plot LR vs best val loss."""
    results_sorted = sorted(results, key=lambda x: x["config"]["training"]["learning_rate"])
    lrs = [r["config"]["training"]["learning_rate"] for r in results_sorted]
    losses = [r["best_val_loss"] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, "o-", markersize=8, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("LR Sweep: Tiny Model (1.3M params)")
    ax.grid(True, alpha=0.3)

    # Annotate best
    best_idx = losses.index(min(losses))
    ax.annotate(
        f"Best: {lrs[best_idx]:.1e}\nloss={losses[best_idx]:.4f}",
        xy=(lrs[best_idx], losses[best_idx]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def plot_val_loss_curves(logs: dict[str, list[dict]], output_path: Path) -> None:
    """Plot val loss curves over training steps for each LR."""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for run_name, entries in sorted(logs.items()):
        # Extract eval entries (have eval_val_loss)
        eval_entries = [e for e in entries if "eval_val_loss" in e]
        if not eval_entries:
            continue
        has_data = True

        steps = [e["step"] for e in eval_entries]
        val_losses = [e["eval_val_loss"] for e in eval_entries]
        # Extract LR from run name (tiny_lr_3e-4 → 3e-4)
        lr_str = run_name.replace("tiny_lr_", "")
        ax.plot(steps, val_losses, "o-", label=f"LR={lr_str}", markersize=4)

    if not has_data:
        print(f"  [WARN] No eval data found in training logs, skipping curves plot")
        plt.close(fig)
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("LR Sweep: Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LR sweep results")
    parser.add_argument("--sweep-dir", type=str, default="results/runs/lr_sweep")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: {sweep_dir} does not exist")
        return

    results = load_sweep_results(sweep_dir)
    if not results:
        print("No completed runs found.")
        return

    print_summary_table(results)

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_lr_comparison(results, plots_dir / "lr_sweep_comparison.png")

    logs = load_training_logs(sweep_dir)
    plot_val_loss_curves(logs, plots_dir / "lr_sweep_curves.png")


if __name__ == "__main__":
    main()
