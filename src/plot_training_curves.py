"""
Generate training_curves.png for report Section 4.6 / Figure 8.

Reads training logs from results/runs/{sp,mup}/{size}/training_log.json
and produces a 5x2 subplot grid (5 rows = sizes, 2 cols = SP/µP).

Training log entries come in two types:
  - Training steps: contain 'train_loss', 'lr', 'time', etc.
  - Evaluation steps: contain 'eval_train_loss', 'eval_val_loss', 'eval_val_ppl'
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SIZES = ["tiny", "small", "medium", "large", "xl"]
PARAMS = ["sp", "mup"]
RESULTS_DIR = Path("results/runs")
OUTPUT_PNG = Path("results/plots/training_curves.png")
OUTPUT_JSON = Path("results/plots/training_curves_data.json")


def load_run(param: str, size: str) -> list[dict]:
    """Load training_log.json for a given (param, size) run."""
    path = RESULTS_DIR / param / size / "training_log.json"
    with open(path) as f:
        return json.load(f)


def extract_curves(
    log: list[dict],
) -> tuple[list[int], list[float], list[int], list[float]]:
    """
    Extract (steps_train, losses_train, steps_val, losses_val) from log.

    Log entries are mixed: training entries have 'train_loss',
    evaluation entries have 'eval_val_loss'.
    """
    steps_train: list[int] = []
    losses_train: list[float] = []
    steps_val: list[int] = []
    losses_val: list[float] = []

    for entry in log:
        step = entry["step"]
        if "train_loss" in entry:
            steps_train.append(step)
            losses_train.append(entry["train_loss"])
        if "eval_val_loss" in entry:
            steps_val.append(step)
            losses_val.append(entry["eval_val_loss"])

    return steps_train, losses_train, steps_val, losses_val


def format_param_label(param: str) -> str:
    """Format parameterization label: 'sp' -> 'SP', 'mup' -> 'µP'."""
    if param == "mup":
        return "µP"
    return param.upper()


def main() -> None:
    # 2 rows (SP top, µP bottom) × 5 cols (model sizes left-to-right)
    fig, axes = plt.subplots(2, 5, figsize=(9.0, 2.8), dpi=300)

    all_data: dict[str, dict] = {}
    curves: dict[tuple[str, str], tuple] = {}

    for param in PARAMS:
        for size in SIZES:
            log = load_run(param, size)
            curve = extract_curves(log)
            curves[(param, size)] = curve

    # Y-axis limits per parameterization (row)
    ylims: dict[str, tuple[float, float]] = {
        "sp": (0.0, 2.5),   # SP Large ~1.3, XL ~1.7; Tiny-Med converge <0.5
        "mup": (0.3, 1.5),  # all µP runs converge 0.38-0.47; show from early training
    }

    # rows=params (SP/µP), cols=sizes (Tiny→XL)
    for row, param in enumerate(PARAMS):
        for col, size in enumerate(SIZES):
            ax = axes[row, col]
            steps_t, losses_t, steps_v, losses_v = curves[(param, size)]

            ax.plot(steps_t, losses_t, "-", color="C0", linewidth=0.8, label="train")
            ax.plot(steps_v, losses_v, "--", color="C1", linewidth=0.8, label="val")

            ax.set_title(
                f"{format_param_label(param)} {size.upper()}",
                fontsize=7, pad=2,
            )
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=4)
            ax.set_ylim(*ylims[param])

            # Left column: y label and legend
            if col == 0:
                ax.set_ylabel(f"{format_param_label(param)} Loss", fontsize=6)
                ax.legend(fontsize=4, loc="upper right")
            else:
                ax.set_yticklabels([])

            # Bottom row: x label
            if row == len(PARAMS) - 1:
                ax.set_xlabel("Step", fontsize=6)

            all_data[f"{param}_{size}"] = {
                "step_train": steps_t,
                "loss_train": [round(v, 6) for v in losses_t],
                "step_val": steps_v,
                "loss_val": [round(v, 6) for v in losses_v],
            }

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
