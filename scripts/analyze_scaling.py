"""Analyze scaling study results: summary table + power law fit + plot.

Usage:
    python scripts/analyze_scaling.py
    python scripts/analyze_scaling.py --scaling-dir results/runs/scaling_study
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_scaling_results(scaling_dir: Path) -> list[dict]:
    """Load summary.json from each completed run."""
    results = []
    for run_dir in sorted(scaling_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"  [WARN] {run_dir.name}: no summary.json (skipped)")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        summary["run_name"] = run_dir.name
        results.append(summary)
    return results


def print_summary_table(results: list[dict]) -> None:
    """Print a comparison table of scaling results."""
    print("\n=== Scaling Study Results ===\n")
    print(f"{'Model':>8s}  {'Params':>10s}  {'Final Val Loss':>14s}  "
          f"{'Best Val Loss':>13s}  {'Final PPL':>10s}  {'Time':>10s}")
    print("-" * 75)

    for r in sorted(results, key=lambda x: x["n_params"]):
        name = r["run_name"]
        params_str = f"{r['n_params']/1e6:.1f}M"
        time_str = f"{r['total_time_s']/60:.1f}m"
        print(f"{name:>8s}  {params_str:>10s}  {r['final_val_loss']:>14.4f}  "
              f"{r['best_val_loss']:>13.4f}  "
              f"{r['final_val_ppl']:>10.2f}  {time_str:>10s}")


def power_law_3p(n: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """L = a * N^(-alpha) + c"""
    return a * np.power(n, -alpha) + c


def power_law_2p(n: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """L = a * N^(-alpha)"""
    return a * np.power(n, -alpha)


def fit_power_law(n_params: np.ndarray, losses: np.ndarray) -> dict:
    """Fit power law L = a * N^(-alpha) + c, with 2-param fallback."""
    # Try 3-parameter fit
    try:
        popt, pcov = curve_fit(
            power_law_3p, n_params, losses,
            p0=[10.0, 0.1, 1.0],
            bounds=([0, 0, 0], [np.inf, 2.0, np.inf]),
            maxfev=10000,
        )
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))
        residuals = losses - power_law_3p(n_params, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "model": "3-param",
            "a": a, "alpha": alpha, "c": c,
            "a_err": perr[0], "alpha_err": perr[1], "c_err": perr[2],
            "r_squared": r_squared,
            "predict_fn": lambda n: power_law_3p(n, a, alpha, c),
        }
    except RuntimeError:
        pass

    # Fallback: 2-parameter fit (c=0)
    print("  [WARN] 3-param fit failed, using 2-param fallback (c=0)")
    try:
        popt, pcov = curve_fit(
            power_law_2p, n_params, losses,
            p0=[10.0, 0.1],
            bounds=([0, 0], [np.inf, 2.0]),
            maxfev=10000,
        )
        a, alpha = popt
        perr = np.sqrt(np.diag(pcov))
        residuals = losses - power_law_2p(n_params, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "model": "2-param",
            "a": a, "alpha": alpha, "c": 0.0,
            "a_err": perr[0], "alpha_err": perr[1], "c_err": 0.0,
            "r_squared": r_squared,
            "predict_fn": lambda n: power_law_2p(n, a, alpha),
        }
    except RuntimeError:
        print("  [ERROR] Both fits failed")
        return None


def plot_scaling_law(
    results: list[dict],
    fit_result: dict | None,
    output_path: Path,
) -> None:
    """Create log-log scaling plot with data points + fit curve."""
    results_sorted = sorted(results, key=lambda x: x["n_params"])
    n_params = np.array([r["n_params"] for r in results_sorted])
    losses = np.array([r["final_val_loss"] for r in results_sorted])
    names = [r["run_name"] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Data points
    ax.scatter(n_params, losses, s=80, zorder=5, color="C0", edgecolors="black", linewidth=0.5)
    for n, l, name in zip(n_params, losses, names):
        ax.annotate(name, (n, l), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    # Fit curve
    if fit_result is not None:
        n_range = np.logspace(
            np.log10(n_params.min() * 0.5),
            np.log10(n_params.max() * 2),
            200,
        )
        l_pred = fit_result["predict_fn"](n_range)
        model_str = fit_result["model"]
        alpha = fit_result["alpha"]
        c = fit_result["c"]
        r2 = fit_result["r_squared"]

        label = f"Fit: L = {fit_result['a']:.2f}·N^(-{alpha:.4f})"
        if model_str == "3-param":
            label += f" + {c:.4f}"
        label += f"\nα={alpha:.4f} (Kaplan NLP: 0.076), R²={r2:.4f}"

        ax.plot(n_range, l_pred, "--", color="C1", linewidth=2, label=label)
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation Loss (after 1 epoch)")
    ax.set_title("Scaling Law: SVG Language Model")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze scaling study results")
    parser.add_argument("--scaling-dir", type=str, default="results/runs/scaling_study")
    args = parser.parse_args()

    scaling_dir = Path(args.scaling_dir)
    if not scaling_dir.exists():
        print(f"Error: {scaling_dir} does not exist")
        return

    results = load_scaling_results(scaling_dir)
    if not results:
        print("No completed runs found.")
        return

    print_summary_table(results)

    # Power law fit (using final val loss = "after 1 epoch")
    n_params = np.array([r["n_params"] for r in results])
    losses = np.array([r["final_val_loss"] for r in results])

    fit_result = None
    if len(results) >= 2:
        print("\n--- Power Law Fit ---")
        fit_result = fit_power_law(n_params, losses)
        if fit_result:
            print(f"  Model: {fit_result['model']}")
            print(f"  a = {fit_result['a']:.4f} ± {fit_result['a_err']:.4f}")
            print(f"  α = {fit_result['alpha']:.4f} ± {fit_result['alpha_err']:.4f}")
            if fit_result['model'] == '3-param':
                print(f"  c = {fit_result['c']:.4f} ± {fit_result['c_err']:.4f}")
            print(f"  R² = {fit_result['r_squared']:.4f}")
            print(f"  Kaplan NLP α_N ≈ 0.076 → SVG α = {fit_result['alpha']:.4f}")
    else:
        print(f"\n  [WARN] Only {len(results)} data point(s), need ≥2 for fit")

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling_law(results, fit_result, plots_dir / "scaling_law.png")

    # Save fit results as JSON for later use
    if fit_result:
        fit_export = {k: v for k, v in fit_result.items() if k != "predict_fn"}
        with open(scaling_dir / "fit_results.json", "w") as f:
            json.dump(fit_export, f, indent=2)
        print(f"Saved: {scaling_dir / 'fit_results.json'}")


if __name__ == "__main__":
    main()
