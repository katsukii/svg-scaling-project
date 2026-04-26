"""Analyze µP results: LR sweep, SP vs µP scaling comparison, LR transfer.

Produces:
  1. µP LR Sweep plot — LR vs final_val_loss
  2. SP vs µP Scaling Law — overlaid power law fits on log-log plot
  3. LR Transfer plot — showing SP LR=3e-3 fails at scale, µP stays stable
  4. Comparison table — SP vs µP per-model final_val_loss
  5. Extrapolation — predict loss for 10× model using µP fit

Usage:
    python scripts/analyze_mup.py
    python scripts/analyze_mup.py --sp-dir results/runs/scaling_study \
                                  --mup-dir results/runs/mup_scaling_study \
                                  --mup-sweep-dir results/runs/mup_lr_sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_results(results_dir: Path) -> list[dict]:
    """Load summary.json from each subdirectory."""
    results = []
    if not results_dir.exists():
        print(f"  [WARN] {results_dir} does not exist")
        return results
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        summary["run_name"] = run_dir.name
        results.append(summary)
    return results


def load_lr_sweep(sweep_dir: Path) -> list[dict]:
    """Load LR sweep results from subdirectories.

    Reads the learning rate from each run's summary.json rather than
    parsing directory names, so it works regardless of naming convention
    (e.g. ``lr_3e-3``, ``tiny_lr_3e-3``, or arbitrary names).
    """
    results = []
    if not sweep_dir.exists():
        print(f"  [WARN] {sweep_dir} does not exist")
        return results
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        # Read LR from the saved config (authoritative source)
        try:
            lr_val = float(summary["config"]["training"]["learning_rate"])
        except (KeyError, TypeError, ValueError):
            print(f"  [WARN] {run_dir.name}: could not read LR from summary.json (skipped)")
            continue
        summary["lr_str"] = f"{lr_val:.0e}"
        summary["lr_val"] = lr_val
        summary["run_name"] = run_dir.name
        results.append(summary)
    return sorted(results, key=lambda x: x["lr_val"])


def power_law_3p(n: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """L = a * N^(-alpha) + c"""
    return a * np.power(n, -alpha) + c


def fit_power_law(n_params: np.ndarray, losses: np.ndarray, label: str = "") -> dict | None:
    """Fit L = a * N^(-alpha) + c."""
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
            "a": a, "alpha": alpha, "c": c,
            "a_err": perr[0], "alpha_err": perr[1], "c_err": perr[2],
            "r_squared": r_squared,
            "label": label,
            "pcov": pcov.tolist(),  # covariance matrix for CI estimation
        }
    except RuntimeError:
        print(f"  [WARN] Power law fit failed for {label}")
        return None


# --- Plot 1: µP LR Sweep ---

def plot_mup_lr_sweep(sweep_results: list[dict], output_path: Path) -> None:
    """Plot LR vs final_val_loss for µP sweep."""
    if not sweep_results:
        print("  [SKIP] No µP LR sweep results")
        return

    lrs = [r["lr_val"] for r in sweep_results]
    losses = [r["final_val_loss"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, 'o-', color="C2", markersize=8, linewidth=2)

    best_idx = np.argmin(losses)
    ax.scatter([lrs[best_idx]], [losses[best_idx]], s=150, color="red",
               zorder=5, label=f"Best: LR={lrs[best_idx]:.0e}, loss={losses[best_idx]:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Loss (after 1 epoch)")
    ax.set_title("µP LR Sweep (Tiny Model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Plot 1b: Combined SP + µP LR Sweep ---

def plot_combined_lr_sweep(
    sp_sweep: list[dict],
    mup_sweep: list[dict],
    output_path: Path,
) -> None:
    """Overlay SP and µP LR sweep results on the same plot."""
    if not sp_sweep and not mup_sweep:
        print("  [SKIP] No LR sweep results for either parameterization")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for sweep, color, marker, label in [
        (sp_sweep, "C0", "o", "SP"),
        (mup_sweep, "C2", "s", "µP"),
    ]:
        if not sweep:
            continue
        lrs = [r["lr_val"] for r in sweep]
        losses = [r["final_val_loss"] for r in sweep]
        ax.plot(lrs, losses, f'{marker}-', color=color, markersize=8, linewidth=2, label=label)

        best_idx = int(np.argmin(losses))
        ax.scatter([lrs[best_idx]], [losses[best_idx]], s=150, color=color,
                   edgecolors="red", linewidth=2, zorder=5,
                   label=f"{label} best: LR={lrs[best_idx]:.0e}")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Loss (after 1 epoch)")
    ax.set_title("LR Sweep Comparison: SP vs µP (Tiny Model)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Plot 2: SP vs µP Scaling Law ---

def plot_sp_vs_mup_scaling(
    sp_results: list[dict],
    mup_results: list[dict],
    sp_fit: dict | None,
    mup_fit: dict | None,
    output_path: Path,
) -> None:
    """Overlay SP and µP scaling laws on the same log-log plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for results, fit, color, marker, label_prefix in [
        (sp_results, sp_fit, "C0", "o", "SP"),
        (mup_results, mup_fit, "C2", "s", "µP"),
    ]:
        if not results:
            continue
        results_sorted = sorted(results, key=lambda x: x["n_params"])
        n = np.array([r["n_params"] for r in results_sorted])
        l = np.array([r["final_val_loss"] for r in results_sorted])
        names = [r["run_name"] for r in results_sorted]

        ax.scatter(n, l, s=80, color=color, edgecolors="black", linewidth=0.5,
                   marker=marker, zorder=5, label=f"{label_prefix} data")

        for ni, li, name in zip(n, l, names):
            ax.annotate(name, (ni, li), textcoords="offset points",
                        xytext=(8, 5), fontsize=7, color=color)

        if fit is not None:
            n_range = np.logspace(
                np.log10(n.min() * 0.3),
                np.log10(n.max() * 15),  # extend for extrapolation
                200,
            )
            l_pred = power_law_3p(n_range, fit["a"], fit["alpha"], fit["c"])
            fit_label = (f"{label_prefix}: L={fit['a']:.2f}·N^(-{fit['alpha']:.4f})+{fit['c']:.2f}"
                         f"  (α={fit['alpha']:.4f}, R²={fit['r_squared']:.4f})")
            ax.plot(n_range, l_pred, "--", color=color, linewidth=2, label=fit_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation Loss (after 1 epoch)")
    ax.set_title("SP vs µP Scaling Laws for SVG Language Model")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Plot 3: LR Transfer ---

def plot_lr_transfer(
    sp_results: list[dict],
    mup_results: list[dict],
    output_path: Path,
) -> None:
    """Show that µP transfers LR across scales while SP fails."""
    if not sp_results or not mup_results:
        print("  [SKIP] Missing SP or µP results for LR transfer plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for results, color, label in [
        (sp_results, "C0", "SP"),
        (mup_results, "C2", "µP"),
    ]:
        results_sorted = sorted(results, key=lambda x: x["n_params"])
        n = [r["n_params"] for r in results_sorted]
        l = [r["final_val_loss"] for r in results_sorted]
        names = [r["run_name"] for r in results_sorted]
        ax.plot(n, l, 'o-', color=color, markersize=8, linewidth=2, label=label)
        for ni, li, name in zip(n, l, names):
            ax.annotate(name, (ni, li), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation Loss (after 1 epoch)")
    ax.set_title("LR Transfer: SP vs µP\n(Same base LR transferred across model sizes)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Comparison Table + Extrapolation ---

def print_comparison(sp_results: list[dict], mup_results: list[dict]) -> None:
    """Print SP vs µP comparison table."""
    # Build lookup by config_name
    sp_lookup = {r["config_name"]: r for r in sp_results}
    mup_lookup = {r["config_name"]: r for r in mup_results}
    all_names = sorted(set(sp_lookup) | set(mup_lookup),
                       key=lambda x: sp_lookup.get(x, mup_lookup.get(x, {})).get("n_params", 0))

    print("\n=== SP vs µP Comparison ===\n")
    print(f"{'Model':>8s}  {'Params':>10s}  {'SP Loss':>10s}  {'µP Loss':>10s}  {'Δ':>10s}")
    print("-" * 55)

    for name in all_names:
        sp = sp_lookup.get(name)
        mup = mup_lookup.get(name)
        params_str = f"{(sp or mup)['n_params']/1e6:.1f}M"
        sp_loss = f"{sp['final_val_loss']:.4f}" if sp else "—"
        mup_loss = f"{mup['final_val_loss']:.4f}" if mup else "—"
        if sp and mup:
            delta = mup['final_val_loss'] - sp['final_val_loss']
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "—"
        print(f"{name:>8s}  {params_str:>10s}  {sp_loss:>10s}  {mup_loss:>10s}  {delta_str:>10s}")


def extrapolate(fit: dict, target_factor: float = 10.0, largest_n: int = 88_100_000) -> dict | None:
    """Predict loss for a model target_factor× larger than the largest.

    Returns dict with predicted loss and 95% confidence interval.
    """
    if fit is None:
        return None
    target_n = largest_n * target_factor
    a, alpha, c = fit["a"], fit["alpha"], fit["c"]
    predicted_loss = power_law_3p(np.array([target_n]), a, alpha, c)[0]

    # 95% CI via Jacobian propagation: J @ pcov @ J^T
    ci_lower, ci_upper = None, None
    if "pcov" in fit and fit["pcov"] is not None:
        pcov = np.array(fit["pcov"])
        # Jacobian of L = a * N^(-alpha) + c w.r.t. (a, alpha, c)
        J = np.array([
            target_n ** (-alpha),                      # dL/da
            -a * target_n ** (-alpha) * np.log(target_n),  # dL/dalpha
            1.0,                                        # dL/dc
        ])
        sigma_L = np.sqrt(J @ pcov @ J.T)
        ci_lower = predicted_loss - 1.96 * sigma_L
        ci_upper = predicted_loss + 1.96 * sigma_L

    print(f"\n=== Extrapolation ({fit['label']}) ===")
    print(f"  Largest model: {largest_n/1e6:.1f}M params")
    print(f"  Target: {target_n/1e6:.0f}M params ({target_factor:.0f}× larger)")
    print(f"  Predicted loss: {predicted_loss:.4f}")
    if ci_lower is not None:
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  (using α={fit['alpha']:.4f}, c={fit['c']:.4f})")

    return {
        "target_n": target_n,
        "predicted_loss": predicted_loss,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze µP vs SP scaling results")
    parser.add_argument("--sp-dir", type=str, default="results/runs/scaling_study")
    parser.add_argument("--mup-dir", type=str, default="results/runs/mup_scaling_study")
    parser.add_argument("--sp-sweep-dir", type=str, default="results/runs/lr_sweep")
    parser.add_argument("--mup-sweep-dir", type=str, default="results/runs/mup_lr_sweep")
    args = parser.parse_args()

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("Loading results...")
    sp_results = load_results(Path(args.sp_dir))
    mup_results = load_results(Path(args.mup_dir))
    sp_sweep = load_lr_sweep(Path(args.sp_sweep_dir))
    mup_sweep = load_lr_sweep(Path(args.mup_sweep_dir))

    print(f"  SP:       {len(sp_results)} models")
    print(f"  µP:       {len(mup_results)} models")
    print(f"  SP sweep: {len(sp_sweep)} LRs")
    print(f"  µP sweep: {len(mup_sweep)} LRs")

    # Plot 1a: µP LR Sweep (standalone)
    print("\n--- µP LR Sweep ---")
    plot_mup_lr_sweep(mup_sweep, plots_dir / "mup_lr_sweep.png")

    # Plot 1b: Combined SP + µP LR Sweep
    print("\n--- SP vs µP LR Sweep ---")
    plot_combined_lr_sweep(sp_sweep, mup_sweep, plots_dir / "sp_vs_mup_lr_sweep.png")

    # Fit power laws
    sp_fit, mup_fit = None, None

    if len(sp_results) >= 3:
        sp_n = np.array([r["n_params"] for r in sp_results])
        sp_l = np.array([r["final_val_loss"] for r in sp_results])
        sp_fit = fit_power_law(sp_n, sp_l, "SP")
        if sp_fit:
            print(f"\n  SP fit:  α={sp_fit['alpha']:.4f}, c={sp_fit['c']:.4f}, R²={sp_fit['r_squared']:.4f}")

    if len(mup_results) >= 3:
        mup_n = np.array([r["n_params"] for r in mup_results])
        mup_l = np.array([r["final_val_loss"] for r in mup_results])
        mup_fit = fit_power_law(mup_n, mup_l, "µP")
        if mup_fit:
            print(f"  µP fit:  α={mup_fit['alpha']:.4f}, c={mup_fit['c']:.4f}, R²={mup_fit['r_squared']:.4f}")

    # Plot 2: SP vs µP Scaling Laws
    print("\n--- SP vs µP Scaling Law ---")
    plot_sp_vs_mup_scaling(sp_results, mup_results, sp_fit, mup_fit,
                           plots_dir / "sp_vs_mup_scaling.png")

    # Plot 3: LR Transfer
    print("\n--- LR Transfer ---")
    plot_lr_transfer(sp_results, mup_results, plots_dir / "lr_transfer.png")

    # Comparison table
    print_comparison(sp_results, mup_results)

    # Extrapolation
    extrap_results = {}
    if mup_fit:
        largest = max(r["n_params"] for r in mup_results)
        extrap_results["mup"] = extrapolate(mup_fit, target_factor=10.0, largest_n=largest)
    if sp_fit:
        largest = max(r["n_params"] for r in sp_results)
        extrap_results["sp"] = extrapolate(sp_fit, target_factor=10.0, largest_n=largest)

    # Save fit results
    fit_export = {}
    if sp_fit:
        fit_export["sp"] = {k: v for k, v in sp_fit.items() if k != "predict_fn"}
    if mup_fit:
        fit_export["mup"] = {k: v for k, v in mup_fit.items() if k != "predict_fn"}
    if extrap_results:
        fit_export["extrapolation"] = extrap_results
    if fit_export:
        out_path = plots_dir / "mup_fit_results.json"
        with open(out_path, "w") as f:
            json.dump(fit_export, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
