#!/usr/bin/env python3
"""
Sweep lewm_bisim checkpoints across CEM n_steps values and generate a publication plot.

Usage (from inside the le-wm conda env):
    python sweep_bisim.py

Produces:  bisim_sweep.pdf
"""

import ast
import concurrent.futures
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── sweep parameters ──────────────────────────────────────────────────────────
EPOCHS       = list(range(1, 11))
N_STEPS      = [1, 5, 10, 20, 30]
MAX_PARALLEL = 2          # ≈6 GB VRAM each → 2 × 6 = 12 GB total
MODEL_PREFIX = "lewm_bisim_epoch"
REPO_DIR     = Path(__file__).parent.resolve()
# ─────────────────────────────────────────────────────────────────────────────


def _parse_success(stdout: str) -> float:
    """Extract success rate from the dict printed by eval.py."""
    # Strategy 1: regex scan for known keys anywhere in stdout
    for key in ("success_rate", "success", "coverage", "avg_success"):
        m = re.search(rf"['\"]?{key}['\"]?\s*:\s*([0-9.eE+\-]+)", stdout)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val  # normalise % → fraction
    # Strategy 2: find a {...} block and ast-parse it
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "}" in line:
            try:
                d = ast.literal_eval(line)
                if isinstance(d, dict):
                    for v in d.values():
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            continue
            except Exception:
                continue
    return float("nan")


def run_single(epoch: int, n_steps: int) -> tuple:
    label  = f"epoch={epoch:2d}  n_steps={n_steps:2d}"
    policy = f"{MODEL_PREFIX}_{epoch}"
    outfile = f"bisim_sweep/epoch{epoch:02d}_nsteps{n_steps:02d}.txt"

    cmd = [
        sys.executable, "eval.py",
        f"policy={policy}",
        f"solver.n_steps={n_steps}",
        f"output.filename={outfile}",
    ]

    print(f"[START] {label}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_DIR),
    )

    if proc.returncode != 0:
        tail = proc.stderr[-600:] or proc.stdout[-600:]
        print(f"[FAIL]  {label}\n{tail}")
        return epoch, n_steps, float("nan")

    sr = _parse_success(proc.stdout)
    if np.isnan(sr):
        print(f"[WARN]  {label}  →  could not parse metrics. stdout tail:\n{proc.stdout[-800:]}")
    print(f"[DONE]  {label}  →  success={sr:.3f}")
    return epoch, n_steps, sr


def plot(results: list, out: str = "bisim_sweep.pdf") -> None:
    """Single publication-quality axes: one line per epoch, x=n_steps, y=success."""

    data: dict = {}
    for epoch, n_steps, sr in results:
        data.setdefault(epoch, {})[n_steps] = sr

    epochs_sorted = sorted(data)
    n_epochs = len(epochs_sorted)

    plt.rcParams.update({
        "font.family":         "serif",
        "font.size":           10,
        "axes.linewidth":      0.8,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.framealpha":   0.9,
        "legend.edgecolor":    "0.75",
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # viridis: dark purple (early) → yellow-green (late)
    palette = cm.get_cmap("viridis", n_epochs + 2)
    colors  = [palette(i / (n_epochs + 1)) for i in range(1, n_epochs + 1)]

    for color, epoch in zip(colors, epochs_sorted):
        xs = sorted(data[epoch])
        ys = [data[epoch].get(x, float("nan")) for x in xs]
        ax.plot(
            xs, ys,
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=color,
            label=f"Epoch {epoch}",
            zorder=3,
        )

    ax.set_xlabel(r"CEM Optimization Steps ($n_{\mathrm{steps}}$)", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_xticks(N_STEPS)
    ax.set_xticklabels(N_STEPS)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(left=-0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        fontsize=8,
        ncol=2,
        loc="lower right",
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    ax.annotate(
        "earlier → later epoch",
        xy=(0.98, 0.05), xycoords="axes fraction",
        fontsize=7, color="0.45",
        ha="right", va="bottom",
    )

    plt.tight_layout(pad=0.6)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


def main() -> None:
    jobs = [(e, s) for e in EPOCHS for s in N_STEPS]
    print(f"Running {len(jobs)} eval jobs with {MAX_PARALLEL} parallel workers.\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futs = {pool.submit(run_single, e, s): (e, s) for e, s in jobs}
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    results.sort()

    print("\n╔══ SWEEP SUMMARY ════════════════════════════════╗")
    for epoch, n_steps, sr in results:
        bar = "█" * int(sr * 25) if not np.isnan(sr) else "  (failed)"
        print(f"  epoch={epoch:2d}  n_steps={n_steps:2d}  {sr:.3f}  {bar}")
    print("╚══════════════════════════════════════════════════╝\n")

    plot(results)


if __name__ == "__main__":
    main()