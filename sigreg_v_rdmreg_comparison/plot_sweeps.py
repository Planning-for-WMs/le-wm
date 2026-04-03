#!/usr/bin/env python3
"""
Plot sweep results with x=epoch, colormap=n_steps.
Reads existing result files — no re-running eval.

Usage: python plot_sweeps.py
Produces: sweeps_by_epoch.pdf
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import stable_worldmodel as swm

# ── parameters ────────────────────────────────────────────────────────────────
EPOCHS  = list(range(1, 11))
N_STEPS = [1, 5, 10, 20, 30]
CACHE   = Path(swm.data.utils.get_cache_dir())

SWEEPS = {
    "LeWM + Bisim":   "bisim_sweep",
    "LeWM + RDMReg":  "rdmreg_sweep",
}
# ─────────────────────────────────────────────────────────────────────────────


def _parse_file(path: Path) -> float:
    if not path.exists():
        return float("nan")
    text = path.read_text()
    for key in ("success_rate", "success", "coverage", "avg_success"):
        m = re.search(rf"['\"]?{key}['\"]?\s*:\s*([0-9.eE+\-]+)", text)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
    return float("nan")


def load_sweep(sweep_dir: str) -> dict:
    """Returns data[epoch][n_steps] = success_rate."""
    data: dict = {}
    for epoch in EPOCHS:
        for n_steps in N_STEPS:
            path = CACHE / sweep_dir / f"epoch{epoch:02d}_nsteps{n_steps:02d}.txt"
            data.setdefault(epoch, {})[n_steps] = _parse_file(path)
    return data


def plot(out: str = "sweeps_by_epoch.pdf") -> None:
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

    n_panels = len(SWEEPS)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.0),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # colormap for n_steps: few → dark, many → bright
    palette = cm.get_cmap("plasma", len(N_STEPS) + 1)
    colors  = [palette(i / len(N_STEPS)) for i in range(len(N_STEPS))]

    for ax, (title, sweep_dir) in zip(axes, SWEEPS.items()):
        data = load_sweep(sweep_dir)

        for color, n_steps in zip(colors, N_STEPS):
            xs = EPOCHS
            ys = [data.get(e, {}).get(n_steps, float("nan")) for e in xs]
            ax.plot(
                xs, ys,
                marker="o",
                markersize=4.5,
                linewidth=1.5,
                color=color,
                label=f"$n={n_steps}$",
                zorder=3,
            )

        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Training Epoch", fontsize=11)
        ax.set_xticks(EPOCHS)
        ax.set_xlim(0.5, len(EPOCHS) + 0.5)
        ax.set_ylim(-0.02, 1.04)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Success Rate", fontsize=11)

    # shared legend outside the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title=r"$n_{\mathrm{steps}}$",
        title_fontsize=9,
        fontsize=8,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        handlelength=1.6,
    )

    plt.tight_layout(pad=0.6, rect=[0, 0, 0.88, 1])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    plot()