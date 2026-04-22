#!/usr/bin/env python3
"""Plot success rate vs epoch for mu=-2 and mu=0, styled after sigreg_v_rdmreg."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

EPOCHS = list(range(1, 11))

REPO = Path("/home/ubuntu/le-wm-palash")
SWEEPS = {
    r"$\mu = 0$ (baseline)":
        REPO / "lp_jepa" / "eval_sweep_pusht_expert.log",
    r"$\mu = -2$":
        REPO / "lewm-pusht" / "lp_jepa_mu_sweep" / "mu-2",
}


def _pct(val: float) -> float:
    return val / 100.0 if val > 1.0 else val


def load_baseline_log(log_path: Path) -> dict[int, float]:
    text = log_path.read_text()
    rates = re.findall(r"'success_rate':\s*([0-9.]+)", text)
    return {e: _pct(float(r)) for e, r in zip(EPOCHS, rates[: len(EPOCHS)])}


def load_per_epoch_dir(dir_path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for e in EPOCHS:
        f = dir_path / f"epoch{e}.txt"
        if not f.exists():
            out[e] = float("nan")
            continue
        m = re.search(r"'success_rate':\s*([0-9.]+)", f.read_text())
        out[e] = _pct(float(m.group(1))) if m else float("nan")
    return out


def plot(out: str = "mu_comparison.pdf") -> None:
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

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    palette = cm.get_cmap("plasma", len(SWEEPS) + 1)
    colors = [palette(i / len(SWEEPS)) for i in range(len(SWEEPS))]

    for color, (label, src) in zip(colors, SWEEPS.items()):
        data = load_baseline_log(src) if src.is_file() else load_per_epoch_dir(src)
        ys = [data.get(e, float("nan")) for e in EPOCHS]
        ax.plot(
            EPOCHS, ys,
            marker="o", markersize=4.5, linewidth=1.5,
            color=color, label=label, zorder=3,
        )

    ax.set_title(r"LeWM + RDMReg: $\mu$ sweep", fontsize=11, pad=6)
    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_xticks(EPOCHS)
    ax.set_xlim(0.5, len(EPOCHS) + 0.5)
    ax.set_ylim(-0.02, 1.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        title=r"$\mu$", title_fontsize=9, fontsize=9,
        loc="lower right", frameon=True, handlelength=1.6,
    )

    plt.tight_layout(pad=0.6)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Plot saved → {Path(out).resolve()}")


if __name__ == "__main__":
    plot()
