#!/usr/bin/env python3
"""
Sweep evaluation across LeWM checkpoints trained with different values of the
RDMReg shape parameter p, and write a CSV summary so the effect of p on
downstream success rate is easy to compare.

Assumes you've already trained one checkpoint per p value via:

    python lp_jepa/train_lp_jepa.py loss.rdmreg.kwargs.p=0.5
    python lp_jepa/train_lp_jepa.py loss.rdmreg.kwargs.p=1.0
    python lp_jepa/train_lp_jepa.py loss.rdmreg.kwargs.p=2.0

Each training run writes a checkpoint named ``lewm_lp_p<p>`` (because of the
``output_model_name`` interpolation in ``config/train/lewm_lp_jepa.yaml``).
This script loops over the configured p values, calls the project-root
``eval.py`` once per checkpoint with ``policy=lewm_lp_p<p>``, parses the
printed metrics dict, and writes ``lp_jepa_eval.csv``.

Usage (from the repo root):

    python lp_jepa/eval_lp_jepa.py
    python lp_jepa/eval_lp_jepa.py --p 0.5 1.0 2.0 --n-steps 10
"""

import argparse
import ast
import concurrent.futures
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_P_VALUES = [0.5, 1.0, 2.0]
DEFAULT_N_STEPS = 10
MAX_PARALLEL = 2  # ≈6 GB VRAM each → 2 × 6 = 12 GB total


def _parse_success(stdout: str) -> float:
    """Extract success rate from the dict printed by eval.py.

    Mirrors the parser used in sigreg_v_rdmreg_comparison/sweep_rdmreg.py.
    """
    for key in ("success_rate", "success", "coverage", "avg_success"):
        m = re.search(rf"['\"]?{key}['\"]?\s*:\s*([0-9.eE+\-]+)", stdout)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
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


def run_single(p: float, n_steps: int) -> tuple:
    # Match the checkpoint name produced by lewm_lp_jepa.yaml's
    # `output_model_name: lewm_lp_p${loss.rdmreg.kwargs.p}` interpolation.
    policy = f"lewm_lp_p{p}"
    label = f"p={p}  n_steps={n_steps}"
    outfile = f"lp_jepa_eval/p{p}_nsteps{n_steps:02d}.txt"

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
        return p, n_steps, float("nan")

    sr = _parse_success(proc.stdout)
    print(f"[DONE]  {label}  →  success={sr:.3f}")
    return p, n_steps, sr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p", type=float, nargs="+", default=DEFAULT_P_VALUES,
        help="p values to evaluate (must already be trained).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=DEFAULT_N_STEPS,
        help="solver.n_steps to pass to eval.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "lp_jepa_eval.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=MAX_PARALLEL,
        help="Number of eval jobs to run concurrently.",
    )
    args = parser.parse_args()

    jobs = [(p, args.n_steps) for p in args.p]
    print(f"Running {len(jobs)} eval jobs with {args.max_parallel} parallel workers.\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futs = {pool.submit(run_single, p, n): (p, n) for p, n in jobs}
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    # Sort by p for stable output
    results.sort(key=lambda r: r[0])

    print("\n╔══ p SWEEP SUMMARY ═════════════════════════════╗")
    for p, n_steps, sr in results:
        bar = "█" * int(sr * 25) if sr == sr else "  (failed)"  # NaN check
        print(f"  p={p:<5}  n_steps={n_steps:<3}  {sr:.3f}  {bar}")
    print("╚══════════════════════════════════════════════════╝\n")

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "n_steps", "success_rate"])
        for p, n_steps, sr in results:
            writer.writerow([p, n_steps, sr])
    print(f"CSV written → {args.out}")


if __name__ == "__main__":
    main()
