"""
benchmark.py — grid search over inference hyperparameters, no training required.

Usage:
    python benchmark.py --vae_model_path checkpoints/VQVAE.pth \
                        --var_test_path   checkpoints/VARSR.pth

Results are saved to benchmark_results.csv and printed as a comparison table.
Each config writes predictions to its own subfolder so outputs don't overwrite each other.
"""

import os
import sys
import csv
import copy
import time
import itertools

import torch

# ── bootstrap distributed (single-GPU / CPU mode) ─────────────────────────────
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import dist
from utils import arg_util
from test_varsr import main, metrics


# ── grid of configurations to sweep ───────────────────────────────────────────
# Add or remove entries freely. Every combination will be run.
GRID = {
    "cfg":       [5.0, 6.0, 7.0, 8.0],
    "top_k":     [900],           # 900 = paper-style nucleus; 1 = greedy (current default)
    "top_p":     [0.95],
    "diff_temp": [0.6, 0.75, 1.0],
    "color_fix": ["adain", "wavelet"],
}

# Fixed seed so every config is comparable
INFER_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────


def make_tag(cfg_val, top_k, top_p, diff_temp, color_fix):
    return f"VARPrediction_cfg{cfg_val}_tk{top_k}_tp{top_p}_dt{diff_temp}_{color_fix}"


def run_grid():
    base_args: arg_util.Args = arg_util.init_dist_and_get_args()
    base_args.infer_seed = INFER_SEED

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    print(f"\n{'='*60}")
    print(f"  VARSR Benchmark — {len(combos)} configurations")
    print(f"  Seed: {INFER_SEED}")
    print(f"{'='*60}\n")

    all_rows = []

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        tag = make_tag(**params)

        print(f"\n[{i}/{len(combos)}] {params}")

        # Apply params to a fresh copy of args so builds don't interfere
        args = copy.copy(base_args)
        for k, v in params.items():
            setattr(args, k, v)

        t0 = time.time()
        main(args, output_tag=tag)
        elapsed = time.time() - t0
        print(f"  inference done in {elapsed:.1f}s")

        folder_results = metrics(output_tag=tag)

        # Average metrics across all test folders
        if folder_results:
            avg = {}
            for metric_key in next(iter(folder_results.values())).keys():
                avg[metric_key] = sum(r[metric_key] for r in folder_results.values()) / len(folder_results)
            row = {**params, "tag": tag, **{f"avg_{k}": v for k, v in avg.items()}}
            all_rows.append(row)

    # ── print comparison table ─────────────────────────────────────────────────
    if not all_rows:
        print("\nNo results collected.")
        return

    metric_cols = [k for k in all_rows[0] if k.startswith("avg_")]
    param_cols  = keys + ["tag"]

    # Sort by avg PSNR descending
    all_rows.sort(key=lambda r: r.get("avg_psnr", 0), reverse=True)

    header = param_cols + metric_cols
    col_w  = max(len(h) for h in header) + 2

    print(f"\n{'='*60}")
    print("  RESULTS (sorted by avg PSNR ↓)")
    print(f"{'='*60}")
    print("  " + "".join(h.ljust(col_w) for h in header))
    print("  " + "-" * (col_w * len(header)))
    for row in all_rows:
        line = ""
        for h in header:
            v = row.get(h, "")
            line += (f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(col_w)
        print("  " + line)

    # ── save CSV ───────────────────────────────────────────────────────────────
    csv_path = "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved to {csv_path}")

    # ── highlight best per metric ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BEST CONFIG PER METRIC")
    print(f"{'='*60}")
    lower_is_better = {"avg_lpips", "avg_dists", "avg_niqe", "avg_fid"}
    for mc in metric_cols:
        best = min(all_rows, key=lambda r: r[mc]) if mc in lower_is_better else max(all_rows, key=lambda r: r[mc])
        direction = "↓" if mc in lower_is_better else "↑"
        print(f"  {mc:<20} {direction}  {best[mc]:.4f}  ←  cfg={best['cfg']}  diff_temp={best['diff_temp']}  color_fix={best['color_fix']}")


if __name__ == "__main__":
    run_grid()
