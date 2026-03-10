#!/usr/bin/env python3
"""
plot_summary.py

Read runner.py summary.csv and generate benchmark plots for the
CUDA Streams and Events assignment.

Usage:
    python3 plot_summary.py --csv summary.csv
    python3 plot_summary.py --csv runner_results/summary.csv --outdir plots

If the CSV contains optional columns like native_covered / scaled_covered,
you can filter to only valid runs:
    python3 plot_summary.py --csv summary.csv --valid-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark summary CSV.")
    parser.add_argument("--csv", required=True, help="Path to summary.csv")
    parser.add_argument("--outdir", default="plots", help="Directory for output plots")
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Keep only rows where native_covered and scaled_covered are true, if present",
    )
    return parser.parse_args()


def load_data(csv_path: Path, valid_only: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {
        "threads",
        "blocks",
        "repeat",
        "total_gpu_ms_mean",
        "fps_estimate",
        "fr_up_ms_mean",
        "lg_up_ms_mean",
        "self_diff_ms_mean",
        "cross_diff_ms_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if valid_only and {"native_covered", "scaled_covered"}.issubset(df.columns):
        df = df[(df["native_covered"].astype(bool)) & (df["scaled_covered"].astype(bool))].copy()

    df = df.sort_values(["threads", "blocks", "repeat"]).reset_index(drop=True)
    return df


def save_scatter_all_runs(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for blocks in sorted(df["blocks"].unique()):
        sub = df[df["blocks"] == blocks]
        plt.scatter(sub["threads"], sub["total_gpu_ms_mean"], label=f"blocks={blocks}", s=70)

    plt.xlabel("Threads per block")
    plt.ylabel("Mean total GPU time (ms)")
    plt.title("All benchmark runs: mean total GPU time vs threads")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_total_gpu_vs_threads.png", dpi=200)
    plt.close()


def save_lines_by_block(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for blocks in sorted(df["blocks"].unique()):
        sub = df[df["blocks"] == blocks].sort_values("threads")
        plt.plot(sub["threads"], sub["total_gpu_ms_mean"], marker="o", label=f"blocks={blocks}")

    plt.xlabel("Threads per block")
    plt.ylabel("Mean total GPU time (ms)")
    plt.title("Mean total GPU time across thread settings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "lines_total_gpu_by_block.png", dpi=200)
    plt.close()


def save_lines_by_thread(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for threads in sorted(df["threads"].unique()):
        sub = df[df["threads"] == threads].sort_values("blocks")
        plt.plot(sub["blocks"], sub["total_gpu_ms_mean"], marker="o", label=f"threads={threads}")

    plt.xlabel("Blocks")
    plt.ylabel("Mean total GPU time (ms)")
    plt.title("Mean total GPU time across block settings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "lines_total_gpu_by_thread.png", dpi=200)
    plt.close()


def save_top_configs(df: pd.DataFrame, outdir: Path, top_n: int = 10) -> None:
    best = df.nsmallest(top_n, "total_gpu_ms_mean").copy()
    best["label"] = best.apply(
        lambda r: f"T{int(r['threads'])}\nB{int(r['blocks'])}", axis=1
    )

    plt.figure(figsize=(10, 6))
    plt.bar(best["label"], best["total_gpu_ms_mean"])
    plt.xlabel("Configuration")
    plt.ylabel("Mean total GPU time (ms)")
    plt.title(f"Top {top_n} configurations by GPU time")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "top_configs_bar.png", dpi=200)
    plt.close()


def save_heatmap(df: pd.DataFrame, outdir: Path, value_col: str, filename: str, title: str) -> None:
    pivot = df.pivot_table(index="threads", columns="blocks", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Blocks")
    plt.ylabel("Threads")
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=200)
    plt.close()


def save_kernel_breakdown(df: pd.DataFrame, outdir: Path) -> None:
    best = df.nsmallest(1, "total_gpu_ms_mean").iloc[0]

    labels = [
        "FireRed upscale",
        "LeafGreen upscale",
        "Self heatmap",
        "Cross heatmap",
    ]
    values = [
        best["fr_up_ms_mean"],
        best["lg_up_ms_mean"],
        best["self_diff_ms_mean"],
        best["cross_diff_ms_mean"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean kernel time (ms)")
    plt.title(
        f"Kernel timing breakdown for best config "
        f"(threads={int(best['threads'])}, blocks={int(best['blocks'])})"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "best_config_kernel_breakdown.png", dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path, args.valid_only)

    if df.empty:
        raise ValueError("No rows available to plot after filtering.")

    save_scatter_all_runs(df, outdir)
    save_lines_by_block(df, outdir)
    save_lines_by_thread(df, outdir)
    save_top_configs(df, outdir, top_n=min(10, len(df)))
    save_heatmap(
        df,
        outdir,
        value_col="total_gpu_ms_mean",
        filename="heatmap_total_gpu_ms.png",
        title="Heatmap of mean total GPU time (ms)",
    )
    save_heatmap(
        df,
        outdir,
        value_col="fps_estimate",
        filename="heatmap_fps.png",
        title="Heatmap of FPS estimate",
    )
    save_kernel_breakdown(df, outdir)

    best = df.nsmallest(1, "total_gpu_ms_mean").iloc[0]
    print("[BEST CONFIG]")
    print(
        f"threads={int(best['threads'])} "
        f"blocks={int(best['blocks'])} "
        f"mean_total_gpu_ms={best['total_gpu_ms_mean']:.4f} "
        f"fps_estimate={best['fps_estimate']:.2f}"
    )
    print(f"[PLOTS WRITTEN] {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())