#!/usr/bin/env python3
"""
charts.py

Generate charts from sweep_results.csv produced by sweep_launch_configs.py.

Outputs (PNG):
  1) runtime_vs_threads_blocksX.png
  2) runtime_vs_blocks_threadsY.png
  3) speedup_vs_threads_blocksX.png
  4) speedup_vs_blocks_threadsY.png
  5) gpu_branch_penalty_vs_threads_blocksX.png
  6) gpu_branch_penalty_vs_blocks_threadsY.png
  7) scatter_cpu_vs_gpu.png

Usage:
  python charts.py --csv sweep_out/sweep_results.csv --out charts_out

Optional:
  python charts.py --csv sweep_out/sweep_results.csv --out charts_out \
      --blocks_fixed 1024 --threads_fixed 256

Notes:
- Assumes CSV has columns:
  requested_blocks, requested_threads,
  gpu_baseline_ns, gpu_branched_ns,
  cpu_baseline_ns, cpu_branched_ns
- Uses requested_* for chart axes (what you swept). If you want to plot final clamped
  configs instead, change to final_blocks/final_threads.
"""

import argparse
import csv
from pathlib import Path
from math import isnan

import matplotlib.pyplot as plt


def to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_csv_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "requested_blocks": int(r["requested_blocks"]),
                "requested_threads": int(r["requested_threads"]),
                "gpu_baseline_ns": to_float_or_nan(r.get("gpu_baseline_ns", "")),
                "gpu_branched_ns": to_float_or_nan(r.get("gpu_branched_ns", "")),
                "cpu_baseline_ns": to_float_or_nan(r.get("cpu_baseline_ns", "")),
                "cpu_branched_ns": to_float_or_nan(r.get("cpu_branched_ns", "")),
            })
    return rows


def unique_sorted(rows, key):
    return sorted(set(r[key] for r in rows))


def pick_default(value_list, preferred=None):
    """
    Pick a reasonable default.
    - If preferred exists, use it.
    - Else pick the middle value (median-ish).
    """
    if preferred is not None and preferred in value_list:
        return preferred
    if not value_list:
        return None
    return value_list[len(value_list) // 2]


def filter_rows(rows, **conds):
    out = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def save_lineplot(x, series, xlabel, ylabel, title, out_png, xlog=False, ylog=False):
    plt.figure()
    for label, y in series:
        plt.plot(x, y, marker="o", label=label)
    if xlog:
        plt.xscale("log", base=2)
    if ylog:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_scatter(x, y, xlabel, ylabel, title, out_png):
    plt.figure()
    plt.scatter(x, y)
    # y=x reference line (using min/max of combined data)
    finite = [v for v in (x + y) if v == v]  # v==v filters NaN
    if finite:
        lo = min(finite)
        hi = max(finite)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def ns_to_ms(arr):
    return [v / 1e6 for v in arr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sweep_results.csv")
    ap.add_argument("--out", default="charts_out", help="Output directory for PNGs")
    ap.add_argument("--blocks_fixed", type=int, default=None,
                    help="Blocks value to use for the runtime-vs-threads plots")
    ap.add_argument("--threads_fixed", type=int, default=None,
                    help="Threads value to use for the runtime-vs-blocks plots")
    ap.add_argument("--units", choices=["ns", "ms"], default="ms",
                    help="Y-axis units for runtime charts")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(csv_path)

    blocks_list = unique_sorted(rows, "requested_blocks")
    threads_list = unique_sorted(rows, "requested_threads")

    blocks_fixed = pick_default(blocks_list, args.blocks_fixed)
    threads_fixed = pick_default(threads_list, args.threads_fixed)

    if blocks_fixed is None or threads_fixed is None:
        raise SystemExit("Could not determine blocks/threads defaults from CSV.")

    # -----------------------------
    # 1) Runtime vs Threads (blocks fixed)
    # -----------------------------
    sub = filter_rows(rows, requested_blocks=blocks_fixed)
    sub = sorted(sub, key=lambda r: r["requested_threads"])
    x_threads = [r["requested_threads"] for r in sub]

    gpu_base = [r["gpu_baseline_ns"] for r in sub]
    gpu_bran = [r["gpu_branched_ns"] for r in sub]
    cpu_base = [r["cpu_baseline_ns"] for r in sub]
    cpu_bran = [r["cpu_branched_ns"] for r in sub]

    if args.units == "ms":
        y_gpu_base = ns_to_ms(gpu_base)
        y_gpu_bran = ns_to_ms(gpu_bran)
        y_cpu_base = ns_to_ms(cpu_base)
        y_cpu_bran = ns_to_ms(cpu_bran)
        ylabel = "Runtime (ms)"
    else:
        y_gpu_base = gpu_base
        y_gpu_bran = gpu_bran
        y_cpu_base = cpu_base
        y_cpu_bran = cpu_bran
        ylabel = "Runtime (ns)"

    save_lineplot(
        x_threads,
        [
            ("GPU baseline", y_gpu_base),
            ("GPU branched", y_gpu_bran),
            ("CPU baseline", y_cpu_base),
            ("CPU branched", y_cpu_bran),
        ],
        xlabel="Threads per block (requested)",
        ylabel=ylabel,
        title=f"Runtime vs Threads (blocks fixed at {blocks_fixed})",
        out_png=outdir / f"runtime_vs_threads_blocks{blocks_fixed}.png",
        xlog=False,
        ylog=False
    )

    # -----------------------------
    # 2) Runtime vs Blocks (threads fixed) [log2 x-axis]
    # -----------------------------
    sub = filter_rows(rows, requested_threads=threads_fixed)
    sub = sorted(sub, key=lambda r: r["requested_blocks"])
    x_blocks = [r["requested_blocks"] for r in sub]

    gpu_base = [r["gpu_baseline_ns"] for r in sub]
    gpu_bran = [r["gpu_branched_ns"] for r in sub]
    cpu_base = [r["cpu_baseline_ns"] for r in sub]
    cpu_bran = [r["cpu_branched_ns"] for r in sub]

    if args.units == "ms":
        y_gpu_base = ns_to_ms(gpu_base)
        y_gpu_bran = ns_to_ms(gpu_bran)
        y_cpu_base = ns_to_ms(cpu_base)
        y_cpu_bran = ns_to_ms(cpu_bran)
        ylabel = "Runtime (ms)"
    else:
        y_gpu_base = gpu_base
        y_gpu_bran = gpu_bran
        y_cpu_base = cpu_base
        y_cpu_bran = cpu_bran
        ylabel = "Runtime (ns)"

    save_lineplot(
        x_blocks,
        [
            ("GPU baseline", y_gpu_base),
            ("GPU branched", y_gpu_bran),
            ("CPU baseline", y_cpu_base),
            ("CPU branched", y_cpu_bran),
        ],
        xlabel="Blocks (requested)",
        ylabel=ylabel,
        title=f"Runtime vs Blocks (threads fixed at {threads_fixed})",
        out_png=outdir / f"runtime_vs_blocks_threads{threads_fixed}.png",
        xlog=True,
        ylog=False
    )

    # -----------------------------
    # 3) Speedup plots: CPU / GPU (baseline + branched)
    # -----------------------------
    # Speedup vs threads (blocks fixed)
    sub = filter_rows(rows, requested_blocks=blocks_fixed)
    sub = sorted(sub, key=lambda r: r["requested_threads"])
    x_threads = [r["requested_threads"] for r in sub]

    def safe_ratio(num, den):
        out = []
        for a, b in zip(num, den):
            if a == a and b == b and b != 0:
                out.append(a / b)
            else:
                out.append(float("nan"))
        return out

    speedup_base = safe_ratio([r["cpu_baseline_ns"] for r in sub],
                              [r["gpu_baseline_ns"] for r in sub])
    speedup_bran = safe_ratio([r["cpu_branched_ns"] for r in sub],
                              [r["gpu_branched_ns"] for r in sub])

    save_lineplot(
        x_threads,
        [("Speedup baseline (CPU/GPU)", speedup_base),
         ("Speedup branched (CPU/GPU)", speedup_bran)],
        xlabel="Threads per block (requested)",
        ylabel="Speedup (×)",
        title=f"GPU Speedup vs Threads (blocks fixed at {blocks_fixed})",
        out_png=outdir / f"speedup_vs_threads_blocks{blocks_fixed}.png",
        xlog=False,
        ylog=False
    )

    # Speedup vs blocks (threads fixed)
    sub = filter_rows(rows, requested_threads=threads_fixed)
    sub = sorted(sub, key=lambda r: r["requested_blocks"])
    x_blocks = [r["requested_blocks"] for r in sub]

    speedup_base = safe_ratio([r["cpu_baseline_ns"] for r in sub],
                              [r["gpu_baseline_ns"] for r in sub])
    speedup_bran = safe_ratio([r["cpu_branched_ns"] for r in sub],
                              [r["gpu_branched_ns"] for r in sub])

    save_lineplot(
        x_blocks,
        [("Speedup baseline (CPU/GPU)", speedup_base),
         ("Speedup branched (CPU/GPU)", speedup_bran)],
        xlabel="Blocks (requested)",
        ylabel="Speedup (×)",
        title=f"GPU Speedup vs Blocks (threads fixed at {threads_fixed})",
        out_png=outdir / f"speedup_vs_blocks_threads{threads_fixed}.png",
        xlog=True,
        ylog=False
    )

    # -----------------------------
    # 4) Branching penalty (GPU branched / GPU baseline)
    # -----------------------------
    sub = filter_rows(rows, requested_blocks=blocks_fixed)
    sub = sorted(sub, key=lambda r: r["requested_threads"])
    x_threads = [r["requested_threads"] for r in sub]
    branch_penalty_threads = safe_ratio([r["gpu_branched_ns"] for r in sub],
                                        [r["gpu_baseline_ns"] for r in sub])

    save_lineplot(
        x_threads,
        [("GPU branching penalty (branched/baseline)", branch_penalty_threads)],
        xlabel="Threads per block (requested)",
        ylabel="Penalty ratio (×)",
        title=f"GPU Branching Penalty vs Threads (blocks fixed at {blocks_fixed})",
        out_png=outdir / f"gpu_branch_penalty_vs_threads_blocks{blocks_fixed}.png",
        xlog=False,
        ylog=False
    )

    sub = filter_rows(rows, requested_threads=threads_fixed)
    sub = sorted(sub, key=lambda r: r["requested_blocks"])
    x_blocks = [r["requested_blocks"] for r in sub]
    branch_penalty_blocks = safe_ratio([r["gpu_branched_ns"] for r in sub],
                                       [r["gpu_baseline_ns"] for r in sub])

    save_lineplot(
        x_blocks,
        [("GPU branching penalty (branched/baseline)", branch_penalty_blocks)],
        xlabel="Blocks (requested)",
        ylabel="Penalty ratio (×)",
        title=f"GPU Branching Penalty vs Blocks (threads fixed at {threads_fixed})",
        out_png=outdir / f"gpu_branch_penalty_vs_blocks_threads{threads_fixed}.png",
        xlog=True,
        ylog=False
    )

    # -----------------------------
    # 5) Scatter: CPU vs GPU (baseline and branched)
    # -----------------------------
    # Use ms for readability unless user wants ns
    if args.units == "ms":
        cpu_b = [r["cpu_baseline_ns"] / 1e6 for r in rows]
        gpu_b = [r["gpu_baseline_ns"] / 1e6 for r in rows]
        cpu_br = [r["cpu_branched_ns"] / 1e6 for r in rows]
        gpu_br = [r["gpu_branched_ns"] / 1e6 for r in rows]
        xlabel = "CPU runtime (ms)"
        ylabel = "GPU runtime (ms)"
    else:
        cpu_b = [r["cpu_baseline_ns"] for r in rows]
        gpu_b = [r["gpu_baseline_ns"] for r in rows]
        cpu_br = [r["cpu_branched_ns"] for r in rows]
        gpu_br = [r["gpu_branched_ns"] for r in rows]
        xlabel = "CPU runtime (ns)"
        ylabel = "GPU runtime (ns)"

    save_scatter(cpu_b, gpu_b, xlabel, ylabel, "CPU vs GPU Scatter (Baseline)", outdir / "scatter_cpu_vs_gpu_baseline.png")
    save_scatter(cpu_br, gpu_br, xlabel, ylabel, "CPU vs GPU Scatter (Branched)", outdir / "scatter_cpu_vs_gpu_branched.png")

    print(f"Done. Wrote charts to: {outdir}")
    print(f"Used blocks_fixed={blocks_fixed}, threads_fixed={threads_fixed}")


if __name__ == "__main__":
    main()
