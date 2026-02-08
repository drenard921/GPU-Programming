#!/usr/bin/env python3
"""
sweep_launch_configs.py

Runs assignment.exe across a grid of (blocks, threads_per_block) configs,
parses timing output, and produces:
  - per-run stdout logs
  - summary CSV
  - basic heatmap charts

Usage:
  python sweep_launch_configs.py --exe ./assignment.exe --out sweep_out

Notes:
  - Assumes assignment.exe prints the "Final launch configuration" and
    the four timing lines:
      GPU baseline:
      GPU branched:
      CPU baseline:
      CPU branched:
"""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


# --------- EDIT THESE LISTS AS YOU WANT ----------
BLOCKS = [1, 4, 16, 64, 256, 1024, 4096, 16384]

# Your current list (6 values) => 48 runs:
THREADS = [8, 16, 32, 64, 128, 256, 512, 1024]

# If you want a true 8x8 (64 runs), use one of these instead:
# THREADS = [32, 64, 96, 128, 256, 384, 512, 1024]
# THREADS = [32, 64, 128, 192, 256, 512, 768, 1024]
# -----------------------------------------------


RE_GPU_BASE = re.compile(r"GPU baseline:\s+(\d+)")
RE_GPU_BRAN = re.compile(r"GPU branched:\s+(\d+)")
RE_CPU_BASE = re.compile(r"CPU baseline:\s+(\d+)")
RE_CPU_BRAN = re.compile(r"CPU branched:\s+(\d+)")

RE_FINAL_BLOCKS = re.compile(r"Blocks:\s+(\d+)")
RE_FINAL_THREADS = re.compile(r"Threads per block:\s+(\d+)")

def run_once(exe: str, blocks: int, threads: int) -> tuple[int, str]:
    """Run the executable once and return (returncode, stdout+stderr)."""
    proc = subprocess.run(
        [exe, str(blocks), str(threads)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    return proc.returncode, proc.stdout


def parse_timings(output: str) -> dict:
    """Extract timing numbers (ns) from stdout. Returns dict with None if missing."""
    def m_int(regex):
        m = regex.search(output)
        return int(m.group(1)) if m else None

    return {
        "gpu_baseline_ns": m_int(RE_GPU_BASE),
        "gpu_branched_ns": m_int(RE_GPU_BRAN),
        "cpu_baseline_ns": m_int(RE_CPU_BASE),
        "cpu_branched_ns": m_int(RE_CPU_BRAN),
    }


def parse_final_config(output: str) -> dict:
    """
    Try to parse the final clamped launch config from stdout.
    Your program prints:
      Final launch configuration:
        Blocks: ...
        Threads per block: ...
    We'll just grab the first occurrences after that section appears.
    """
    # A simple approach: just pick the first "Blocks:" and "Threads per block:"
    # shown in the "Final launch configuration" chunk.
    # If your output also prints device limits, this still tends to work because
    # those lines are labeled differently.
    return {
        "final_blocks": parse_first_int(RE_FINAL_BLOCKS, output),
        "final_threads": parse_first_int(RE_FINAL_THREADS, output),
    }


def parse_first_int(regex, text):
    m = regex.search(text)
    return int(m.group(1)) if m else None


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "requested_blocks", "requested_threads",
        "final_blocks", "final_threads",
        "returncode",
        "gpu_baseline_ns", "gpu_branched_ns",
        "cpu_baseline_ns", "cpu_branched_ns",
        "log_file",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def heatmap(metric_key: str, rows: list[dict], blocks: list[int], threads: list[int], out_png: Path) -> None:
    # Build matrix [len(blocks) x len(threads)]
    mat = [[None for _ in threads] for _ in blocks]
    lookup = {(r["requested_blocks"], r["requested_threads"]): r for r in rows}

    for bi, b in enumerate(blocks):
        for ti, t in enumerate(threads):
            r = lookup.get((b, t))
            mat[bi][ti] = r.get(metric_key) if r else None

    # Convert None -> NaN for plotting
    import math
    plot_mat = [[(v if v is not None else float("nan")) for v in row] for row in mat]

    plt.figure()
    plt.imshow(plot_mat, aspect="auto")  # default colormap is fine
    plt.colorbar(label=f"{metric_key} (ns)")
    plt.xticks(range(len(threads)), threads, rotation=45, ha="right")
    plt.yticks(range(len(blocks)), blocks)
    plt.xlabel("Threads per block (requested)")
    plt.ylabel("Blocks (requested)")
    plt.title(metric_key)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./assignment.exe", help="Path to assignment.exe")
    ap.add_argument("--out", default="sweep_out", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.out)
    logs_dir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(BLOCKS) * len(THREADS)
    print(f"Running {total} configurations...")
    print(f"Blocks:  {BLOCKS}")
    print(f"Threads: {THREADS}")

    for b in BLOCKS:
        for t in THREADS:
            print(f"  -> blocks={b}, threads={t}")
            rc, out = run_once(args.exe, b, t)

            log_file = logs_dir / f"run_blocks{b}_threads{t}.txt"
            log_file.write_text(out, encoding="utf-8")

            timings = parse_timings(out)
            final_cfg = parse_final_config(out)

            rows.append({
                "requested_blocks": b,
                "requested_threads": t,
                "final_blocks": final_cfg["final_blocks"],
                "final_threads": final_cfg["final_threads"],
                "returncode": rc,
                **timings,
                "log_file": str(log_file),
            })

    csv_path = outdir / "sweep_results.csv"
    write_csv(rows, csv_path)
    print(f"Wrote CSV: {csv_path}")

    # Generate heatmaps
    heatmap("gpu_baseline_ns", rows, BLOCKS, THREADS, outdir / "heat_gpu_baseline.png")
    heatmap("gpu_branched_ns", rows, BLOCKS, THREADS, outdir / "heat_gpu_branched.png")
    heatmap("cpu_baseline_ns", rows, BLOCKS, THREADS, outdir / "heat_cpu_baseline.png")
    heatmap("cpu_branched_ns", rows, BLOCKS, THREADS, outdir / "heat_cpu_branched.png")
    print(f"Wrote heatmaps to: {outdir}")

if __name__ == "__main__":
    main()
