#!/usr/bin/env python3
"""
runner.py

Benchmark runner for the CUDA Streams and Events assignment.

This script launches the CUDA program across multiple thread/block
configurations, collects the per-frame CSV output from each run, and writes
a summarized CSV that can be used in the README or final report.

Example:
    python3 runner.py --frames 300

Optional:
    python3 runner.py --frames 300 --scale 3 \
        --threads 128 256 512 768 1024 \
        --blocks 0 60 120 240
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA streams benchmark sweeps.")
    parser.add_argument(
        "--exe",
        default="./streams_hw_cuda",
        help="Path to compiled executable",
    )
    parser.add_argument(
        "--outdir",
        default="runner_results",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Frames per run",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Scale factor passed to the CUDA program",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[128, 256, 512, 768, 1024],
        help="Thread counts to test",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[0, 60, 120, 240],
        help="Block counts to test (0 means auto)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per configuration",
    )
    parser.add_argument(
        "--core",
        default=None,
        help="Optional libretro core path",
    )
    parser.add_argument(
        "--fr",
        default=None,
        help="Optional FireRed ROM path",
    )
    parser.add_argument(
        "--lg",
        default=None,
        help="Optional LeafGreen ROM path",
    )
    return parser.parse_args()


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Executable not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Executable path is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"File exists but is not executable: {path}")


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def read_metrics_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {
            "frame",
            "threads",
            "blocks",
            "fr_up_ms",
            "lg_up_ms",
            "self_diff_ms",
            "cross_diff_ms",
            "total_gpu_ms",
            "heat_self_mean",
            "heat_cross_mean",
            "nonzero_diff_pixels",
        }

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing = expected.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing expected columns {sorted(missing)}: {csv_path}")

        for row in reader:
            rows.append(
                {
                    "frame": float(row["frame"]),
                    "threads": float(row["threads"]),
                    "blocks": float(row["blocks"]),
                    "fr_up_ms": float(row["fr_up_ms"]),
                    "lg_up_ms": float(row["lg_up_ms"]),
                    "self_diff_ms": float(row["self_diff_ms"]),
                    "cross_diff_ms": float(row["cross_diff_ms"]),
                    "total_gpu_ms": float(row["total_gpu_ms"]),
                    "heat_self_mean": float(row["heat_self_mean"]),
                    "heat_cross_mean": float(row["heat_cross_mean"]),
                    "nonzero_diff_pixels": float(row["nonzero_diff_pixels"]),
                }
            )

    return rows


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    total_gpu = [r["total_gpu_ms"] for r in rows]
    fr_up = [r["fr_up_ms"] for r in rows]
    lg_up = [r["lg_up_ms"] for r in rows]
    self_diff = [r["self_diff_ms"] for r in rows]
    cross_diff = [r["cross_diff_ms"] for r in rows]
    heat_self = [r["heat_self_mean"] for r in rows]
    heat_cross = [r["heat_cross_mean"] for r in rows]
    nz = [r["nonzero_diff_pixels"] for r in rows]

    mean_total_gpu = safe_mean(total_gpu)
    fps_est = 1000.0 / mean_total_gpu if mean_total_gpu > 0.0 else float("nan")

    return {
        "frames_collected": len(rows),
        "fr_up_ms_mean": safe_mean(fr_up),
        "fr_up_ms_std": safe_stdev(fr_up),
        "lg_up_ms_mean": safe_mean(lg_up),
        "lg_up_ms_std": safe_stdev(lg_up),
        "self_diff_ms_mean": safe_mean(self_diff),
        "self_diff_ms_std": safe_stdev(self_diff),
        "cross_diff_ms_mean": safe_mean(cross_diff),
        "cross_diff_ms_std": safe_stdev(cross_diff),
        "total_gpu_ms_mean": mean_total_gpu,
        "total_gpu_ms_std": safe_stdev(total_gpu),
        "fps_estimate": fps_est,
        "heat_self_mean_avg": safe_mean(heat_self),
        "heat_cross_mean_avg": safe_mean(heat_cross),
        "nonzero_diff_pixels_avg": safe_mean(nz),
    }


def run_one_config(
    exe: Path,
    outdir: Path,
    frames: int,
    scale: int,
    threads: int,
    blocks: int,
    repeat_idx: int,
    core: str | None,
    fr: str | None,
    lg: str | None,
) -> tuple[int, Path, Path]:
    run_name = f"t{threads}_b{blocks}_r{repeat_idx}"
    run_dir = outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "metrics.csv"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    cmd = [
        str(exe),
        "--frames",
        str(frames),
        "--scale",
        str(scale),
        "--threads",
        str(threads),
        "--blocks",
        str(blocks),
        "--csv",
        str(csv_path),
        "--print-every",
        str(max(1, frames // 5)),
    ]

    if core:
        cmd.extend(["--core", core])
    if fr:
        cmd.extend(["--fr", fr])
    if lg:
        cmd.extend(["--lg", lg])

    print(f"[RUN] threads={threads:>4} blocks={blocks:>4} repeat={repeat_idx}")

    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        proc = subprocess.run(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        print(f"[FAIL] {run_name} exited with code {proc.returncode}")

    return proc.returncode, csv_path, run_dir


def write_summary_csv(summary_csv: Path, summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        return

    fieldnames = [
        "threads",
        "blocks",
        "repeat",
        "frames_collected",
        "fr_up_ms_mean",
        "fr_up_ms_std",
        "lg_up_ms_mean",
        "lg_up_ms_std",
        "self_diff_ms_mean",
        "self_diff_ms_std",
        "cross_diff_ms_mean",
        "cross_diff_ms_std",
        "total_gpu_ms_mean",
        "total_gpu_ms_std",
        "fps_estimate",
        "heat_self_mean_avg",
        "heat_cross_mean_avg",
        "nonzero_diff_pixels_avg",
        "run_dir",
    ]

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    exe = Path(args.exe).resolve()
    outdir = Path(args.outdir).resolve()

    ensure_executable(exe)

    if outdir.exists():
        print(f"[INFO] Output directory exists: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failures: list[str] = []

    configs = list(itertools.product(args.threads, args.blocks, range(1, args.repeats + 1)))
    print(f"[INFO] Total runs: {len(configs)}")

    for threads, blocks, repeat_idx in configs:
        returncode, csv_path, run_dir = run_one_config(
            exe=exe,
            outdir=outdir,
            frames=args.frames,
            scale=args.scale,
            threads=threads,
            blocks=blocks,
            repeat_idx=repeat_idx,
            core=args.core,
            fr=args.fr,
            lg=args.lg,
        )

        if returncode != 0:
            failures.append(f"{run_dir.name} (exit={returncode})")
            print(f"[WARN] Run failed: {run_dir.name}")
            continue

        if not csv_path.exists():
            failures.append(f"{run_dir.name} (missing csv)")
            print(f"[WARN] Missing metrics CSV: {run_dir.name}")
            continue

        try:
            rows = read_metrics_csv(csv_path)
        except Exception as e:
            failures.append(f"{run_dir.name} (csv parse error: {e})")
            print(f"[WARN] Failed to parse CSV for {run_dir.name}: {e}")
            continue

        if not rows:
            failures.append(f"{run_dir.name} (empty csv)")
            print(f"[WARN] Empty metrics CSV: {run_dir.name}")
            continue

        summary = summarize_rows(rows)
        summary_rows.append(
            {
                "threads": threads,
                "blocks": blocks,
                "repeat": repeat_idx,
                **summary,
                "run_dir": str(run_dir),
            }
        )

    summary_csv = outdir / "summary.csv"
    write_summary_csv(summary_csv, summary_rows)

    print(f"\n[INFO] Summary written to: {summary_csv}")
    print(f"[INFO] Successful runs: {len(summary_rows)}")
    print(f"[INFO] Failed runs: {len(failures)}")

    if failures:
        fail_log = outdir / "failures.txt"
        fail_log.write_text("\n".join(failures) + "\n")
        print(f"[INFO] Failure log written to: {fail_log}")

    if summary_rows:
        best = min(summary_rows, key=lambda r: float(r["total_gpu_ms_mean"]))
        print("\n[BEST CONFIG]")
        print(
            f"threads={best['threads']} blocks={best['blocks']} repeat={best['repeat']} "
            f"mean_total_gpu_ms={float(best['total_gpu_ms_mean']):.4f} "
            f"fps_estimate={float(best['fps_estimate']):.2f}"
        )

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())