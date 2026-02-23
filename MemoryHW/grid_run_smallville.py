#!/usr/bin/env python3
import argparse
import csv
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

RESULT_RE = re.compile(r"^RESULT\s+(.*)$", re.MULTILINE)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_cmd_capture(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    """
    Runs cmd and returns (exit_code, combined_stdout_stderr).
    Uses text mode; works on Windows/Linux.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, out

def parse_result_kv(output: str) -> Optional[Dict[str, str]]:
    """
    Parse the single-line RESULT key=val output from cuda_filter.
    Returns dict or None if not found.
    """
    m = RESULT_RE.search(output)
    if not m:
        return None
    tail = m.group(1).strip()
    kv = {}
    for token in tail.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

def to_float(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def to_int(x: str, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Data model
# ----------------------------

@dataclass
class RunRecord:
    experiment: str
    run_id: int
    input_image: str
    output_image: str

    # config
    mode: str
    intensity: float
    bx: int
    by: int
    blur: int
    passes: int

    # derived
    tpb: int

    # result from program
    w: int
    h: int
    gridx: int
    gridy: int

    htod_ms: float
    kernel_ms: float
    dtoh_ms: float
    total_ms: float

    alloc_bytes: int
    free0: int
    free1: int
    total_mem: int

    # extras
    exit_code: int

    # computed throughput metrics (simple model)
    image_bytes: int
    effective_bytes: int
    kernel_GBs: float
    total_GBs: float


def compute_effective_bytes(image_bytes: int, blur: int, passes: int) -> int:
    """
    Simple, consistent accounting:
      - LUT stage reads + writes one RGB image: 2 * image_bytes
      - Each blur pass reads + writes one RGB image: 2 * image_bytes per pass
    This isn't "true DRAM traffic" but it’s a consistent effective throughput model.
    """
    eff = 2 * image_bytes
    if blur:
        eff += 2 * image_bytes * max(1, passes)
    return eff


# ----------------------------
# Plotting (consistent + readable)
# ----------------------------

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

def save_line_plot(x, y, xlabel, ylabel, title, outpath: Path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_bar_plot(x_labels, y, xlabel, ylabel, title, outpath: Path):
    plt.figure()
    plt.bar(range(len(x_labels)), y)
    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_stacked_barh(labels, htod, kernel, dtoh, title, outpath: Path):
    """
    Horizontal stacked bars: much more readable for timing breakdown.
    """
    plt.figure(figsize=(10, 6))
    y = list(range(len(labels)))

    left1 = [0.0] * len(labels)
    plt.barh(y, htod, left=left1, label="HtoD")

    left2 = [htod[i] for i in range(len(labels))]
    plt.barh(y, kernel, left=left2, label="Kernel")

    left3 = [htod[i] + kernel[i] for i in range(len(labels))]
    plt.barh(y, dtoh, left=left3, label="DtoH")

    plt.yticks(y, labels)
    plt.xlabel("ms")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ----------------------------
# Experiment runner
# ----------------------------

def build_cmd(exe: Path, input_img: Path, output_img: Path,
              bx: int, by: int, blur: int, passes: int,
              intensity: float = 1.0) -> List[str]:
    """
    BW-only runs.
    """
    cmd = [
        str(exe),
        "--in", str(input_img),
        "--out", str(output_img),
        "--mode", "bw",
        "--intensity", str(float(intensity)),
        "--bx", str(int(bx)),
        "--by", str(int(by)),
        "--blur", str(int(blur)),
    ]
    if blur:
        cmd += ["--blur-passes", str(int(passes))]
    return cmd

def run_one(experiment: str, run_id: int,
            exe: Path, repo_root: Path,
            input_img: Path, out_dir_images: Path,
            bx: int, by: int, blur: int, passes: int,
            intensity: float = 1.0) -> RunRecord:

    out_name = f"{input_img.stem}_{experiment}_r{run_id:02d}_bw_bx{bx}_by{by}_blur{blur}_p{passes}{input_img.suffix}"
    out_img = out_dir_images / out_name

    cmd = build_cmd(exe, input_img, out_img, bx, by, blur, passes, intensity=intensity)
    code, out = run_cmd_capture(cmd, cwd=repo_root)

    kv = parse_result_kv(out)
    if kv is None:
        # Provide a helpful error with a common gotcha
        hint = ""
        if platform.system().lower().startswith("win") and exe.suffix == "":
            hint = (
                "\n\n[hint] On Windows, WinError 193 usually means you're trying to run a Linux/WSL binary.\n"
                "       If your cuda_filter was built in WSL, run this script inside WSL OR point --exe to a Windows .exe.\n"
            )
        raise RuntimeError(
            f"Failed to parse RESULT line for {experiment} run {run_id} (exit={code}).\n"
            f"Command:\n  {' '.join(cmd)}\n\nOutput:\n{out}{hint}"
        )

    w = to_int(kv.get("w", "-1"))
    h = to_int(kv.get("h", "-1"))
    image_bytes = w * h * 3 if (w > 0 and h > 0) else 0
    eff_bytes = compute_effective_bytes(image_bytes, blur=blur, passes=passes)

    htod_ms = to_float(kv.get("htod_ms", "nan"))
    kernel_ms = to_float(kv.get("kernel_ms", "nan"))
    dtoh_ms = to_float(kv.get("dtoh_ms", "nan"))
    total_ms = to_float(kv.get("total_ms", "nan"))

    # Effective throughput (GB/s) using decimal GB for readability
    # (bytes / (ms/1000)) / 1e9
    kernel_GBs = (eff_bytes / (kernel_ms / 1000.0) / 1e9) if (kernel_ms > 0 and eff_bytes > 0) else float("nan")
    total_GBs  = (eff_bytes / (total_ms / 1000.0) / 1e9) if (total_ms > 0 and eff_bytes > 0) else float("nan")

    rec = RunRecord(
        experiment=experiment,
        run_id=run_id,
        input_image=str(input_img),
        output_image=str(out_img),
        mode="bw",
        intensity=float(intensity),
        bx=int(bx),
        by=int(by),
        blur=int(blur),
        passes=int(passes),
        tpb=int(bx) * int(by),

        w=w,
        h=h,
        gridx=to_int(kv.get("gridx", "-1")),
        gridy=to_int(kv.get("gridy", "-1")),

        htod_ms=htod_ms,
        kernel_ms=kernel_ms,
        dtoh_ms=dtoh_ms,
        total_ms=total_ms,

        alloc_bytes=to_int(kv.get("alloc_bytes", "-1")),
        free0=to_int(kv.get("free0", "-1")),
        free1=to_int(kv.get("free1", "-1")),
        total_mem=to_int(kv.get("total_mem", "-1")),

        exit_code=int(code),

        image_bytes=int(image_bytes),
        effective_bytes=int(eff_bytes),
        kernel_GBs=float(kernel_GBs),
        total_GBs=float(total_GBs),
    )
    return rec


def write_csv(path: Path, rows: List[RunRecord]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def summarize_and_plot(experiment: str, rows: List[RunRecord], out_plots: Path) -> None:
    """
    Make a small, consistent set of plots per experiment.
    """
    ensure_dir(out_plots)

    # Sort by the "sweep parameter" depending on experiment
    if "bx_sweep" in experiment:
        rows_sorted = sorted(rows, key=lambda r: r.bx)
        x = [r.bx for r in rows_sorted]
        xlab = "bx"
    elif "by_sweep" in experiment:
        rows_sorted = sorted(rows, key=lambda r: r.by)
        x = [r.by for r in rows_sorted]
        xlab = "by"
    elif "passes_sweep" in experiment:
        rows_sorted = sorted(rows, key=lambda r: r.passes)
        x = [r.passes for r in rows_sorted]
        xlab = "passes"
    else:
        rows_sorted = rows
        x = [r.run_id for r in rows_sorted]
        xlab = "run"

    kernel = [r.kernel_ms for r in rows_sorted]
    total  = [r.total_ms for r in rows_sorted]
    htod   = [r.htod_ms for r in rows_sorted]
    dtoh   = [r.dtoh_ms for r in rows_sorted]

    # Simple line plots
    save_line_plot(
        x, kernel,
        xlabel=xlab,
        ylabel="kernel_ms",
        title=f"{experiment}: kernel time vs {xlab}",
        outpath=out_plots / f"{experiment}_kernel_vs_{xlab}.png",
    )
    save_line_plot(
        x, total,
        xlabel=xlab,
        ylabel="total_ms",
        title=f"{experiment}: total time vs {xlab}",
        outpath=out_plots / f"{experiment}_total_vs_{xlab}.png",
    )

    # Throughput lines (if available)
    kernel_gbs = [r.kernel_GBs for r in rows_sorted]
    total_gbs  = [r.total_GBs for r in rows_sorted]
    save_line_plot(
        x, kernel_gbs,
        xlabel=xlab,
        ylabel="effective GB/s",
        title=f"{experiment}: effective kernel throughput vs {xlab}",
        outpath=out_plots / f"{experiment}_throughput_kernel_vs_{xlab}.png",
    )
    save_line_plot(
        x, total_gbs,
        xlabel=xlab,
        ylabel="effective GB/s",
        title=f"{experiment}: effective end-to-end throughput vs {xlab}",
        outpath=out_plots / f"{experiment}_throughput_total_vs_{xlab}.png",
    )

    # Timing breakdown as stacked horizontal bars (top-to-bottom)
    labels = [f"bx{r.bx} by{r.by} blur{r.blur} p{r.passes}" for r in rows_sorted]
    save_stacked_barh(
        labels=labels,
        htod=htod,
        kernel=kernel,
        dtoh=dtoh,
        title=f"{experiment}: timing breakdown (HtoD/Kernel/DtoH)",
        outpath=out_plots / f"{experiment}_timing_breakdown_barh.png",
    )


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="5 experiments × 8 runs BW-only grid search for cuda_filter.")
    parser.add_argument("--repo", type=str, default=".", help="Repo root (where cuda_filter lives).")
    parser.add_argument("--exe", type=str, default="", help="Path to cuda_filter executable (default: <repo>/cuda_filter or cuda_filter.exe).")
    parser.add_argument("--img", type=str, required=True, help="Input image path (e.g., InputImages/Smallville.png).")
    parser.add_argument("--out-plots", type=str, default="output_plots", help="Directory to save plots.")
    parser.add_argument("--out-csv", type=str, default="", help="Optional path for consolidated CSV (default: output_plots/<stamp>_all_runs.csv).")
    parser.add_argument("--by-fixed", type=int, default=16, help="Fixed by for bx sweeps.")
    parser.add_argument("--bx-fixed", type=int, default=16, help="Fixed bx for by sweeps.")
    parser.add_argument("--max-passes", type=int, default=8, help="Max blur passes for passes sweeps and max-pass sweeps.")
    args = parser.parse_args()

    set_plot_style()

    repo_root = Path(args.repo).resolve()
    input_img = Path(args.img).resolve()
    out_plots = (repo_root / args.out_plots).resolve()
    out_dir_images = (repo_root / "OutputImages").resolve()

    ensure_dir(out_plots)
    ensure_dir(out_dir_images)

    # Resolve executable
    exe = None
    if args.exe:
        exe = Path(args.exe).resolve()
    else:
        # Try repo_root/cuda_filter(.exe)
        cand1 = repo_root / "cuda_filter"
        cand2 = repo_root / "cuda_filter.exe"
        exe = cand1 if cand1.exists() else cand2

    if not exe.exists():
        raise FileNotFoundError(f"cuda_filter not found at: {exe}. Use --exe to point to it.")

    if not input_img.exists():
        raise FileNotFoundError(f"Input image not found: {input_img}")

    # 8 values for bx and by sweeps (kept within blur cap of 32 for blur runs)
    bx_vals = [4, 8, 12, 16, 20, 24, 28, 32]
    by_vals = [4, 8, 12, 16, 20, 24, 28, 32]
    passes_vals = list(range(1, 9))
    max_passes = int(args.max_passes)

    all_rows: List[RunRecord] = []
    per_exp: Dict[str, List[RunRecord]] = {}

    def do_experiment(exp_name: str, configs: List[Tuple[int, int, int, int]]):
        """
        configs: list of (bx, by, blur, passes)
        """
        rows: List[RunRecord] = []
        print(f"\n[info] Experiment: {exp_name}  ({len(configs)} runs)")
        for i, (bx, by, blur, passes) in enumerate(configs, start=1):
            print(f"[run {i}/{len(configs)}] bx={bx} by={by} blur={blur} passes={passes}")
            rec = run_one(
                experiment=exp_name,
                run_id=i,
                exe=exe,
                repo_root=repo_root,
                input_img=input_img,
                out_dir_images=out_dir_images,
                bx=bx,
                by=by,
                blur=blur,
                passes=passes,
                intensity=1.0,
            )
            rows.append(rec)

        per_exp[exp_name] = rows
        all_rows.extend(rows)

        # Write per-experiment CSV + plots
        exp_csv = out_plots / f"{exp_name}_runs.csv"
        write_csv(exp_csv, rows)
        summarize_and_plot(exp_name, rows, out_plots)

    # 1) Increase bx, hold everything else constant (blur=0)
    exp1 = "A_bx_sweep_blur0"
    configs1 = [(bx, args.by_fixed, 0, 0) for bx in bx_vals]
    do_experiment(exp1, configs1)

    # 2) Increase by, hold everything else constant (blur=0)
    exp2 = "B_by_sweep_blur0"
    configs2 = [(args.bx_fixed, by, 0, 0) for by in by_vals]
    do_experiment(exp2, configs2)

    # 3) Increase passes, hold everything else constant (blur=1)
    exp3 = "C_passes_sweep_blur1"
    configs3 = [(args.bx_fixed, args.by_fixed, 1, p) for p in passes_vals]
    do_experiment(exp3, configs3)

    # 4) With max passes, increase bx (blur=1)
    exp4 = f"D_bx_sweep_blur1_p{max_passes}"
    configs4 = [(bx, args.by_fixed, 1, max_passes) for bx in bx_vals]
    do_experiment(exp4, configs4)

    # 5) With max passes, increase by (blur=1)
    exp5 = f"E_by_sweep_blur1_p{max_passes}"
    configs5 = [(args.bx_fixed, by, 1, max_passes) for by in by_vals]
    do_experiment(exp5, configs5)

    # Consolidated outputs
    stamp = now_stamp()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (out_plots / f"{stamp}_all_runs.csv")
    write_csv(out_csv, all_rows)

    # Print a quick “best” table
    best_kernel = sorted(all_rows, key=lambda r: r.kernel_ms)[:10]
    best_total  = sorted(all_rows, key=lambda r: r.total_ms)[:10]

    def fmt(r: RunRecord) -> str:
        return (f"{r.experiment} r{r.run_id:02d}  bx={r.bx:2d} by={r.by:2d} "
                f"blur={r.blur} p={r.passes}  kernel_ms={r.kernel_ms:.6f} total_ms={r.total_ms:.6f}")

    print("\n=== Top 10 by kernel_ms (lower is better) ===")
    for r in best_kernel:
        print(fmt(r))

    print("\n=== Top 10 by total_ms (lower is better) ===")
    for r in best_total:
        print(fmt(r))

    print(f"\n[done] Wrote consolidated CSV: {out_csv}")
    print(f"[done] Plots + per-experiment CSVs in: {out_plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())