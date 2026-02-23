# overall_kernel_vs_total_pareto.py
#
# Generates: output_plots/overall_kernel_vs_total_pareto.png
# Reads:      a CSV produced by your grid runner (expects columns like kernel_ms, total_ms, bx, by, blur, passes, mode)

import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Found: {list(df.columns)}")


def make_label(row: pd.Series, bx_col: str, by_col: str, blur_col: str, passes_col: str) -> str:
    bx = int(row[bx_col])
    by = int(row[by_col])
    blur = int(row[blur_col])
    passes = int(row[passes_col]) if blur else 0
    return f"{bx}x{by} tpb={bx*by} " + ("blur p=" + str(passes) if blur else "no-blur")


def pareto_front(points):
    """
    Given list of (x, y, idx), return indices on Pareto frontier for minimizing both x and y.
    """
    pts = sorted(points, key=lambda t: (t[0], t[1]))
    front = []
    best_y = float("inf")
    for x, y, idx in pts:
        if y < best_y:
            front.append(idx)
            best_y = y
    return set(front)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to all_runs.csv (from your grid script)")
    ap.add_argument("--out", default="output_plots/overall_kernel_vs_total_pareto.png", help="Output PNG path")
    ap.add_argument("--annotate-top", type=int, default=6, help="Annotate this many best-by-total points")
    ap.add_argument("--mode", default="bw", help="Filter mode (default: bw)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Common column variants your runners might use
    kernel_col = pick_col(df, ["kernel_ms", "kernel_ms_mean", "kernel"])
    total_col  = pick_col(df, ["total_ms", "total_ms_mean", "total"])
    htod_col   = pick_col(df, ["htod_ms", "HtoD_ms", "htoD_ms", "htod"])
    dtoh_col   = pick_col(df, ["dtoh_ms", "DtoH_ms", "dToH_ms", "dtoh"])
    bx_col     = pick_col(df, ["bx", "block_x", "blockx"])
    by_col     = pick_col(df, ["by", "block_y", "blocky"])
    blur_col   = pick_col(df, ["blur", "use_blur"])
    passes_col = pick_col(df, ["passes", "blur_passes", "blurpasses"])
    mode_col   = pick_col(df, ["mode", "filter_mode"])

    # BW-only as requested
    df = df[df[mode_col].astype(str).str.lower() == args.mode.lower()].copy()

    # Sanity: numeric
    for c in [kernel_col, total_col, htod_col, dtoh_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[kernel_col, total_col])

    # Build a readable config label
    df["tpb"] = df[bx_col].astype(int) * df[by_col].astype(int)
    df["label"] = df.apply(lambda r: make_label(r, bx_col, by_col, blur_col, passes_col), axis=1)

    # Identify best overall by end-to-end total_ms
    best_total_idx = df[total_col].idxmin()
    best_total_row = df.loc[best_total_idx]

    # Pareto frontier (min kernel, min total)
    pts = [(float(r[kernel_col]), float(r[total_col]), int(i)) for i, r in df.iterrows()]
    front_idx = pareto_front(pts)

    # Grouping helpers for nicer marker styles
    df["blur_int"] = df[blur_col].astype(int)
    # Split "no-blur" vs "blur"
    no_blur = df[df["blur_int"] == 0]
    blur    = df[df["blur_int"] == 1]

    # Plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Scatter points
    ax.scatter(no_blur[kernel_col], no_blur[total_col], alpha=0.75, label="no-blur")
    ax.scatter(blur[kernel_col], blur[total_col], alpha=0.75, label="blur")

    # Pareto front overlay
    front_df = df[df.index.isin(front_idx)]
    ax.scatter(front_df[kernel_col], front_df[total_col], s=90, marker="D", alpha=0.9, label="Pareto front")

    # Star the best overall (by total)
    ax.scatter([best_total_row[kernel_col]], [best_total_row[total_col]], s=250, marker="*", label="best total")

    # Annotate top-N best by total
    topN = df.nsmallest(args.annotate_top, total_col)
    for _, r in topN.iterrows():
        txt = r["label"]
        ax.annotate(
            txt,
            (r[kernel_col], r[total_col]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )

    # Axes labels/title
    ax.set_title(f"Overall: kernel time vs end-to-end time (mode={args.mode})")
    ax.set_xlabel("kernel_ms")
    ax.set_ylabel("total_ms")

    # Add a small summary box about the best config
    summary = (
        "Best (by total_ms)\n"
        f"{best_total_row['label']}\n"
        f"kernel={best_total_row[kernel_col]:.6g} ms\n"
        f"total={best_total_row[total_col]:.6g} ms\n"
    )
    ax.text(
        0.02, 0.98, summary,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close(fig)

    print(f"[ok] saved: {args.out}")


if __name__ == "__main__":
    main()