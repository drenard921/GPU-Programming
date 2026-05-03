# plot_benchmark_results.py
#
# Generate seaborn benchmark plots for the Rotea CPU vs CUDA benchmark runs.
#
# Inputs:
#   rotea_cpu_cuda_grid_benchmark.csv
#   rotea_cpu_cuda_grid_benchmark_large.csv
#
# Outputs:
#   benchmark_plots/*.png
#   benchmark_plots/benchmark_summary_clean.csv
#   benchmark_plots/benchmark_summary_by_group.csv

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # -----------------------------
# # Configuration
# # -----------------------------
# SMALL_CSV = "rotea_cpu_cuda_grid_benchmark.csv"
# LARGE_CSV = "rotea_cpu_cuda_grid_benchmark_large.csv"

# OUTDIR = Path("benchmark_plots")
# OUTDIR.mkdir(exist_ok=True)




sns.set_theme(style="whitegrid", context="talk")


# # -----------------------------
# # Load and combine data
# # -----------------------------
# def load_benchmark_csv(path: str, batch_name: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["batch"] = batch_name
#     return df


# small = load_benchmark_csv(SMALL_CSV, "small_grid")
# large = load_benchmark_csv(LARGE_CSV, "large_grid")

# df = pd.concat([small, large], ignore_index=True)

# -----------------------------
# Configuration
# -----------------------------
HEADLESS_CSV = "rotea_cpu_cuda_headless_benchmark.csv"

OUTDIR = Path("benchmark_plots_headless")
OUTDIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(HEADLESS_CSV)
df["batch"] = "headless"

# Normalize useful columns
df["particles"] = pd.to_numeric(df["particles"], errors="coerce")
df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
df["reported_frames"] = pd.to_numeric(df["reported_frames"], errors="coerce")
df["average_fps"] = pd.to_numeric(df["average_fps"], errors="coerce")
df["avg_full_frame_ms"] = pd.to_numeric(df["avg_full_frame_ms"], errors="coerce")
df["avg_particle_update_ms"] = pd.to_numeric(df["avg_particle_update_ms"], errors="coerce")
df["avg_render_ms"] = pd.to_numeric(df["avg_render_ms"], errors="coerce")
df["reported_overall_runtime_s"] = pd.to_numeric(
    df["reported_overall_runtime_s"], errors="coerce"
)

# Create clean backend label
# Your files have both backend and mode columns. backend is enough here.
df["backend"] = df["backend"].astype(str).str.upper()

# A valid timing row must have completed successfully and contain full timing metrics.
valid = df[
    (df["return_code"].astype(str) == "0")
    & df["average_fps"].notna()
    & df["avg_full_frame_ms"].notna()
    & df["avg_particle_update_ms"].notna()
    & df["avg_render_ms"].notna()
].copy()

invalid = df[~df.index.isin(valid.index)].copy()

# Save cleaned data
valid.to_csv(OUTDIR / "benchmark_summary_clean.csv", index=False)


# -----------------------------
# Grouped summary
# -----------------------------
summary = (
    valid.groupby(["particles", "step", "backend"], as_index=False)
    .agg(
        mean_fps=("average_fps", "mean"),
        sd_fps=("average_fps", "std"),
        mean_full_frame_ms=("avg_full_frame_ms", "mean"),
        sd_full_frame_ms=("avg_full_frame_ms", "std"),
        mean_particle_update_ms=("avg_particle_update_ms", "mean"),
        sd_particle_update_ms=("avg_particle_update_ms", "std"),
        mean_render_ms=("avg_render_ms", "mean"),
        sd_render_ms=("avg_render_ms", "std"),
        n=("average_fps", "count"),
    )
)

summary.to_csv(OUTDIR / "benchmark_summary_by_group.csv", index=False)

print("\nValid timing rows:")
print(valid.groupby(["particles", "backend"]).size().unstack(fill_value=0))

print("\nInvalid / timeout / failed rows:")
if len(invalid) > 0:
    print(invalid.groupby(["particles", "backend", "return_code"]).size())
else:
    print("None")


# -----------------------------
# Helper plotting functions
# -----------------------------
def save_lineplot(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
):
    plt.figure(figsize=(12, 7))

    ax = sns.lineplot(
        data=data,
        x="particles",
        y=y,
        hue="backend",
        style="backend",
        markers=True,
        dashes=False,
        errorbar="sd",
    )

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Particle Count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Cleaner x-axis labels
    particle_values = sorted(data["particles"].dropna().unique())
    ax.set_xticks(particle_values)
    ax.set_xticklabels([f"{int(x):,}" for x in particle_values], rotation=30)

    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()


def save_step_lineplot(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    filename: str,
):
    g = sns.relplot(
        data=data,
        x="particles",
        y=y,
        hue="backend",
        style="backend",
        col="step",
        kind="line",
        markers=True,
        dashes=False,
        errorbar="sd",
        facet_kws={"sharey": False, "sharex": True},
        height=5,
        aspect=1.1,
    )

    g.set(xscale="log")
    g.set_axis_labels("Particle Count", ylabel)
    g.fig.suptitle(title, y=1.05)

    for ax in g.axes.flat:
        particle_values = sorted(data["particles"].dropna().unique())
        ax.set_xticks(particle_values)
        ax.set_xticklabels([f"{int(x):,}" for x in particle_values], rotation=45)

    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main plots: average across steps/trials
# -----------------------------
save_lineplot(
    data=valid,
    y="average_fps",
    ylabel="Average FPS",
    title="Rotea Simulator FPS: CPU vs CUDA",
    filename="fps_vs_particles.png",
)

save_lineplot(
    data=valid,
    y="avg_full_frame_ms",
    ylabel="Average Full Frame Runtime (ms/frame)",
    title="Full Frame Runtime: CPU vs CUDA",
    filename="full_frame_runtime_vs_particles.png",
)

save_lineplot(
    data=valid,
    y="avg_particle_update_ms",
    ylabel="Average Particle Update Runtime (ms/frame)",
    title="Particle Update Runtime: CPU vs CUDA",
    filename="particle_update_runtime_vs_particles.png",
)

save_lineplot(
    data=valid,
    y="avg_render_ms",
    ylabel="Average Render / Swap / Poll Runtime (ms/frame)",
    title="Rendering Runtime: CPU vs CUDA",
    filename="render_runtime_vs_particles.png",
)


# -----------------------------
# Optional step-specific plots
# -----------------------------
save_step_lineplot(
    data=valid,
    y="average_fps",
    ylabel="Average FPS",
    title="FPS by Protocol Step",
    filename="fps_by_step.png",
)

save_step_lineplot(
    data=valid,
    y="avg_particle_update_ms",
    ylabel="Particle Update Runtime (ms/frame)",
    title="Particle Update Runtime by Protocol Step",
    filename="particle_update_runtime_by_step.png",
)


# -----------------------------
# Timeout/failure summary plot
# -----------------------------
df["valid_timing"] = df.index.isin(valid.index)

outcome = (
    df.groupby(["particles", "backend", "valid_timing"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
)

outcome["outcome"] = np.where(outcome["valid_timing"], "Valid timing", "Invalid / timeout / failed")

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=outcome,
    x="particles",
    y="count",
    hue="outcome",
)

ax.set_xlabel("Particle Count")
ax.set_ylabel("Number of Trials")
ax.set_title("Benchmark Trial Outcomes")
ax.set_xticklabels([f"{int(float(t.get_text())):,}" for t in ax.get_xticklabels()], rotation=30)

plt.tight_layout()
plt.savefig(OUTDIR / "timeout_summary.png", dpi=300)
plt.close()


# -----------------------------
# CUDA vs CPU percent difference summary
# -----------------------------
overall = (
    valid.groupby(["particles", "backend"], as_index=False)
    .agg(
        mean_fps=("average_fps", "mean"),
        mean_full_frame_ms=("avg_full_frame_ms", "mean"),
        mean_particle_update_ms=("avg_particle_update_ms", "mean"),
        mean_render_ms=("avg_render_ms", "mean"),
    )
)

pivot = overall.pivot(index="particles", columns="backend")

comparison = pd.DataFrame(index=pivot.index)

if "CPU" in pivot["mean_fps"].columns and "CUDA" in pivot["mean_fps"].columns:
    comparison["cpu_fps"] = pivot["mean_fps"]["CPU"]
    comparison["cuda_fps"] = pivot["mean_fps"]["CUDA"]
    comparison["cuda_fps_percent_change_vs_cpu"] = (
        (comparison["cuda_fps"] - comparison["cpu_fps"])
        / comparison["cpu_fps"]
        * 100
    )

    comparison["cpu_particle_update_ms"] = pivot["mean_particle_update_ms"]["CPU"]
    comparison["cuda_particle_update_ms"] = pivot["mean_particle_update_ms"]["CUDA"]
    comparison["cuda_particle_update_percent_change_vs_cpu"] = (
        (comparison["cuda_particle_update_ms"] - comparison["cpu_particle_update_ms"])
        / comparison["cpu_particle_update_ms"]
        * 100
    )

comparison = comparison.reset_index()
comparison.to_csv(OUTDIR / "cuda_vs_cpu_percent_difference.csv", index=False)

print("\nCUDA vs CPU percent difference:")
print(comparison.round(3))

print(f"\nDone. Plots and summaries saved to: {OUTDIR.resolve()}")