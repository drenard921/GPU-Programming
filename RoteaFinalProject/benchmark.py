#!/usr/bin/env python3

import csv
import os
import re
import subprocess
import time
from pathlib import Path

PARTICLE_COUNTS = [100000, 200000, 500000, 1000000]
STEPS = ["load", "concentrate", "wash", "harvest"]
MODES = [
    {
        "backend": "cpu",
        "binary": "./bin/rotea_sim",
        "mode": "cpu",
    },
    {
        "backend": "cuda",
        "binary": "./bin/rotea_sim_cuda",
        "mode": "cuda",
    },
]

TRIALS = 3
DURATION_SECONDS = 30
OUTPUT_CSV = "rotea_cpu_cuda_grid_benchmark_large.csv"

# WSL CUDA library path fix
NVIDIA_WSL_DRIVER_DIR = "/usr/lib/wsl/drivers/nvcvi.inf_amd64_cea94aa1dc79fd8a"


def build_env():
    env = os.environ.copy()

    cuda_paths = [
        NVIDIA_WSL_DRIVER_DIR,
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.1/lib64",
    ]

    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        cuda_paths.append(existing)

    env["NVIDIA_WSL_DRIVER_DIR"] = NVIDIA_WSL_DRIVER_DIR
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths)

    return env


def parse_metric(pattern, text, default=None):
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def parse_summary(output):
    return {
        "average_fps": parse_metric(r"Average FPS:\s+([0-9.]+)", output),
        "avg_full_frame_ms": parse_metric(r"Average full frame runtime:\s+([0-9.]+)", output),
        "avg_system_update_ms": parse_metric(r"Average system update runtime:\s+([0-9.]+)", output),
        "avg_particle_update_ms": parse_metric(
            r"Average (?:CPU|CUDA) particle update runtime:\s+([0-9.]+)", output
        ),
        "total_particle_update_s": parse_metric(
            r"Total (?:CPU|CUDA) particle update runtime:\s+([0-9.]+)", output
        ),
        "avg_render_ms": parse_metric(r"Average render/swap/poll runtime:\s+([0-9.]+)", output),
        "reported_overall_runtime_s": parse_metric(r"Overall runtime:\s+([0-9.]+)", output),
        "reported_frames": parse_metric(r"Frames:\s+([0-9.]+)", output),
    }


def run_one(binary, mode, backend, step, particles, trial, env):
    cmd = [
        binary,
        "--mode", mode,
        "--step", step,
        "--particles", str(particles),
        "--duration", str(DURATION_SECONDS),
    ]

    print(
        f"[RUN] backend={backend} step={step} particles={particles} "
        f"trial={trial}/{TRIALS}"
    )
    print("      " + " ".join(cmd))

    wall_start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=DURATION_SECONDS + 90,
            check=False,
        )
        wall_end = time.perf_counter()

        combined_output = result.stdout + "\n" + result.stderr
        metrics = parse_summary(combined_output)

        return {
            "backend": backend,
            "mode": mode,
            "step": step,
            "particles": particles,
            "trial": trial,
            "duration_requested_s": DURATION_SECONDS,
            "return_code": result.returncode,
            "wall_runtime_s": wall_end - wall_start,
            **metrics,
            "command": " ".join(cmd),
            "stdout_tail": result.stdout[-1000:].replace("\n", "\\n"),
            "stderr_tail": result.stderr[-1000:].replace("\n", "\\n"),
        }

    except subprocess.TimeoutExpired as exc:
        wall_end = time.perf_counter()
        return {
            "backend": backend,
            "mode": mode,
            "step": step,
            "particles": particles,
            "trial": trial,
            "duration_requested_s": DURATION_SECONDS,
            "return_code": "TIMEOUT",
            "wall_runtime_s": wall_end - wall_start,
            "average_fps": None,
            "avg_full_frame_ms": None,
            "avg_system_update_ms": None,
            "avg_particle_update_ms": None,
            "total_particle_update_s": None,
            "avg_render_ms": None,
            "reported_overall_runtime_s": None,
            "reported_frames": None,
            "command": " ".join(cmd),
            "stdout_tail": str(exc.stdout)[-1000:].replace("\n", "\\n") if exc.stdout else "",
            "stderr_tail": str(exc.stderr)[-1000:].replace("\n", "\\n") if exc.stderr else "",
        }


def main():
    env = build_env()

    missing = []
    for cfg in MODES:
        if not Path(cfg["binary"]).exists():
            missing.append(cfg["binary"])

    if missing:
        raise FileNotFoundError(
            "Missing binaries: "
            + ", ".join(missing)
            + "\nRun: make clean && make app && make app-cuda"
        )

    fieldnames = [
        "backend",
        "mode",
        "step",
        "particles",
        "trial",
        "duration_requested_s",
        "return_code",
        "wall_runtime_s",
        "reported_overall_runtime_s",
        "reported_frames",
        "average_fps",
        "avg_full_frame_ms",
        "avg_system_update_ms",
        "avg_particle_update_ms",
        "total_particle_update_s",
        "avg_render_ms",
        "command",
        "stdout_tail",
        "stderr_tail",
    ]

    rows = []

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for particles in PARTICLE_COUNTS:
            for step in STEPS:
                for cfg in MODES:
                    for trial in range(1, TRIALS + 1):
                        row = run_one(
                            binary=cfg["binary"],
                            mode=cfg["mode"],
                            backend=cfg["backend"],
                            step=step,
                            particles=particles,
                            trial=trial,
                            env=env,
                        )

                        rows.append(row)
                        writer.writerow(row)
                        f.flush()

                        if row["return_code"] != 0:
                            print(
                                f"[WARN] Nonzero return code for "
                                f"{row['backend']} {row['step']} "
                                f"{row['particles']} trial {row['trial']}: "
                                f"{row['return_code']}"
                            )

    print(f"\nDone. Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()