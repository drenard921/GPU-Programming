#!/usr/bin/env python3

import csv
import os
import re
import subprocess
import time
from pathlib import Path

# Headless benchmark grid.
# These should be safe because rendering is disabled.
PARTICLE_COUNTS = [1000, 2000, 5000, 10000, 50000, 100000, 200000]
STEPS = ["protocol", "load", "concentrate", "wash", "harvest"]

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
UPDATE_STEPS = 10000
OUTPUT_CSV = "rotea_cpu_cuda_headless_benchmark.csv"

# WSL CUDA library path fix
NVIDIA_WSL_DRIVER_DIR = "/usr/lib/wsl/drivers/nvcvi.inf_amd64_cea94aa1dc79fd8a"


def build_env():
    env = os.environ.copy()

    cuda_paths = [
        NVIDIA_WSL_DRIVER_DIR,
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.1/lib64",
        "/usr/local/cuda-13.2/lib64",
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
        "avg_full_frame_ms": parse_metric(
            r"Average full frame runtime:\s+([0-9.]+)", output
        ),
        "avg_system_update_ms": parse_metric(
            r"Average system update runtime:\s+([0-9.]+)", output
        ),
        "avg_particle_update_ms": parse_metric(
            r"Average (?:CPU|CUDA) particle update runtime:\s+([0-9.]+)",
            output,
        ),
        "total_particle_update_s": parse_metric(
            r"Total (?:CPU|CUDA) particle update runtime:\s+([0-9.]+)",
            output,
        ),
        "avg_render_ms": parse_metric(
            r"Average render/swap/poll runtime:\s+([0-9.]+)", output
        ),
        "reported_overall_runtime_s": parse_metric(
            r"Overall runtime:\s+([0-9.]+)", output
        ),
        "reported_frames": parse_metric(r"Frames:\s+([0-9.]+)", output),
        "simulated_seconds": parse_metric(
            r"Simulated seconds:\s+([0-9.]+)", output
        ),
        "reported_update_steps": parse_metric(
            r"Update steps:\s+([0-9.]+)", output
        ),
    }


def run_one(binary, mode, backend, step, particles, trial, env):
    cmd = [
        binary,
        "--mode", mode,
        "--headless",
        "--particles", str(particles),
        "--step", step,
        "--steps", str(UPDATE_STEPS),
    ]

    print(
        f"[RUN] backend={backend} step={step} particles={particles} "
        f"trial={trial}/{TRIALS}"
    )
    print("      " + " ".join(cmd))

    wall_start = time.perf_counter()

    # Headless should be much faster than visual mode.
    # Still give larger particle counts enough time.
    timeout_seconds = max(60, int(UPDATE_STEPS * particles / 15_000_000) + 60)

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
        wall_end = time.perf_counter()

        combined_output = result.stdout + "\n" + result.stderr
        metrics = parse_summary(combined_output)

        return {
            "backend": backend,
            "mode": mode,
            "headless": True,
            "step": step,
            "particles": particles,
            "trial": trial,
            "update_steps_requested": UPDATE_STEPS,
            "return_code": result.returncode,
            "wall_runtime_s": wall_end - wall_start,
            **metrics,
            "command": " ".join(cmd),
            "stdout_tail": result.stdout[-1500:].replace("\n", "\\n"),
            "stderr_tail": result.stderr[-1500:].replace("\n", "\\n"),
        }

    except subprocess.TimeoutExpired as exc:
        wall_end = time.perf_counter()

        return {
            "backend": backend,
            "mode": mode,
            "headless": True,
            "step": step,
            "particles": particles,
            "trial": trial,
            "update_steps_requested": UPDATE_STEPS,
            "return_code": "TIMEOUT",
            "wall_runtime_s": wall_end - wall_start,
            "reported_overall_runtime_s": None,
            "reported_frames": None,
            "average_fps": None,
            "avg_full_frame_ms": None,
            "avg_system_update_ms": None,
            "avg_particle_update_ms": None,
            "total_particle_update_s": None,
            "avg_render_ms": None,
            "simulated_seconds": None,
            "reported_update_steps": None,
            "command": " ".join(cmd),
            "stdout_tail": str(exc.stdout)[-1500:].replace("\n", "\\n")
            if exc.stdout
            else "",
            "stderr_tail": str(exc.stderr)[-1500:].replace("\n", "\\n")
            if exc.stderr
            else "",
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
            + "\nRun: make clean && make && make cuda"
        )

    fieldnames = [
        "backend",
        "mode",
        "headless",
        "step",
        "particles",
        "trial",
        "update_steps_requested",
        "return_code",
        "wall_runtime_s",
        "reported_overall_runtime_s",
        "reported_frames",
        "reported_update_steps",
        "simulated_seconds",
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

    completed = [r for r in rows if str(r["return_code"]) == "0"]
    failed = [r for r in rows if str(r["return_code"]) != "0"]

    print(f"Completed rows: {len(completed)}")
    print(f"Failed/timeout rows: {len(failed)}")

    if completed:
        print("\nQuick completed-run summary:")
        for particles in PARTICLE_COUNTS:
            subset = [r for r in completed if r["particles"] == particles]
            if not subset:
                continue

            cpu = [
                r["avg_particle_update_ms"]
                for r in subset
                if r["backend"] == "cpu" and r["step"] == "protocol"
            ]
            cuda = [
                r["avg_particle_update_ms"]
                for r in subset
                if r["backend"] == "cuda" and r["step"] == "protocol"
            ]

            if cpu and cuda:
                cpu_avg = sum(cpu) / len(cpu)
                cuda_avg = sum(cuda) / len(cuda)
                speedup = cpu_avg / cuda_avg if cuda_avg > 0 else float("nan")

                print(
                    f"  {particles:>8,} particles protocol: "
                    f"CPU {cpu_avg:.4f} ms/update, "
                    f"CUDA {cuda_avg:.4f} ms/update, "
                    f"speedup {speedup:.2f}x"
                )


if __name__ == "__main__":
    main()