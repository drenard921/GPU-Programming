# GPU-Accelerated Counterflow Centrifugation Simulation

**Author:** Dylan Renard  
**Course:** EN.605.617 Introduction to GPU Programming  
**Professor:** Chance Pascale  
**Date:** May 3, 2026  

---

## 1. Project Overview

This project implements a **Rotea-inspired counterflow centrifugation (CFC) particle simulator** with both CPU and CUDA execution paths. The goal is to model particle transport inside a simplified rotating conical chamber while demonstrating CUDA-based parallel particle updates, real-time OpenGL visualization, replay generation, and CPU-vs-GPU benchmarking.

The simulation is inspired by the CTS Rotea Counterflow Centrifugation System, where particles are manipulated by opposing effects from chamber rotation and fluid flow. In the real process, particles/cells can be retained in a fluidized bed or elutriated from the chamber depending on G-force, flow rate, particle size, particle density, media density, and media viscosity.

This project is **not** a validated Rotea process model. It is a physically motivated, visualization-oriented CUDA simulation designed for a GPU programming final project.

---

## 2. Quick Start

From the project root, build everything:

```bash
make clean
make all
```

Run a short CPU visual demo:

```bash
./bin/rotea_sim --mode cpu --particles 5000 --step protocol --duration 10
```

Run a short CUDA visual demo:

```bash
./bin/rotea_sim_cuda --mode cuda --particles 5000 --step protocol --duration 10
```

Run a CPU headless benchmark:

```bash
./bin/rotea_sim --mode cpu --headless --particles 10000 --step protocol --steps 10000
```

Run a CUDA headless benchmark:

```bash
./bin/rotea_sim_cuda --mode cuda --headless --particles 10000 --step protocol --steps 10000
```

A successful headless CUDA run should end with a benchmark summary similar to:

```text
Mode: cuda
Step: protocol
Frames: 10000
Average CUDA particle update runtime: ...
Average render/swap/poll runtime: 0.0000 ms/frame
```

---

## 3. Key Features

- Real-time OpenGL visualization of a conical CFC-style chamber
- CPU particle update backend
- CUDA particle update backend
- Persistent CUDA particle buffer for live CUDA mode
- Headless benchmark mode for measuring particle-update performance without rendering overhead
- Standalone CUDA frame generator for replay mode
- Replay mode for visualizing precomputed CUDA particle frames
- Protocol phase simulation:
  - Load
  - Concentrate
  - Wash
  - Harvest
- Rotea-inspired force model using:
  - Stokes-like drag
  - Density-dependent centrifugal migration
  - Particle diameter effects
  - Flow-rate effects
  - G-force effects
  - Conical chamber constraints
- Benchmark scripts and plotting workflow for CPU/CUDA runtime comparison

---

## 4. Repository Layout

Expected project structure:

```text
RoteaFinalProject/
├── Makefile
├── README.md
├── benchmark.py                         # visual/end-to-end CPU vs CUDA benchmark runner
├── benchmark_headless.py                # headless CPU vs CUDA benchmark runner without OpenGL rendering
├── benchmark_plot.py                    # seaborn/matplotlib plotting script for visual benchmark CSVs
├── src/
│   ├── main.cpp                         # CLI, app control, visual mode, replay mode, headless mode
│   ├── sim_cpu.cpp                      # CPU-side system and particle simulation
│   ├── sim_cpu.h
│   ├── simulation.cu                    # CUDA kernel, live CUDA mode, CUDA frame generator
│   ├── simulation.h
│   ├── renderer.cpp                     # OpenGL rendering routines
│   ├── renderer.h
│   ├── camera.cpp                       # camera support, if present
│   ├── shader.cpp                       # shader support, if present
│   └── types.h                          # shared Particle, Bag, Chamber, Line, and Step structures
├── benchmarks/                          # benchmark CSVs and generated plots, recommended
├── media/                               # screenshots or demo video links, recommended
└── bin/                                 # generated binaries after build
```

Generated binaries:

```text
bin/rotea_sim        # CPU/OpenGL app; also supports replay and CPU headless mode
bin/rotea_sim_cuda   # CUDA/OpenGL app; also supports CUDA headless mode
bin/cuda_frames      # standalone CUDA frame generator
```

---

## 5. Requirements

This project was developed and tested in a Linux/WSL2 environment with an NVIDIA GPU.

### Required

- C++17 compiler, such as `g++`
- NVIDIA CUDA Toolkit with `nvcc`
- OpenGL development libraries
- GLFW development libraries
- GNU Make
- Python 3 for benchmark scripts
- Python packages for plotting: `pandas`, `numpy`, `seaborn`, `matplotlib`

### Example Ubuntu/WSL package dependencies

```bash
sudo apt update
sudo apt install build-essential make libglfw3-dev libgl1-mesa-dev
```

CUDA should be installed separately from NVIDIA. The Makefile assumes:

```text
CUDA_HOME=/usr/local/cuda
```

If your CUDA installation is elsewhere, override it during build:

```bash
make CUDA_HOME=/path/to/cuda
```

---

## 6. Build Instructions

From the project root:

```bash
make clean
make all
```

This builds:

```text
bin/rotea_sim
bin/rotea_sim_cuda
bin/cuda_frames
```

You can also build components separately:

```bash
make app          # CPU/OpenGL application
make app-cuda     # CUDA/OpenGL application
make cuda-frames  # standalone CUDA frame generator
```

To inspect build settings:

```bash
make info
```

To remove generated build products:

```bash
make clean
```

---

## 7. Running the Visual Simulator

### CPU visual simulation

```bash
./bin/rotea_sim --mode cpu --particles 10000 --step protocol --duration 30
```

### CUDA visual simulation

```bash
./bin/rotea_sim_cuda --mode cuda --particles 10000 --step protocol --duration 30
```

### Run a single protocol phase

```bash
./bin/rotea_sim --mode cpu --particles 10000 --step load --duration 15
./bin/rotea_sim --mode cpu --particles 10000 --step concentrate --duration 15
./bin/rotea_sim --mode cpu --particles 10000 --step wash --duration 15
./bin/rotea_sim --mode cpu --particles 10000 --step harvest --duration 15
```

CUDA equivalent:

```bash
./bin/rotea_sim_cuda --mode cuda --particles 10000 --step wash --duration 15
```

Supported step options:

```text
protocol
load
concentrate
wash
harvest
```

---

## 8. Headless Benchmark Mode

The headless benchmark mode runs the simulation update loop without creating an OpenGL window. This is useful because visual benchmarks include rendering, swap/poll, display timing, and other interactive overhead.

### CPU headless benchmark

```bash
./bin/rotea_sim --mode cpu --headless --particles 10000 --step protocol --steps 10000
```

### CUDA headless benchmark

```bash
./bin/rotea_sim_cuda --mode cuda --headless --particles 10000 --step protocol --steps 10000
```

Example output fields:

```text
Average FPS
Average full frame runtime
Average system update runtime
Average CPU/CUDA particle update runtime
Total CPU/CUDA particle update runtime
Average render/swap/poll runtime
```

In headless mode, render time should report as approximately:

```text
Average render/swap/poll runtime: 0.0000 ms/frame
```

This mode is the cleanest way to compare CPU and CUDA particle-update performance.

---

## 9. Standalone CUDA Frame Generator and Replay Mode

The project also supports offline CUDA frame generation. The CUDA frame generator writes a binary replay file that can be visualized later.

### Generate replay frames

```bash
./bin/cuda_frames --particles 5000 --frames 300 --output frames.bin
```

### Replay generated frames

```bash
./bin/rotea_sim --mode replay --replay frames.bin --duration 30
```

The Makefile includes shortcuts:

```bash
make test-frames
make replay
make cuda-replay
```

Replay files use the following format:

```text
magic[8] = "ROTEAFRM"
version
particleCount
frameCount
dt
Particle frame data
```

---

## 10. Benchmarking and Plotting Workflow

The repository includes three Python helper scripts for benchmarking and plotting. These scripts are not required to launch the simulator manually, but they make the performance experiments reproducible.

### `benchmark.py` — visual/end-to-end benchmark runner

`benchmark.py` runs the OpenGL application in CPU and CUDA modes across a grid of particle counts, protocol steps, and trials. It captures stdout/stderr, parses the benchmark summary printed by the simulator, records return codes and wall-clock runtime, and writes the results to a CSV file.

The default configuration is aimed at larger visual stress tests:

```text
particle counts: 100000, 200000, 500000, 1000000
steps: load, concentrate, wash, harvest
modes: cpu, cuda
trials: 3
visual duration: 30 seconds
output: rotea_cpu_cuda_grid_benchmark_large.csv
```

Run it with:

```bash
python benchmark.py
```

This benchmark measures the full interactive application loop:

```text
system update + particle update + render/swap/poll + display/event overhead
```

Because this includes OpenGL rendering and display synchronization, it should be interpreted as an end-to-end visual application benchmark, not as a pure CUDA kernel benchmark. Timeout or failed rows should be treated as stress-test outcomes rather than valid timing measurements.

### `benchmark_headless.py` — simulation-update benchmark runner

`benchmark_headless.py` runs the same CPU and CUDA executables using `--headless --steps`, which disables OpenGL rendering and measures only the update loop. This is the cleaner benchmark for comparing CPU and CUDA particle-update cost.

The default configuration is:

```text
particle counts: 1000, 2000, 5000, 10000, 50000, 100000, 200000
steps: protocol, load, concentrate, wash, harvest
modes: cpu, cuda
trials: 3
update steps: 10000
output: rotea_cpu_cuda_headless_benchmark.csv
```

Run it with:

```bash
python benchmark_headless.py
```

For a faster first pass, reduce the grid inside the script to something like:

```python
PARTICLE_COUNTS = [10000, 50000, 100000]
STEPS = ["protocol"]
TRIALS = 3
UPDATE_STEPS = 10000
```

### `benchmark_plot.py` — plotting and summary script

`benchmark_plot.py` reads benchmark CSV files, filters rows that completed successfully, summarizes CPU/CUDA timing by particle count and protocol step, and generates report-ready plots in a `benchmark_plots/` folder.

Expected inputs:

```text
rotea_cpu_cuda_grid_benchmark.csv
rotea_cpu_cuda_grid_benchmark_large.csv
```

Main outputs:

```text
benchmark_plots/fps_vs_particles.png
benchmark_plots/full_frame_runtime_vs_particles.png
benchmark_plots/particle_update_runtime_vs_particles.png
benchmark_plots/render_runtime_vs_particles.png
benchmark_plots/timeout_summary.png
benchmark_plots/fps_by_step.png
benchmark_plots/particle_update_runtime_by_step.png
benchmark_plots/benchmark_summary_clean.csv
benchmark_plots/benchmark_summary_by_group.csv
benchmark_plots/cuda_vs_cpu_percent_difference.csv
```

Run it with:

```bash
python benchmark_plot.py
```

**Note:** `benchmark_plot.py` currently targets the visual benchmark CSVs. Headless benchmark results are written to `rotea_cpu_cuda_headless_benchmark.csv` and can be plotted separately or added to the plotting script.

Recommended plots for the final report:

- CPU vs CUDA average FPS for valid visual runs
- CPU vs CUDA particle update runtime
- Full-frame visual runtime
- Rendering/runtime overhead
- Timeout/failure summary for high-count visual stress tests
- Headless particle update runtime from `rotea_cpu_cuda_headless_benchmark.csv`

### Benchmark interpretation

Use the two benchmark modes for different claims:

```text
Visual benchmark:  full application behavior, including rendering and display overhead
Headless benchmark: update-loop behavior without OpenGL rendering
```

The strongest final-project performance claim should come from headless mode because it isolates the particle update. Visual mode is still valuable because it shows the real interactive cost of the application.

---

## 11. Simulation Model

The simulation represents the CFC chamber as a simplified conical annulus. Particles represent either fluid-like particles or denser cell-like particles.

### Particle types

```text
Blue particles   = lower-density fluid-like particles
Yellow particles = denser cell-like particles
```

### Major modeled effects

The CPU and CUDA update paths are designed around a normalized Rotea-inspired force model:

```text
F_net = F_centrifugal - F_Stokes_drag
```

The simulation includes dependencies on:

- Particle diameter
- Particle density
- Media density
- Media viscosity
- Radial position
- G-force
- Flow rate
- Local cone radius / chamber geometry

The model is intentionally normalized for visual stability and runtime performance. It preserves the qualitative dependencies of CFC behavior but does not claim SI-unit calibration or experimental validation.

---

## 12. Protocol Phases

The simulator includes four major processing phases.

### Load

Particles enter and establish a fluidized-bed-like pattern. Denser cell-like particles begin to separate from fluid-like particles.

### Concentrate

Higher G-force and lower flow promote retention and concentration of denser particles.

### Wash

Flow through the bed is increased to mimic buffer exchange while denser particles remain preferentially retained.

### Harvest

Flow is reversed to recover concentrated cell-like particles from the chamber, inspired by concentrate recovery behavior.

---

## 13. CUDA Implementation Details

The CUDA implementation is contained primarily in:

```text
src/simulation.cu
src/simulation.h
```

CUDA features demonstrated:

- Device memory allocation with `cudaMalloc`
- Persistent CUDA particle buffer for live CUDA mode
- Host-to-device and device-to-host particle transfer
- CUDA kernel launch
- Grid-stride loop for particle counts larger than the launch grid
- One logical CUDA thread per particle update
- CUDA error checking
- CUDA frame generator mode
- CUDA event timing in standalone frame generation

The live CUDA visual mode currently copies the particle buffer between CPU and GPU each frame so that the OpenGL renderer can consume CPU-side particle positions. This design is simple and reliable, but it limits end-to-end speedup.

---

## 14. Performance Interpretation

The project includes both visual and headless timing modes because they answer different questions.

### Visual mode answers

```text
How fast is the full interactive application?
```

This includes simulation, CUDA transfer/synchronization, rendering, swap, and event polling.

### Headless mode answers

```text
How fast is the update loop without OpenGL rendering?
```

This gives a cleaner CPU-vs-CUDA comparison for particle-update performance.

In testing, visual benchmarks showed that rendering and synchronization can dominate the frame loop. Headless benchmarking was added to isolate simulation update time and better evaluate whether CUDA improves particle update performance.

The main CUDA lesson from this project is that GPU acceleration is not automatically beneficial just because a problem is parallel. CUDA is most effective when the workload has enough computation per element and the application minimizes host-device transfer and synchronization overhead.

---

## 15. Known Limitations

This project is a GPU programming simulation and visualization project, not a validated process-development tool.

Known limitations:

- The model uses normalized simulation units rather than calibrated SI units.
- The chamber geometry is simplified.
- The fluid field is approximated rather than solved with CFD.
- There is no experimental validation against Rotea run data.
- Live CUDA mode still copies particles between host and device each frame.
- CUDA/OpenGL interoperability is not implemented.
- Bubble trap behavior, pressure triggers, OD sensor behavior, and detailed kit fluidics are simplified or omitted.
- High-particle-count visual timing can become unreliable because the visual loop and timing harness may dominate runtime.

---

## 16. Future Work

Potential future improvements:

- CUDA/OpenGL interoperability so CUDA can update an OpenGL vertex buffer directly
- Structure-of-arrays particle memory layout for improved GPU memory coalescing
- CUDA event-based timing for live mode
- More direct CPU/CUDA numerical validation tests
- Calibrated physical units for flow rate, chamber dimensions, and angular velocity
- More realistic local cross-sectional flow calculations
- Pressure, bubble sensor, and OD sensor abstractions
- UI controls for G-force, flow rate, and media properties
- Experimental comparison against bead or cell runs

The most important optimization would be CUDA/OpenGL interoperability:

```text
Current live CUDA design:
CPU particles -> copy to GPU -> CUDA update -> copy back to CPU -> OpenGL render

Future optimized design:
OpenGL VBO registered with CUDA -> CUDA updates GPU buffer -> OpenGL renders GPU buffer
```

---

## 17. Quick Verification Commands

Use these commands to verify that the main project components work.

### Build everything

```bash
make clean
make all
```

### CPU visual demo

```bash
./bin/rotea_sim --mode cpu --particles 5000 --step protocol --duration 10
```

### CUDA visual demo

```bash
./bin/rotea_sim_cuda --mode cuda --particles 5000 --step protocol --duration 10
```

### CPU headless benchmark

```bash
./bin/rotea_sim --mode cpu --headless --particles 10000 --step protocol --steps 10000
```

### CUDA headless benchmark

```bash
./bin/rotea_sim_cuda --mode cuda --headless --particles 10000 --step protocol --steps 10000
```

### CUDA frame generation

```bash
./bin/cuda_frames --particles 5000 --frames 300 --output frames.bin
```

### Replay mode

```bash
./bin/rotea_sim --mode replay --replay frames.bin --duration 10
```

### Visual benchmark script

```bash
python benchmark.py
```

### Headless benchmark script

```bash
python benchmark_headless.py
```

### Benchmark plot script

```bash
python benchmark_plot.py
```

---

## 18. Troubleshooting

### CUDA binary cannot find `libcudart`

Check your CUDA library path:

```bash
ldd ./bin/rotea_sim_cuda | grep cudart
```

If needed:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

On WSL, you may also need:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### OpenGL window does not open

Confirm GLFW and OpenGL development libraries are installed:

```bash
sudo apt install libglfw3-dev libgl1-mesa-dev
```

If running through WSL, make sure WSLg or another display server is available.

### CUDA mode reports no device

Check CUDA visibility:

```bash
nvidia-smi
nvcc --version
```

### High-particle visual benchmarks time out

Use headless mode for timing:

```bash
./bin/rotea_sim_cuda --mode cuda --headless --particles 100000 --step protocol --steps 10000
```

Visual mode is best treated as an interactive demonstration. Headless mode is better for update-performance measurement.

---

## 19. File Guide

### Core source files

| File | Purpose |
|---|---|
| `src/main.cpp` | Command-line parsing, visual app loop, replay mode, protocol selection, benchmark summary printing, and headless benchmark mode. |
| `src/sim_cpu.cpp` / `src/sim_cpu.h` | CPU-side bag, chamber, protocol, and particle-update logic. |
| `src/simulation.cu` / `src/simulation.h` | CUDA particle kernel, persistent device buffer, CUDA live update function, and standalone CUDA frame generator. |
| `src/renderer.cpp` / `src/renderer.h` | OpenGL rendering of chamber geometry and particles. |
| `src/types.h` | Shared data structures for particles, bags, chamber state, lines, and protocol steps. |
| `Makefile` | Build targets for CPU app, CUDA app, CUDA frame generator, replay tests, and cleanup. |

### Benchmark and analysis files

| File | Purpose |
|---|---|
| `benchmark.py` | Runs visual/end-to-end CPU vs CUDA benchmarks using duration-based OpenGL simulations. |
| `benchmark_headless.py` | Runs headless CPU vs CUDA benchmarks using fixed update steps and no OpenGL rendering. |
| `benchmark_plot.py` | Generates seaborn/matplotlib plots and grouped summary CSVs from benchmark results. |

### Recommended generated outputs

| File or folder | Purpose |
|---|---|
| `rotea_cpu_cuda_grid_benchmark.csv` | Smaller visual benchmark CSV, if generated. |
| `rotea_cpu_cuda_grid_benchmark_large.csv` | Larger visual benchmark/stress-test CSV generated by `benchmark.py`. |
| `rotea_cpu_cuda_headless_benchmark.csv` | Headless benchmark CSV generated by `benchmark_headless.py`. |
| `benchmark_plots/` | Plot images and summary tables generated by `benchmark_plot.py`. |
| `frames.bin` | Optional binary replay file generated by `bin/cuda_frames`. |

---

## 20. Acknowledgments

This project was inspired by counterflow centrifugation workflows used in cell and gene therapy manufacturing. The simulator is Rotea-inspired but is not affiliated with, endorsed by, or validated by Thermo Fisher Scientific.

---

## 21. Short Project Summary

This project demonstrates a CUDA-capable, Rotea-inspired counterflow centrifugation simulation. It combines OpenGL visualization, CPU and CUDA particle-update modes, headless benchmarking, and CUDA replay generation. The final result shows both the value and the limits of GPU acceleration: CUDA can improve isolated particle-update performance, but full application speedup depends on rendering overhead, data movement, synchronization, and overall software architecture.
