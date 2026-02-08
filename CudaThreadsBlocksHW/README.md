# CUDA Threads and Blocks Assignment  
## Matrix Addition with Picture Frame Border Detection Branching

---

## Overview

This project implements **CPU and GPU versions of the same matrix addition algorithm** using CUDA and compares their performance under different execution configurations.

Two GPU kernels are evaluated:

1. **Baseline kernel** – performs element-wise addition with no conditional branching  
2. **Branched kernel** – excludes a fixed “picture frame” border region, introducing conditional logic

The goal of this assignment is to demonstrate:

- How CUDA threads and blocks execute kernels  
- How execution configuration impacts performance  
- How conditional branching affects GPU execution (warp divergence)  
- The performance gap between CPU and GPU implementations  

---

## Problem Setup

- Matrix size: **2048 × 2048**
- Total elements: **N = 4,194,304**
- Border thickness (branched case): **32 pixels**
- Data type: `float`
- GPU kernel uses **grid-stride loops** to ensure full coverage regardless of launch size

---

## Build Instructions

```bash
make
```

Produces:

```text
assignment.exe
```

---

## How to Run

```bash
./assignment.exe <num_blocks> <threads_per_block>
```

Required example:

```bash
./assignment.exe 512 256
```

---

## Timing Methodology

- CPU timing uses `std::chrono::high_resolution_clock`
- GPU timing measures kernel launch + execution using `cudaDeviceSynchronize()`
- A warm-up kernel removes first-launch overhead
- All timings reported in **nanoseconds**

---

## Parameter Sweep

Blocks tested:
```
1, 4, 16, 64, 256, 1024, 4096, 16384
```

Threads per block tested:
```
8, 16, 32, 64, 128, 256, 512, 1024
```

All combinations were executed and recorded in `sweep_results.csv`.

All charts generated are recorded in the charts_out directory

---

## Results and Analysis

### GPU Performance
- Underutilization occurs with very small block counts
- Performance stabilizes once sufficient parallelism is reached
- Excessively large thread blocks can reduce occupancy

### Branching Effects
- Branched kernel is consistently slower due to warp divergence
- Branch penalty observed between **5–35%**

### CPU vs GPU
- CPU runtime is largely constant across runs
- GPU provides **20×–26× speedup** for the baseline kernel
- Speedup decreases for branched kernel but remains substantial

---

## Conclusion

This assignment demonstrates how CUDA execution configuration and branching behavior influence performance. Proper selection of blocks and threads is critical to achieving high GPU utilization, and conditional branching introduces measurable performance penalties due to warp divergence.
