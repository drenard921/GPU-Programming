# CUDA Memory Assignment

## 1D LUT Filter + Shared Memory Gaussian Blur

**Dylan Renard**
EN.605.617 Introduction to GPU Programming
Professor Chance Pascale
February 22nd 2026



# Overview

This project evaluates CUDA memory hierarchy performance using:

-   A 1D color Look Up Table (LUT) filter
-   An optional shared memory 3×3 Gaussian blur
-   Configurable block geometry
-   Explicit measurement of HtoD, kernel, and DtoH timings
-   Structured 8×8 parameter sweeps
-   Pareto frontier performance analysis

The goal is not only functional correctness, but systematic evaluation
of how memory hierarchy and execution configuration influence GPU
performance.


# Repository Structure

    MemoryHW/
    │
    ├── InputImages/
    │   └── Smallville.png
    │
    ├── OutputImages/
    │   └── (processed images written here)
    │
    ├── output_plots/
    │   ├── run_logs/
    │   ├── 20260222_200230_all_runs.csv
    │   ├── results.csv
    │   ├── overall_kernel_vs_total_pareto.png
    │   ├── A_bx_sweep_blur0_*.png
    │   ├── B_by_sweep_blur0_*.png
    │   ├── C_passes_sweep_blur1_*.png
    │   ├── D_bx_sweep_blur1_p8_*.png
    │   ├── E_by_sweep_blur1_p8_*.png
    │   └── (throughput and timing breakdown plots)
    │
    ├── assignment.cu
    ├── cuda_filter
    ├── Makefile
    ├── grid_run_smallville.py
    ├── overall_kernel_vs_total_pareto.py
    ├── MemoryAssignment.pdf
    └── README.md



# Build Instructions

From inside `MemoryHW/`:

``` bash
make
```

# Running the CUDA Program

``` bash
./cuda_filter --in InputImages/Smallville.png               --out test.png               --mode bw               --bx 16               --by 16               --blur 1               --passes 3
```

Console output includes:

-   HtoD time (ms)
-   Kernel time (ms)
-   DtoH time (ms)
-   Total time (ms)


# Running the Grid Search

``` bash
python3 grid_run_smallville.py --img InputImages/Smallville.png
```

Outputs saved to:

    output_plots/

Master CSV:

    output_plots/20260222_200230_all_runs.csv


# Generating the Pareto Plot

``` bash
python3 overall_kernel_vs_total_pareto.py   --csv output_plots/20260222_200230_all_runs.csv   --mode bw
```

Output:

    output_plots/overall_kernel_vs_total_pareto.png

## Conclusion: 
This project demonstrates that CUDA performance is shaped not only by algorithmic correctness, but by careful interaction with the memory hierarchy and execution configuration. Shared memory tiling significantly reduces redundant global memory accesses during Gaussian blur, improving locality and increasing arithmetic intensity. Block geometry, particularly the X dimension, strongly influences memory coalescing due to row major image layout. Structured sweeps reveal that insufficient threads per block underutilize the GPU, while excessively large configurations provide diminishing returns once occupancy saturation is reached. Performance is therefore a balance between parallelism, memory access efficiency, and resource utilization.

This project demonstrates that CUDA performance is shaped not only by algorithmic correctness, but by careful interaction with the memory hierarchy and execution configuration. Shared memory tiling significantly reduces redundant global memory accesses during Gaussian blur, improving locality and increasing arithmetic intensity. Block geometry, particularly the X dimension, strongly influences memory coalescing due to row major image layout. Structured sweeps reveal that insufficient threads per block underutilize the GPU, while excessively large configurations provide diminishing returns once occupancy saturation is reached. Performance is therefore a balance between parallelism, memory access efficiency, and resource utilization.