# CUDA Threads and Blocks Assignment  
## Matrix Addition with Picture Frame Branching

---

## Overview

This project demonstrates CPU and GPU implementations of the **same algorithm** (matrix addition) using CUDA. It compares execution time between CPU and GPU code and evaluates the impact of **conditional branching** on GPU performance using a fixed “picture frame” (border exclusion) approach.

The goal of this assignment is to illustrate:
- How CUDA threads and blocks execute a kernel
- How the same algorithm behaves on CPU vs GPU
- How conditional branching affects GPU performance (warp divergence)
- How execution configuration (blocks and threads) impacts runtime

---

## Project Structure
* assignment.cu # Main CUDA source file
* timing_results.txt # Generated at runtime (timing output)
* performance_chart.png # (Optional) Chart created from timing results
* README.md # This file

---

## Build Instructions

From the directory containing the provided `Makefile`, run:

```bash
make
```

This will compile assignment.cu and produce the excecutable

assignment.exe

## How to Run

./assignment.exe <num_blocks> <threads_per_block>

## Required example (per assignment specification):
./assignment.exe 512 256

## Additional examples
./assignment.exe 128 256
./assignment.exe 512 128
./assignment.exe 1024 256
