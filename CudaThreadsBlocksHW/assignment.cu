// =====================================================================================
// Dylan Renard
// EN.605.617 Introduction to GPU Programming (JHU)
// Professor Chance Pascale
// February 7th 2026
//
// CUDA Threads & Blocks Assignment — “Picture Frame” Matrix Addition
//
// Summary
// -------
// This program compares CPU vs GPU performance for element-wise matrix addition on a
// 2048x2048 matrix (N = 4,194,304 elements). It implements two versions of the same
// computation on both CPU and GPU:
//
//   (1) Baseline (minimal branching)
//       C[i] = A[i] + B[i]   for all i
//
//   (2) Branched (“picture frame” border exclusion)
//       If (i is in the interior region): C[i] = A[i] + B[i]
//       Else (border pixels):            C[i] = A[i]
//
// The branch models a common image-processing pattern where boundary pixels are treated
// differently, and it demonstrates warp divergence on the GPU.
//
// Command-line usage
// ------------------
//   ./assignment.exe <blocks> <threads_per_block>
// Example:
//   ./assignment.exe 512 256
//
// Argument behavior and guarantees:
//   - If <blocks> or <threads_per_block> is omitted, default values (512, 256) are used.
//   - If either argument is zero, negative, or non-numeric, it is clamped to 1.
//   - If <threads_per_block> exceeds the device limit, it is clamped to maxThreadsPerBlock.
//
//   Dangerous Size Edge Case Guardrails:
//   - If the requested launch size is extremely large relative to the problem size (N),
//     the number of blocks is automatically reduced using a conservative upper bound
//     (a small constant multiple of N). This allows modest oversubscription while
//     preventing integer overflow and excessive kernel launch overhead.
//
//     Example (conservative clamp):
//       For WIDTH = HEIGHT = 2048,
//         N = WIDTH * HEIGHT = 2048 * 2048 = 4,194,304 elements (2^22).
//
//       A request such as:
//         ./assignment.exe 100000000 256
//       would nominally launch 25.6 billion threads. Instead, the program clamps the
//       number of blocks to 163,840 (with 256 threads per block), resulting in
//       41,943,040 total threads (~10 × N), which is safe and sufficient to cover all
//       elements using a grid-stride loop.
//   - Note: Even if <blocks> is below the device’s theoretical maxGridSize[0], extremely
//     large block counts are still clamped relative to N to avoid overflow and wasted work.    
//
//   Terminal and File Output:
//   - Device information (GPU name and device limits) is printed at startup.
//   - The final launch configuration actually used (after all guardrails are applied)
//     is printed to stdout:
//       • Blocks
//       • Threads per block
//       • Total threads launched = blocks × threads
//   - Timing results are printed to stdout (nanoseconds):
//       • GPU baseline time
//       • GPU branched time
//       • CPU baseline time
//       • CPU branched time
//   - The same configuration + timing results are written to timing_results.txt.
//
//   Timing methodology:
//   - Timing uses std::chrono::high_resolution_clock.
//   - GPU timings measure elapsed host time around the kernel launch AND include a
//     cudaDeviceSynchronize() call to ensure the kernel has completed before stopping
//     the timer (i.e., kernel launch + execution time is captured).
//   - A one-time GPU warm-up kernel is executed before timing to avoid including CUDA
//     context initialization overhead in the measured GPU runtimes.
//   - CPU timings measure the full execution time of the corresponding CPU loop.
//
//   - Note: std::chrono GPU timing can include small host-side overhead; CUDA events
//     would provide more precise device-only timing, but are not required for this assignment.

// =====================================================================================

#include <cuda_runtime.h>

#include <chrono>     // timing
#include <cstdlib>    // atoi
#include <fstream>    // std::ofstream
#include <iostream>   // std::cout
#include <string>     // std::string

// -------------------------------------------------------------------------------------
// 1) Constants (code quality: no magic numbers)
// -------------------------------------------------------------------------------------
static constexpr int WIDTH  = 2048;              // matrix "image" width
static constexpr int HEIGHT = 2048;              // matrix "image" height
static constexpr int N      = WIDTH * HEIGHT;    // total elements
static constexpr int BORDER = 32;                // picture-frame border thickness

// -------------------------------------------------------------------------------------
// 2) CUDA error-checking helper
//
// Wraps CUDA runtime API calls and checks their return status.
// If an error occurs, prints a descriptive message (file + line number)
// and terminates the program. This ensures CUDA errors are caught
// immediately instead of failing silently.
// -------------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "      \
                << cudaGetErrorString(err) << std::endl;                          \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)


// Query GPU device properties and clamp the user-requested launch configuration
// to valid hardware limits. Ensures at least one block and thread, clamps
// threads-per-block to maxThreadsPerBlock, and clamps the grid size to the
// device’s maximum supported X dimension.
void clampLaunchConfigToDevice(int& blocks, int& threads) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Guard rails: basic sanity
  if (threads <= 0) threads = 1;
  if (blocks  <= 0) blocks  = 1;

  // Clamp threads-per-block
  if (threads > prop.maxThreadsPerBlock) {
    std::cerr << "[Guardrail] threads_per_block=" << threads
              << " exceeds device maxThreadsPerBlock=" << prop.maxThreadsPerBlock
              << ". Clamping.\n";
    threads = prop.maxThreadsPerBlock;
  }

  // Clamp blocks in X dimension (we only use a 1D grid)
  int maxBlocksX = prop.maxGridSize[0];
  if (blocks > maxBlocksX) {
    std::cerr << "[Guardrail] blocks=" << blocks
              << " exceeds device maxGridSize[0]=" << maxBlocksX
              << ". Clamping.\n";
    blocks = maxBlocksX;
  }

  // Optional: print device info once (useful evidence for submission)
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "Device limits: maxThreadsPerBlock=" << prop.maxThreadsPerBlock
            << ", maxGridSize[0]=" << prop.maxGridSize[0] << "\n";
}


// -------------------------------------------------------------------------------------
// 3) CPU implementations
//    Reference implementations that perform the same computations as the GPU kernels,
//    executed serially on the CPU for correctness and performance comparison.
// -------------------------------------------------------------------------------------

// Baseline CPU implementation: performs element-wise addition for all N elements
// with no conditional branching inside the loop.
void matrixAddCPU(const float* A, const float* B, float* C) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

// Branched CPU: picture-frame border exclusion.
// Only compute interior pixels; leave border ("crust") as A[i].
void matrixAddCPU_Branched(const float* A, const float* B, float* C) {
  for (int i = 0; i < N; i++) {
    // Convert 1D index -> 2D coordinates (row, col)
    int row = i / WIDTH;
    int col = i % WIDTH;

    bool interior =
        (row >= BORDER) && (row < (HEIGHT - BORDER)) &&
        (col >= BORDER) && (col < (WIDTH  - BORDER));

    if (interior) {
      C[i] = A[i] + B[i];
    } else {
      // Border region: skip add
      C[i] = A[i];
    }
  }
}

// -------------------------------------------------------------------------------------
// 4) GPU kernels
//    Each thread computes one element at index i.
// -------------------------------------------------------------------------------------

// Baseline GPU kernel: minimal branching (only bounds check).
__global__ void matrixAddGPU(const float* A, const float* B, float* C) {
    unsigned long long tid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;

    unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long i = tid; i < (unsigned long long)N; i += stride) {
        int idx = (int)i;  // safe because i < N
        C[idx] = A[idx] + B[idx];
    }
}

// Branched GPU kernel: picture-frame border exclusion (branching inside kernel).
__global__ void matrixAddGPU_Branched(const float* A, const float* B, float* C) {
    unsigned long long tid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;

    unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long i = tid; i < (unsigned long long)N; i += stride) {
        int idx = (int)i;          // safe because i < N
        int row = idx / WIDTH;
        int col = idx % WIDTH;

        bool interior =
            (row >= BORDER) && (row < HEIGHT - BORDER) &&
            (col >= BORDER) && (col < WIDTH  - BORDER);

        if (interior) C[idx] = A[idx] + B[idx];
        else          C[idx] = A[idx];
    }
}

// -------------------------------------------------------------------------------------
// 5) Utility: write timing results to a text file
// -------------------------------------------------------------------------------------
void writeTimingResults(const std::string& filename,
                        int blocks,
                        int threads,
                        long long gpu_baseline_ns,
                        long long gpu_branched_ns,
                        long long cpu_baseline_ns,
                        long long cpu_branched_ns) {
  std::ofstream out(filename, std::ios::out);  // overwrite each run (simple & clear)

  if (!out.is_open()) {
    std::cerr << "WARNING: Could not open " << filename << " for writing.\n";
    return;
  }

  out << "Matrix Addition Timing Results (Picture Frame Branching)\n";
  out << "Matrix size: " << WIDTH << " x " << HEIGHT << " (N=" << N << ")\n";
  out << "Border thickness (fixed): " << BORDER << "\n\n";

  out << "CUDA launch configuration (from CLI):\n";
  out << "  Blocks: " << blocks << "\n";
  out << "  Threads per block: " << threads << "\n";
  out << "  Total threads launched: " << (static_cast<long long>(blocks) * threads) << "\n\n";

  out << "Timings (nanoseconds):\n";
  out << "  GPU baseline:   " << gpu_baseline_ns << "\n";
  out << "  GPU branched:   " << gpu_branched_ns << "\n";
  out << "  CPU baseline:   " << cpu_baseline_ns << "\n";
  out << "  CPU branched:   " << cpu_branched_ns << "\n\n";

  out << "Notes:\n";
  out << "  - GPU timings include kernel launch + execution time and use cudaDeviceSynchronize().\n";
  out << "  - CPU timings include the full loop execution.\n";
  out << "  - Branching may slow the GPU due to warp divergence near the border region.\n";

  out.close();
}

// -------------------------------------------------------------------------------------
// 6) Main: parse CLI args, allocate memory, run CPU/GPU, time, print, write file
// -------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // Defaults (safe; still allows running with no args)
  int blocks  = 512;
  int threads = 256;

  // The provided assignment skeleton typically supports:
  //   ./assignment.exe <blocks>
  //   ./assignment.exe <blocks> <threads>
  if (argc == 2) {
    blocks = std::atoi(argv[1]);
  } else if (argc == 3) {
    blocks  = std::atoi(argv[1]);
    threads = std::atoi(argv[2]);
  }

  clampLaunchConfigToDevice(blocks, threads);
  // ------------------------------------------------------------------
// Guardrail: prevent absurdly large launches relative to problem size
// ------------------------------------------------------------------
long long totalThreadsLaunched = 1LL * blocks * threads;
long long maxReasonableThreads = 10LL * N;   // allow up to 10x oversubscription

if (totalThreadsLaunched > maxReasonableThreads) {
    std::cerr << "[Guardrail] Requested launch (blocks * threads = "
              << totalThreadsLaunched
              << ") is much larger than problem size N=" << N << ".\n";

    long long maxBlocks = (maxReasonableThreads + threads - 1) / threads;
    if (maxBlocks < 1) maxBlocks = 1;

    if (maxBlocks < blocks) {
        std::cerr << "[Guardrail] Clamping blocks from "
                  << blocks << " to " << maxBlocks
                  << " to avoid overflow and wasted work.\n";
        blocks = static_cast<int>(maxBlocks);
    }
}

// Final launch configuration (always print this)
std::cout << "Final launch configuration:\n";
std::cout << "  Blocks: " << blocks << "\n";
std::cout << "  Threads per block: " << threads << "\n";
std::cout << "  Total threads launched: "
          << (1LL * blocks * threads) << "\n\n";

  // ------------------------------------------------------------------
  // Allocate host memory (CPU-side arrays)
  // ------------------------------------------------------------------
  float* h_A      = new float[N];
  float* h_B      = new float[N];
  float* h_C_cpu  = new float[N];  // store CPU results
  float* h_C_gpu  = new float[N];  // store GPU results (copied back)

  // Initialize input matrices with deterministic "image-like" values.
  // This keeps runs reproducible and avoids file I/O complexity.
  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i % 256) / 255.0f;  // range approx [0, 1]
    h_B[i] = 0.5f;
  }

  // ------------------------------------------------------------------
  // Allocate device memory (GPU-side arrays)
  // ------------------------------------------------------------------
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

  // Copy inputs to GPU once (we reuse them for both kernels)
  CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

  // ------------------------------------------------------------------
  // GPU WARM-UP
  // The first CUDA call can include one-time initialization overhead.
  // We run a quick kernel once BEFORE timing so the measurements are fair.
  // ------------------------------------------------------------------
  matrixAddGPU<<<blocks, threads>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ------------------------------------------------------------------
  // TIMING: GPU baseline
  // We synchronize after launch so timing includes kernel execution.
  // ------------------------------------------------------------------
  long long gpu_baseline_ns = 0;
  {
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddGPU<<<blocks, threads>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());          // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // wait until kernel finishes
    auto stop = std::chrono::high_resolution_clock::now();

    gpu_baseline_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  }

  // ------------------------------------------------------------------
  // TIMING: GPU branched
  // ------------------------------------------------------------------
  long long gpu_branched_ns = 0;
  {
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddGPU_Branched<<<blocks, threads>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    gpu_branched_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  }

  // Optionally copy GPU output back (useful for sanity checks / evidence)
  CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

  // ------------------------------------------------------------------
  // TIMING: CPU baseline
  // ------------------------------------------------------------------
  long long cpu_baseline_ns = 0;
  {
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddCPU(h_A, h_B, h_C_cpu);
    auto stop = std::chrono::high_resolution_clock::now();

    cpu_baseline_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  }

  // ------------------------------------------------------------------
  // TIMING: CPU branched
  // ------------------------------------------------------------------
  long long cpu_branched_ns = 0;
  {
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddCPU_Branched(h_A, h_B, h_C_cpu);
    auto stop = std::chrono::high_resolution_clock::now();

    cpu_branched_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  }

  // ------------------------------------------------------------------
  // Print results to terminal (for screenshots / evidence)
  // ------------------------------------------------------------------
  std::cout << "--- Timing Results (nanoseconds) ---\n";
  std::cout << "GPU baseline:   " << gpu_baseline_ns << "\n";
  std::cout << "GPU branched:   " << gpu_branched_ns << "\n";
  std::cout << "CPU baseline:   " << cpu_baseline_ns << "\n";
  std::cout << "CPU branched:   " << cpu_branched_ns << "\n\n";

  // ------------------------------------------------------------------
  // Write results to a text file (for submission)
  // ------------------------------------------------------------------
  writeTimingResults("timing_results.txt",
                     blocks, threads,
                     gpu_baseline_ns, gpu_branched_ns,
                     cpu_baseline_ns, cpu_branched_ns);

  std::cout << "Wrote timing_results.txt\n";

  // ------------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------------
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_cpu;
  delete[] h_C_gpu;

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}
