// =====================================================================================
// module3/assignment.cu
//
// CUDA Threads & Blocks Assignment — “Picture Frame” Matrix Addition
//
// What this program demonstrates (matches the rubric):
//   1) SAME algorithm on CPU and GPU with minimal branching (baseline).
//        - Baseline: C[i] = A[i] + B[i] for ALL elements.
//   2) SAME algorithm on CPU and GPU WITH conditional branching (branched).
//        - Branched: Only compute interior pixels (picture frame). Border is left unchanged.
//   3) Command-line arguments control CUDA execution configuration:
//        ./assignment.exe <blocks> <threads_per_block>
//      Example required by spec: ./assignment.exe 512 256
//   4) Prints timings to terminal AND writes them to a text file (timing_results.txt).
//
// Notes:
//   - We treat the matrix as a flattened 1D array in row-major order.
//   - Each GPU thread processes exactly one element.
//   - BORDER is fixed (not a CLI arg) to keep the experiment controlled.
//   - Timing uses std::chrono. For GPU, we synchronize after the kernel to ensure completion.
//
// Build/run expectation from assignment spec: make -> assignment.exe, then run with args.
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
// 2) CUDA error-checking helper (makes debugging MUCH easier)
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


// Query GPU limits and clamp blocks/threads to what the device supports.
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
//    These functions do the SAME work as the GPU kernels, but serially on the CPU.
// -------------------------------------------------------------------------------------

// Baseline CPU: no branching besides the for-loop itself.
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
        C[i] = A[i] + B[i];
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
