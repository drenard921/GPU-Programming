#include <cuda_runtime.h>
#include <cstdio>

int main() {
    float* d = nullptr;
    cudaError_t err = cudaMallocManaged(&d, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::fprintf(stderr, "cudaMallocManaged succeeded\n");
    cudaFree(d);
    return 0;
}
