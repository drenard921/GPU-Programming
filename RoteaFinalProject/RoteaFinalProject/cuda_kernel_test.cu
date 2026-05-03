#include <cuda_runtime.h>
#include <cstdio>

__global__ void tiny_kernel(float* x) {
    int i = threadIdx.x;
    if (i == 0) x[0] += 1.0f;
}

int main() {
    float* x = nullptr;

    std::fprintf(stderr, "allocating managed memory\n");
    cudaError_t err = cudaMallocManaged(&x, sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    x[0] = 1.0f;

    std::fprintf(stderr, "launching kernel\n");
    tiny_kernel<<<1, 32>>>(x);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(x);
        return 1;
    }

    std::fprintf(stderr, "result: %f\n", x[0]);
    cudaFree(x);
    return 0;
}
