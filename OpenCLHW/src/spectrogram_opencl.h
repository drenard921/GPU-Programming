#ifndef SPECTROGRAM_OPENCL_H
#define SPECTROGRAM_OPENCL_H

#include <string>
#include <utility>
#include <vector>

#include "spectrogram_cpu.h"

enum class OpenCLKernelMode {
    Naive,
    Local
};

SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins,
    double* gpu_time_ms = nullptr,
    OpenCLKernelMode kernel_mode = OpenCLKernelMode::Local
);

std::vector<std::pair<std::string, SpectrogramResult>>
compute_spectrogram_opencl_batch(
    const std::vector<std::pair<std::string, std::vector<float>>>& batch_inputs,
    int window_size,
    int hop_size,
    int num_bins,
    std::vector<double>* gpu_times_ms = nullptr,
    OpenCLKernelMode kernel_mode = OpenCLKernelMode::Local
);

#endif