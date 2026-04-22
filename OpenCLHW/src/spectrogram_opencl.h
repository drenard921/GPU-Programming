/*
 * spectrogram_opencl.h
 *
 * Declarations for the OpenCL-based spectrogram implementation.
 *
 * This header defines the kernel-mode options used by the OpenCL pipeline
 * and exposes functions for computing spectrograms on a single audio signal
 * or across a batch of named inputs. The OpenCL implementation serves as the
 * GPU-accelerated counterpart to the CPU baseline.
 */

#ifndef SPECTROGRAM_OPENCL_H
#define SPECTROGRAM_OPENCL_H

#include <string>
#include <utility>
#include <vector>

#include "spectrogram_cpu.h"

/*
 * Selects which DFT kernel variant the OpenCL pipeline should use.
 * Naive reads frame data directly from global memory, while Local
 * uses local-memory staging when device capacity allows.
 */
enum class OpenCLKernelMode {
    Naive,
    Local
};

/*
 * Computes a spectrogram for one audio signal using the OpenCL pipeline.
 * Optionally returns the summed GPU kernel time in milliseconds.
 */
SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins,
    double* gpu_time_ms = nullptr,
    OpenCLKernelMode kernel_mode = OpenCLKernelMode::Local
);

/*
 * Computes spectrograms for multiple named audio inputs using a shared
 * OpenCL context. Optionally returns one GPU timing value per batch item.
 */
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