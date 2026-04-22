/*
 * spectrogram_cpu.h
 *
 * Declarations for the CPU-based spectrogram implementation.
 *
 * This header defines the SpectrogramResult structure used to store
 * spectrogram dimensions and power values, along with the function
 * that computes a spectrogram directly on the CPU. The CPU version
 * serves as both a correctness reference and a performance baseline
 * for comparison against the OpenCL implementation.
 */

#ifndef SPECTROGRAM_CPU_H
#define SPECTROGRAM_CPU_H

#include <vector>

/*
 * Stores the computed spectrogram as a flattened 2D array of power values.
 * values is laid out row-major as consecutive frames, each containing num_bins values.
 */
struct SpectrogramResult {
    int num_frames = 0;
    int num_bins = 0;
    std::vector<float> values;
};

/*
 * Computes a spectrogram from input audio samples on the CPU using
 * overlapping windows and direct frequency-bin power calculation.
 */
SpectrogramResult compute_spectrogram_cpu(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
);

#endif