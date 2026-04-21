#ifndef SPECTROGRAM_OPENCL_H
#define SPECTROGRAM_OPENCL_H

#include <vector>
#include "spectrogram_cpu.h"

SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
);

#endif