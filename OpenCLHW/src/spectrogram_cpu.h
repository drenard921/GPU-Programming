#ifndef SPECTROGRAM_CPU_H
#define SPECTROGRAM_CPU_H

#include <vector>

struct SpectrogramResult {
    int num_frames = 0;
    int num_bins = 0;
    std::vector<float> values;
};

SpectrogramResult compute_spectrogram_cpu(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
);

#endif