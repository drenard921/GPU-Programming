/*
 * spectrogram_cpu.cpp
 *
 * CPU-based spectrogram computation used as a correctness reference and
 * performance baseline for the OpenCL implementation.
 *
 * This file computes a spectrogram directly from audio samples by dividing
 * the signal into overlapping frames, applying a Hann window to each frame,
 * and then evaluating a direct Discrete Fourier Transform (DFT) for each
 * requested frequency bin. The resulting power values are stored in a
 * flattened frame-by-bin output buffer.
 */

#include "spectrogram_cpu.h"

#include <cmath>
#include <vector>

namespace {
/* Constant value of pi used in Hann window and DFT angle calculations. */
constexpr float kPi = 3.14159265358979323846f;

/*
 * Computes the Hann window coefficient for a given sample position within
 * a frame. The Hann window reduces spectral leakage before the DFT step.
 */
float hann_window(int n, int window_size) {
    if (window_size <= 1) {
        return 1.0f;
    }
    return 0.5f - 0.5f * std::cos((2.0f * kPi * n) /
                                  (window_size - 1));
}
}  // namespace

/*
 * Computes a spectrogram on the CPU by:
 *   1. Splitting the signal into overlapping frames,
 *   2. Applying a Hann window to each frame,
 *   3. Computing direct DFT-based power values for each frequency bin.
 *
 * The output is stored as a flattened 2D array in row-major order:
 * consecutive frames, each containing num_bins power values.
 */
SpectrogramResult compute_spectrogram_cpu(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
) {
    SpectrogramResult result{};

    /* Reject invalid runtime parameters. */
    if (window_size <= 0 || hop_size <= 0 || num_bins <= 0) {
        return result;
    }

    /* If the signal is shorter than one window, no frames can be formed. */
    if (static_cast<int>(samples.size()) < window_size) {
        return result;
    }

    /* Compute the number of valid overlapping frames in the input signal. */
    result.num_frames =
        1 + static_cast<int>((samples.size() - window_size) / hop_size);
    result.num_bins = num_bins;
    result.values.resize(result.num_frames * result.num_bins, 0.0f);

    /* Normalize power by window_size^2 for scale consistency. */
    const float norm_factor =
        static_cast<float>(window_size * window_size);

    /* Process each frame independently. */
    for (int frame = 0; frame < result.num_frames; ++frame) {
        const int start = frame * hop_size;

        /* Compute the spectral power for each requested frequency bin. */
        for (int bin = 0; bin < result.num_bins; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;

            /*
             * Evaluate the direct DFT for this frame/bin pair.
             * Each sample is windowed before contributing to the sum.
             */
            for (int n = 0; n < window_size; ++n) {
                const float windowed_sample =
                    samples[start + n] * hann_window(n, window_size);

                const float angle =
                    2.0f * kPi * bin * n / window_size;

                real += windowed_sample * std::cos(angle);
                imag -= windowed_sample * std::sin(angle);
            }

            /* Convert the complex DFT result into normalized power. */
            float power = real * real + imag * imag;
            power /= norm_factor;

            result.values[frame * result.num_bins + bin] = power;
        }
    }

    return result;
}