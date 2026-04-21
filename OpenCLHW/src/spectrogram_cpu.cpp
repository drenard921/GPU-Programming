#include "spectrogram_cpu.h"

#include <cmath>
#include <vector>

namespace {
constexpr float kPi = 3.14159265358979323846f;

float hann_window(int n, int window_size) {
    if (window_size <= 1) {
        return 1.0f;
    }
    return 0.5f - 0.5f * std::cos((2.0f * kPi * n) /
                                  (window_size - 1));
}
}  // namespace

SpectrogramResult compute_spectrogram_cpu(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
) {
    SpectrogramResult result{};

    if (window_size <= 0 || hop_size <= 0 || num_bins <= 0) {
        return result;
    }

    if (static_cast<int>(samples.size()) < window_size) {
        return result;
    }

    result.num_frames =
        1 + static_cast<int>((samples.size() - window_size) / hop_size);
    result.num_bins = num_bins;
    result.values.resize(result.num_frames * result.num_bins, 0.0f);

    const float norm_factor =
        static_cast<float>(window_size * window_size);

    for (int frame = 0; frame < result.num_frames; ++frame) {
        const int start = frame * hop_size;

        for (int bin = 0; bin < result.num_bins; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;

            for (int n = 0; n < window_size; ++n) {
                const float windowed_sample =
                    samples[start + n] * hann_window(n, window_size);

                const float angle =
                    2.0f * kPi * bin * n / window_size;

                real += windowed_sample * std::cos(angle);
                imag -= windowed_sample * std::sin(angle);
            }

            float power = real * real + imag * imag;
            power /= norm_factor;

            result.values[frame * result.num_bins + bin] = power;
        }
    }

    return result;
}