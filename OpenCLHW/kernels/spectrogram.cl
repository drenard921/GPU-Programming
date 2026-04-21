__kernel void spectrogram_kernel(
    __global const float* samples,
    __global float* output,
    const int num_samples,
    const int window_size,
    const int hop_size,
    const int num_bins,
    const int num_frames
) {
    const int frame = get_global_id(0);
    const int bin = get_global_id(1);

    if (frame >= num_frames || bin >= num_bins) {
        return;
    }

    const int start = frame * hop_size;
    if (start + window_size > num_samples) {
        return;
    }

    const float pi = 3.14159265358979323846f;
    float real = 0.0f;
    float imag = 0.0f;

    for (int n = 0; n < window_size; ++n) {
        float w = 1.0f;
        if (window_size > 1) {
            w = 0.5f - 0.5f *
                cos((2.0f * pi * (float)n) /
                    (float)(window_size - 1));
        }

        float sample = samples[start + n] * w;
        float angle =
            (2.0f * pi * (float)bin * (float)n) /
            (float)window_size;

        real += sample * cos(angle);
        imag -= sample * sin(angle);
    }

    float power = real * real + imag * imag;
    power /= (float)(window_size * window_size);

    output[frame * num_bins + bin] = power;
}