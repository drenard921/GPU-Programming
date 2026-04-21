__kernel void window_kernel(
    __global const float* samples,
    __global float* windowed_frames,
    const int num_samples,
    const int window_size,
    const int hop_size,
    const int num_frames
) {
    const int frame = get_global_id(0);
    const int n = get_global_id(1);

    if (frame >= num_frames || n >= window_size) {
        return;
    }

    const int start = frame * hop_size;
    const int sample_index = start + n;

    if (sample_index >= num_samples) {
        return;
    }

    float w = 1.0f;
    if (window_size > 1) {
        const float pi = 3.14159265358979323846f;
        w = 0.5f - 0.5f *
            cos((2.0f * pi * (float)n) /
                (float)(window_size - 1));
    }

    windowed_frames[frame * window_size + n] =
        samples[sample_index] * w;
}

__kernel void dft_power_kernel(
    __global const float* windowed_frames,
    __global float* output,
    __local float* local_frame,
    const int window_size,
    const int num_bins,
    const int num_frames
) {
    const int frame = get_global_id(0);
    const int bin = get_global_id(1);
    const int local_bin = get_local_id(1);
    const int local_size = get_local_size(1);

    if (frame >= num_frames || bin >= num_bins) {
        return;
    }

    const int frame_offset = frame * window_size;

    for (int n = local_bin; n < window_size; n += local_size) {
        local_frame[n] = windowed_frames[frame_offset + n];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float pi = 3.14159265358979323846f;
    float2 accum = (float2)(0.0f, 0.0f);

    for (int n = 0; n < window_size; ++n) {
        const float sample = local_frame[n];
        const float angle =
            (2.0f * pi * (float)bin * (float)n) /
            (float)window_size;

        accum.x += sample * cos(angle);
        accum.y -= sample * sin(angle);
    }

    float power = accum.x * accum.x + accum.y * accum.y;
    power /= (float)(window_size * window_size);

    output[frame * num_bins + bin] = power;
}