/*
 * spectrogram.cl
 *
 * OpenCL kernels for spectrogram generation from audio samples.
 *
 * This file implements the core GPU-side computation used to transform
 * a 1D audio signal into frame-by-frame spectral power values. The pipeline
 * is split into two stages: first, the input signal is divided into overlapping
 * frames and multiplied by a Hann window; second, a direct Discrete Fourier
 * Transform (DFT) is computed for each frame to produce power values for
 * each frequency bin.
 *
 * Included kernels:
 *   1. window_kernel
 *      Extracts overlapping frames from the input signal and applies
 *      a Hann window to reduce spectral leakage.
 *
 *   2. dft_power_kernel_local
 *      Computes the DFT power spectrum using local memory to stage
 *      frame data and reduce repeated global memory access.
 *
 *   3. dft_power_kernel_naive
 *      Computes the same DFT power spectrum directly from global memory.
 *      This serves as a simpler baseline for correctness checks and
 *      performance comparison.
 *
 * Data layout:
 *   - Input samples are stored as a 1D float buffer.
 *   - Windowed frames are stored in a flattened 2D layout:
 *       frame 0 samples, then frame 1 samples, and so on.
 *   - Output power values are also stored in a flattened 2D layout:
 *       frame 0 bins, then frame 1 bins, and so on.
 *
 * Notes:
 *   - The DFT implementation is intentionally direct rather than FFT-based
 *     in order to emphasize parallelization strategy and memory behavior.
 *   - The local-memory kernel assumes a launch configuration where
 *     local_size[0] == 1, so each work-group processes exactly one frame.
 */


/*
 * window_kernel
 *
 * Applies a Hann window to each analysis frame of the input audio signal.
 * The output is a flattened 2D buffer where each frame occupies a contiguous
 * block of length window_size.
 *
 * Global indexing:
 *   get_global_id(0) -> frame index
 *   get_global_id(1) -> sample position within that frame
 */
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

    /* Guard against out of bounds work-items. */
    if (frame >= num_frames || n >= window_size) {
        return;
    }

    /* Compute the starting sample index for this frame. */
    const int start = frame * hop_size;
    const int sample_index = start + n;

    /* Ignore work-items that map past the end of the input signal. */
    if (sample_index >= num_samples) {
        return;
    }

    /*
     * Compute the Hann window coefficient for this sample position.
     * If window_size == 1, fall back to a multiplier of 1.0.
     */
    float w = 1.0f;
    if (window_size > 1) {
        const float pi = 3.14159265358979323846f;
        w = 0.5f - 0.5f *
            cos((2.0f * pi * (float)n) /
                (float)(window_size - 1));
    }

    /* Store the windowed sample into the flattened frame buffer. */
    windowed_frames[frame * window_size + n] =
        samples[sample_index] * w;
}

/*
 * dft_power_kernel_local
 *
 * Computes the power spectrum for each frame/bin pair using a direct DFT.
 * This version first stages one frame into local memory so all work-items
 * in the work-group can reuse the same samples with lower global memory traffic.
 *
 * Global indexing:
 *   get_global_id(0) -> frame index
 *   get_global_id(1) -> frequency bin index
 *
 * Important launch assumption:
 *   local_size[0] == 1
 * so that each work-group processes exactly one frame.
 */
__kernel void dft_power_kernel_local(
    __global const float* windowed_frames,
    __global float* output,
    __local float* local_frame,
    const int window_size,
    const int num_bins,
    const int num_frames
) {
    const int frame = get_global_id(0);
    const int bin = get_global_id(1);

    const int local_frame_id = get_local_id(0);
    const int local_bin_id = get_local_id(1);
    const int local_bins = get_local_size(1);

    /* Guard against out of bounds work-items. */
    if (frame >= num_frames || bin >= num_bins) {
        return;
    }

    /*
     * This kernel assumes local_size[0] == 1 so that each work-group
     * handles exactly one frame. Work-items in local dimension 1
     * cooperate to stage that frame into local memory.
     */
    if (local_frame_id != 0) {
        return;
    }

    /* Compute the starting offset of this frame in the flattened input buffer. */
    const int frame_offset = frame * window_size;

    /*
     * Cooperatively copy the frame from global memory into local memory.
     * Each work-item loads multiple elements in a strided pattern if needed.
     */
    for (int n = local_bin_id; n < window_size; n += local_bins) {
        local_frame[n] = windowed_frames[frame_offset + n];
    }

    /* Ensure the full frame is available in local memory before DFT begins. */
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * Compute one DFT bin for this frame.
     * accum.x stores the real part, accum.y stores the imaginary part.
     */
    const float pi = 3.14159265358979323846f;
    float2 accum = (float2)(0.0f, 0.0f);

    for (int n = 0; n < window_size; ++n) {
        const float sample = local_frame[n];
        const float angle =
            (2.0f * pi * (float)bin * (float)n) /
            (float)window_size;

        const float c = cos(angle);
        const float s = sin(angle);

        /* DFT accumulation: real += x[n]cos(theta), imag -= x[n]sin(theta). */
        accum.x += sample * c;
        accum.y -= sample * s;
    }

    /*
     * Convert the complex DFT result into power.
     * dot(accum, accum) = real^2 + imag^2.
     * Normalize by window_size^2 for scale consistency.
     */
    float power = dot(accum, accum);
    power /= (float)(window_size * window_size);

    /* Store the power value for this frame/bin location. */
    output[frame * num_bins + bin] = power;
}

/*
 * dft_power_kernel_naive
 *
 * Computes the power spectrum for each frame/bin pair using a direct DFT.
 * This baseline version reads samples directly from global memory instead
 * of staging the frame into local memory.
 *
 * It is simpler, but may perform worse due to repeated global memory access.
 */
__kernel void dft_power_kernel_naive(
    __global const float* windowed_frames,
    __global float* output,
    const int window_size,
    const int num_bins,
    const int num_frames
) {
    const int frame = get_global_id(0);
    const int bin = get_global_id(1);

    /* Guard against out of bounds work-items. */
    if (frame >= num_frames || bin >= num_bins) {
        return;
    }

    /* Compute the starting offset of this frame in the flattened input buffer. */
    const int frame_offset = frame * window_size;
    const float pi = 3.14159265358979323846f;
    float2 accum = (float2)(0.0f, 0.0f);

    /*
     * Compute one DFT bin directly from global memory.
     * This kernel serves as a correctness and performance baseline.
     */
    for (int n = 0; n < window_size; ++n) {
        const float sample = windowed_frames[frame_offset + n];
        const float angle =
            (2.0f * pi * (float)bin * (float)n) /
            (float)window_size;

        const float c = cos(angle);
        const float s = sin(angle);

        accum.x += sample * c;
        accum.y -= sample * s;
    }

    /* Convert the complex DFT result into normalized power. */
    float power = dot(accum, accum);
    power /= (float)(window_size * window_size);

    /* Store the power value for this frame/bin location. */
    output[frame * num_bins + bin] = power;
}