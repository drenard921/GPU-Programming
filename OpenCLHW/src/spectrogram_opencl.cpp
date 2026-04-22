/*
 * spectrogram_opencl.cpp
 *
 * OpenCL-based spectrogram computation for both single-input and batch modes.
 *
 * This file manages the host-side GPU execution flow for the spectrogram
 * application. It initializes OpenCL resources, creates kernel objects,
 * allocates device buffers, sets kernel arguments, launches the windowing
 * and DFT kernels, reads results back to the host, and records kernel timing
 * through OpenCL profiling events.
 *
 * Supported execution paths:
 *   1. compute_spectrogram_opencl
 *      Runs the OpenCL spectrogram pipeline for one audio signal.
 *
 *   2. compute_spectrogram_opencl_batch
 *      Runs the same pipeline across multiple audio inputs and stores each
 *      result separately while reusing a shared OpenCL context.
 *
 * Notes:
 *   - The OpenCL implementation supports both naive and local-memory DFT
 *     kernels.
 *   - The local kernel may fall back to the naive kernel if the device does
 *     not report enough local memory for the requested window size.
 *   - Timing is collected with OpenCL events so kernel execution cost can be
 *     compared against the CPU baseline.
 */

#include "spectrogram_opencl.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "opencl_utils.h"
#include "spectrogram_cpu.h"

namespace {

/* Stores per-input output layout information for batch OpenCL execution. */
struct BatchItemInfo {
    int num_frames = 0;
    int num_bins = 0;
    size_t value_count = 0;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

/*
 * Stores OpenCL buffers and profiling events associated with one batch item
 * so they can be released cleanly after execution.
 */
struct BatchRunResources {
    cl_mem samples_buffer = nullptr;
    cl_mem windowed_frames_buffer = nullptr;
    cl_mem output_subbuffer = nullptr;
    cl_event window_event = nullptr;
    cl_event dft_event = nullptr;
    cl_event read_event = nullptr;
};

/*
 * Stores device buffers and profiling events for one single-file OpenCL run.
 * Grouping these together keeps allocation and cleanup logic compact.
 */
struct SingleRunResources {
    cl_mem samples_buffer = nullptr;
    cl_mem windowed_frames_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    cl_event window_event = nullptr;
    cl_event dft_event = nullptr;
};

/*
 * Stores local and global NDRange sizes for the two-stage OpenCL pipeline.
 * The window kernel uses one frame dimension and one sample dimension, while
 * the DFT kernel uses one frame dimension and one frequency-bin dimension.
 */
struct KernelLaunchConfig {
    size_t window_local[2] = {1, 64};
    size_t window_global[2] = {0, 0};
    size_t dft_local[2] = {1, 16};
    size_t dft_global[2] = {0, 0};
};

/* Rounds value up to the nearest multiple used for global work sizes. */
size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }

    const size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }

    return value + multiple - remainder;
}

/* Returns the elapsed execution time of a profiling event in milliseconds. */
double get_event_time_ms(cl_event event) {
    if (event == nullptr) {
        return 0.0;
    }

    cl_ulong start = 0;
    cl_ulong end = 0;

    const cl_int err1 = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        nullptr
    );

    const cl_int err2 = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        nullptr
    );

    if (err1 != CL_SUCCESS || err2 != CL_SUCCESS || end < start) {
        return 0.0;
    }

    return static_cast<double>(end - start) / 1.0e6;
}

/* Prints a labeled event timing value when profiling information is available. */
void print_event_time_ms(
    cl_event event,
    const std::string& label
) {
    const double ms = get_event_time_ms(event);
    if (ms > 0.0) {
        std::cout << label << ": " << ms << " ms\n";
    }
}

/*
 * Creates the OpenCL kernel handles needed by the spectrogram pipeline:
 * one windowing kernel and two DFT kernel variants.
 */
bool create_kernels(
    const OpenCLContext& cl_ctx,
    cl_kernel& window_kernel,
    cl_kernel& dft_kernel_naive,
    cl_kernel& dft_kernel_local
) {
    cl_int err = CL_SUCCESS;

    window_kernel = clCreateKernel(
        cl_ctx.program,
        "window_kernel",
        &err
    );
    if (!check_opencl_error(err, "Failed to create window kernel")) {
        return false;
    }

    dft_kernel_naive = clCreateKernel(
        cl_ctx.program,
        "dft_power_kernel_naive",
        &err
    );
    if (!check_opencl_error(err, "Failed to create naive DFT kernel")) {
        clReleaseKernel(window_kernel);
        window_kernel = nullptr;
        return false;
    }

    dft_kernel_local = clCreateKernel(
        cl_ctx.program,
        "dft_power_kernel_local",
        &err
    );
    if (!check_opencl_error(err, "Failed to create local DFT kernel")) {
        clReleaseKernel(dft_kernel_naive);
        dft_kernel_naive = nullptr;
        clReleaseKernel(window_kernel);
        window_kernel = nullptr;
        return false;
    }

    return true;
}

/* Releases all kernel handles created for the spectrogram pipeline. */
void release_kernel_set(
    cl_kernel& window_kernel,
    cl_kernel& dft_kernel_naive,
    cl_kernel& dft_kernel_local
) {
    if (window_kernel != nullptr) {
        clReleaseKernel(window_kernel);
        window_kernel = nullptr;
    }

    if (dft_kernel_naive != nullptr) {
        clReleaseKernel(dft_kernel_naive);
        dft_kernel_naive = nullptr;
    }

    if (dft_kernel_local != nullptr) {
        clReleaseKernel(dft_kernel_local);
        dft_kernel_local = nullptr;
    }
}

/*
 * Resolves which DFT kernel should actually run.
 * If local-memory mode is requested but the device does not have enough
 * local memory for the frame size, the implementation falls back to
 * the naive kernel for correctness and portability.
 */
OpenCLKernelMode resolve_kernel_mode(
    cl_device_id device,
    int window_size,
    OpenCLKernelMode requested_mode
) {
    if (requested_mode == OpenCLKernelMode::Naive) {
        return OpenCLKernelMode::Naive;
    }

    cl_ulong local_mem_size = 0;
    const cl_int err = clGetDeviceInfo(
        device,
        CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(cl_ulong),
        &local_mem_size,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Warning: could not query local memory size. "
            << "Falling back to naive DFT kernel.\n";
        return OpenCLKernelMode::Naive;
    }

    const size_t bytes_needed =
        sizeof(float) * static_cast<size_t>(window_size);

    if (bytes_needed > static_cast<size_t>(local_mem_size)) {
        std::cerr
            << "Warning: local DFT kernel needs "
            << bytes_needed
            << " bytes of local memory, but device only reports "
            << static_cast<size_t>(local_mem_size)
            << ". Falling back to naive DFT kernel.\n";
        return OpenCLKernelMode::Naive;
    }

    return OpenCLKernelMode::Local;
}

/* Sets all arguments required by the windowing kernel. */
bool set_window_kernel_args(
    cl_kernel window_kernel,
    cl_mem samples_buffer,
    cl_mem windowed_frames_buffer,
    int num_samples,
    int window_size,
    int hop_size,
    int num_frames
) {
    cl_int err = CL_SUCCESS;

    err  = clSetKernelArg(window_kernel, 0, sizeof(cl_mem), &samples_buffer);
    err |= clSetKernelArg(
        window_kernel,
        1,
        sizeof(cl_mem),
        &windowed_frames_buffer
    );
    err |= clSetKernelArg(window_kernel, 2, sizeof(int), &num_samples);
    err |= clSetKernelArg(window_kernel, 3, sizeof(int), &window_size);
    err |= clSetKernelArg(window_kernel, 4, sizeof(int), &hop_size);
    err |= clSetKernelArg(window_kernel, 5, sizeof(int), &num_frames);

    return check_opencl_error(err, "Failed to set window kernel arguments");
}

/*
 * Sets arguments for either the naive or local-memory DFT kernel,
 * accounting for the different parameter layouts of the two kernels.
 */
bool set_dft_kernel_args(
    cl_kernel dft_kernel,
    OpenCLKernelMode kernel_mode,
    cl_mem windowed_frames_buffer,
    cl_mem output_buffer,
    int window_size,
    int num_bins,
    int num_frames
) {
    cl_int err = CL_SUCCESS;

    if (kernel_mode == OpenCLKernelMode::Local) {
        err  = clSetKernelArg(
            dft_kernel,
            0,
            sizeof(cl_mem),
            &windowed_frames_buffer
        );
        err |= clSetKernelArg(dft_kernel, 1, sizeof(cl_mem), &output_buffer);
        err |= clSetKernelArg(
            dft_kernel,
            2,
            sizeof(float) * static_cast<size_t>(window_size),
            nullptr
        );
        err |= clSetKernelArg(dft_kernel, 3, sizeof(int), &window_size);
        err |= clSetKernelArg(dft_kernel, 4, sizeof(int), &num_bins);
        err |= clSetKernelArg(dft_kernel, 5, sizeof(int), &num_frames);

        return check_opencl_error(
            err,
            "Failed to set local DFT kernel arguments"
        );
    }

    err  = clSetKernelArg(
        dft_kernel,
        0,
        sizeof(cl_mem),
        &windowed_frames_buffer
    );
    err |= clSetKernelArg(dft_kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(dft_kernel, 2, sizeof(int), &window_size);
    err |= clSetKernelArg(dft_kernel, 3, sizeof(int), &num_bins);
    err |= clSetKernelArg(dft_kernel, 4, sizeof(int), &num_frames);

    return check_opencl_error(
        err,
        "Failed to set naive DFT kernel arguments"
    );
}

/* Releases buffers and events created for one single-file OpenCL run. */
void release_single_resources(SingleRunResources& res) {
    if (res.dft_event != nullptr) {
        clReleaseEvent(res.dft_event);
        res.dft_event = nullptr;
    }

    if (res.window_event != nullptr) {
        clReleaseEvent(res.window_event);
        res.window_event = nullptr;
    }

    if (res.output_buffer != nullptr) {
        clReleaseMemObject(res.output_buffer);
        res.output_buffer = nullptr;
    }

    if (res.windowed_frames_buffer != nullptr) {
        clReleaseMemObject(res.windowed_frames_buffer);
        res.windowed_frames_buffer = nullptr;
    }

    if (res.samples_buffer != nullptr) {
        clReleaseMemObject(res.samples_buffer);
        res.samples_buffer = nullptr;
    }
}

/*
 * Releases all per-item OpenCL buffers and profiling events created during
 * batch execution.
 */
void release_batch_resources(std::vector<BatchRunResources>& resources) {
    for (auto& res : resources) {
        if (res.read_event != nullptr) {
            clReleaseEvent(res.read_event);
            res.read_event = nullptr;
        }

        if (res.dft_event != nullptr) {
            clReleaseEvent(res.dft_event);
            res.dft_event = nullptr;
        }

        if (res.window_event != nullptr) {
            clReleaseEvent(res.window_event);
            res.window_event = nullptr;
        }

        if (res.output_subbuffer != nullptr) {
            clReleaseMemObject(res.output_subbuffer);
            res.output_subbuffer = nullptr;
        }

        if (res.windowed_frames_buffer != nullptr) {
            clReleaseMemObject(res.windowed_frames_buffer);
            res.windowed_frames_buffer = nullptr;
        }

        if (res.samples_buffer != nullptr) {
            clReleaseMemObject(res.samples_buffer);
            res.samples_buffer = nullptr;
        }
    }
}

/*
 * Validates single-run parameters, initializes the output container, and
 * resets the optional GPU timing output.
 */
bool initialize_single_result(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins,
    SpectrogramResult& result,
    double* gpu_time_ms
) {
    result = SpectrogramResult{};

    if (gpu_time_ms != nullptr) {
        *gpu_time_ms = 0.0;
    }

    if (window_size <= 0 || hop_size <= 0 || num_bins <= 0) {
        std::cerr << "Invalid OpenCL spectrogram parameters.\n";
        return false;
    }

    if (static_cast<int>(samples.size()) < window_size) {
        std::cerr << "Not enough samples for OpenCL spectrogram.\n";
        return false;
    }

    result.num_frames =
        1 + static_cast<int>((samples.size() - window_size) / hop_size);
    result.num_bins = num_bins;
    result.values.resize(
        static_cast<size_t>(result.num_frames) *
            static_cast<size_t>(result.num_bins),
        0.0f
    );

    return true;
}

/* Allocates the device buffers needed for one spectrogram run. */
bool create_single_buffers(
    const OpenCLContext& cl_ctx,
    const std::vector<float>& samples,
    const SpectrogramResult& result,
    int window_size,
    SingleRunResources& res
) {
    cl_int err = CL_SUCCESS;

    res.samples_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * samples.size(),
        const_cast<float*>(samples.data()),
        &err
    );
    if (!check_opencl_error(err, "Failed to create samples buffer")) {
        return false;
    }

    res.windowed_frames_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_WRITE,
        sizeof(float) * static_cast<size_t>(result.num_frames) *
            static_cast<size_t>(window_size),
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create windowed frames buffer")) {
        return false;
    }

    res.output_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * result.values.size(),
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create output buffer")) {
        return false;
    }

    return true;
}

/*
 * Configures both pipeline stages for the single-run path:
 * the window kernel and whichever DFT kernel was selected.
 */
bool set_single_kernel_args(
    cl_kernel window_kernel,
    cl_kernel dft_kernel,
    OpenCLKernelMode actual_mode,
    const SingleRunResources& res,
    int num_samples,
    int window_size,
    int hop_size,
    int num_bins,
    int num_frames
) {
    if (!set_window_kernel_args(
            window_kernel,
            res.samples_buffer,
            res.windowed_frames_buffer,
            num_samples,
            window_size,
            hop_size,
            num_frames)) {
        return false;
    }

    return set_dft_kernel_args(
        dft_kernel,
        actual_mode,
        res.windowed_frames_buffer,
        res.output_buffer,
        window_size,
        num_bins,
        num_frames
    );
}

/* Computes local and global launch sizes for the two-stage pipeline. */
KernelLaunchConfig make_launch_config(
    int num_frames,
    int window_size,
    int num_bins
) {
    KernelLaunchConfig cfg{};

    cfg.window_global[0] = static_cast<size_t>(num_frames);
    cfg.window_global[1] = round_up(
        static_cast<size_t>(window_size),
        cfg.window_local[1]
    );

    cfg.dft_global[0] = static_cast<size_t>(num_frames);
    cfg.dft_global[1] = round_up(
        static_cast<size_t>(num_bins),
        cfg.dft_local[1]
    );

    return cfg;
}

/*
 * Enqueues the window kernel first, then enqueues the DFT kernel with an
 * event dependency so the second stage only begins after windowing finishes.
 */
bool enqueue_single_pipeline(
    const OpenCLContext& cl_ctx,
    cl_kernel window_kernel,
    cl_kernel dft_kernel,
    const KernelLaunchConfig& cfg,
    SingleRunResources& res
) {
    cl_int err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        window_kernel,
        2,
        nullptr,
        cfg.window_global,
        cfg.window_local,
        0,
        nullptr,
        &res.window_event
    );
    if (!check_opencl_error(err, "Failed to enqueue window kernel")) {
        return false;
    }

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        dft_kernel,
        2,
        nullptr,
        cfg.dft_global,
        cfg.dft_local,
        1,
        &res.window_event,
        &res.dft_event
    );
    if (!check_opencl_error(err, "Failed to enqueue DFT kernel")) {
        return false;
    }

    return true;
}

/*
 * Waits for queued work to finish, reports profiling times, and reads the
 * computed spectrogram values back to host memory.
 */
bool finalize_single_run(
    const OpenCLContext& cl_ctx,
    const SingleRunResources& res,
    OpenCLKernelMode actual_mode,
    SpectrogramResult& result,
    double* gpu_time_ms
) {
    cl_int err = clFinish(cl_ctx.queue);
    if (!check_opencl_error(err, "Failed to finish OpenCL queue")) {
        return false;
    }

    print_event_time_ms(res.window_event, "OpenCL window kernel");
    print_event_time_ms(
        res.dft_event,
        actual_mode == OpenCLKernelMode::Local
            ? "OpenCL DFT kernel (local)"
            : "OpenCL DFT kernel (naive)"
    );

    if (gpu_time_ms != nullptr) {
        *gpu_time_ms =
            get_event_time_ms(res.window_event) +
            get_event_time_ms(res.dft_event);
    }

    err = clEnqueueReadBuffer(
        cl_ctx.queue,
        res.output_buffer,
        CL_TRUE,
        0,
        sizeof(float) * result.values.size(),
        result.values.data(),
        0,
        nullptr,
        nullptr
    );

    return check_opencl_error(err, "Failed to read OpenCL output buffer");
}

/*
 * Precomputes the output layout for batch mode.
 * Each valid input is assigned a region inside the parent output buffer, and
 * each corresponding SpectrogramResult is sized in advance.
 */
void initialize_batch_layout(
    const std::vector<std::pair<std::string, std::vector<float>>>& batch_inputs,
    int window_size,
    int hop_size,
    int num_bins,
    std::vector<std::pair<std::string, SpectrogramResult>>& results,
    std::vector<BatchItemInfo>& infos,
    size_t& total_output_bytes
) {
    results.resize(batch_inputs.size());
    infos.resize(batch_inputs.size());
    total_output_bytes = 0;

    for (size_t i = 0; i < batch_inputs.size(); ++i) {
        const auto& name = batch_inputs[i].first;
        const auto& samples = batch_inputs[i].second;

        results[i].first = name;

        if (static_cast<int>(samples.size()) < window_size) {
            continue;
        }

        infos[i].num_frames =
            1 + static_cast<int>((samples.size() - window_size) / hop_size);
        infos[i].num_bins = num_bins;
        infos[i].value_count =
            static_cast<size_t>(infos[i].num_frames) *
            static_cast<size_t>(infos[i].num_bins);
        infos[i].byte_offset = total_output_bytes;
        infos[i].byte_size = infos[i].value_count * sizeof(float);

        total_output_bytes += infos[i].byte_size;

        results[i].second.num_frames = infos[i].num_frames;
        results[i].second.num_bins = num_bins;
        results[i].second.values.resize(infos[i].value_count, 0.0f);
    }
}

/*
 * Creates the shared parent output buffer used in batch mode.
 * Each input's spectrogram output is exposed through a sub-buffer region of
 * this larger allocation.
 */
cl_mem create_parent_output_buffer(
    const OpenCLContext& cl_ctx,
    size_t total_output_bytes
) {
    cl_int err = CL_SUCCESS;

    return clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        total_output_bytes,
        nullptr,
        &err
    );
}

/*
 * Creates per-item buffers, sub-buffer output view, and kernel arguments for
 * one batch input, then enqueues both pipeline stages for that item.
 */
bool enqueue_batch_item(
    const OpenCLContext& cl_ctx,
    cl_kernel window_kernel,
    cl_kernel dft_kernel,
    OpenCLKernelMode actual_mode,
    cl_mem parent_output_buffer,
    const std::vector<float>& samples,
    const BatchItemInfo& info,
    SpectrogramResult& result,
    int window_size,
    int hop_size,
    int num_bins,
    BatchRunResources& res
) {
    cl_int err = CL_SUCCESS;
    const int num_samples = static_cast<int>(samples.size());

    res.samples_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * samples.size(),
        const_cast<float*>(samples.data()),
        &err
    );
    if (!check_opencl_error(err, "Failed to create batch samples buffer")) {
        return false;
    }

    res.windowed_frames_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_WRITE,
        sizeof(float) * static_cast<size_t>(result.num_frames) *
            static_cast<size_t>(window_size),
        nullptr,
        &err
    );
    if (!check_opencl_error(
            err,
            "Failed to create batch windowed frames buffer")) {
        return false;
    }

    cl_buffer_region region{};
    region.origin = info.byte_offset;
    region.size = info.byte_size;

    res.output_subbuffer = clCreateSubBuffer(
        parent_output_buffer,
        CL_MEM_WRITE_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &region,
        &err
    );
    if (!check_opencl_error(err, "Failed to create output sub-buffer")) {
        return false;
    }

    if (!set_window_kernel_args(
            window_kernel,
            res.samples_buffer,
            res.windowed_frames_buffer,
            num_samples,
            window_size,
            hop_size,
            result.num_frames)) {
        return false;
    }

    if (!set_dft_kernel_args(
            dft_kernel,
            actual_mode,
            res.windowed_frames_buffer,
            res.output_subbuffer,
            window_size,
            num_bins,
            result.num_frames)) {
        return false;
    }

    const KernelLaunchConfig cfg = make_launch_config(
        result.num_frames,
        window_size,
        num_bins
    );

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        window_kernel,
        2,
        nullptr,
        cfg.window_global,
        cfg.window_local,
        0,
        nullptr,
        &res.window_event
    );
    if (!check_opencl_error(err, "Failed to enqueue batch window kernel")) {
        return false;
    }

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        dft_kernel,
        2,
        nullptr,
        cfg.dft_global,
        cfg.dft_local,
        1,
        &res.window_event,
        &res.dft_event
    );
    if (!check_opencl_error(err, "Failed to enqueue batch DFT kernel")) {
        return false;
    }

    return true;
}

/*
 * Enqueues asynchronous readbacks for every valid batch item after its DFT
 * event completes.
 */
bool enqueue_batch_reads(
    const OpenCLContext& cl_ctx,
    const std::vector<BatchItemInfo>& infos,
    std::vector<std::pair<std::string, SpectrogramResult>>& results,
    std::vector<BatchRunResources>& resources
) {
    for (size_t i = 0; i < resources.size(); ++i) {
        const auto& info = infos[i];
        auto& result = results[i].second;
        auto& res = resources[i];

        if (result.values.empty() || res.output_subbuffer == nullptr) {
            continue;
        }

        const cl_int err = clEnqueueReadBuffer(
            cl_ctx.queue,
            res.output_subbuffer,
            CL_FALSE,
            0,
            info.byte_size,
            result.values.data(),
            1,
            &res.dft_event,
            &res.read_event
        );

        if (!check_opencl_error(err, "Failed to enqueue batch read buffer")) {
            return false;
        }
    }

    return true;
}

/*
 * Collects and reports batch kernel timings. If requested, one timing value
 * per batch item is stored in gpu_times_ms.
 */
void collect_batch_timings(
    const std::vector<BatchRunResources>& resources,
    OpenCLKernelMode actual_mode,
    std::vector<double>* gpu_times_ms
) {
    double total_window_kernel_ms = 0.0;
    double total_dft_kernel_ms = 0.0;

    if (gpu_times_ms != nullptr) {
        gpu_times_ms->assign(resources.size(), 0.0);
    }

    for (size_t i = 0; i < resources.size(); ++i) {
        const auto& res = resources[i];

        const double window_ms = get_event_time_ms(res.window_event);
        const double dft_ms = get_event_time_ms(res.dft_event);

        total_window_kernel_ms += window_ms;
        total_dft_kernel_ms += dft_ms;

        if (gpu_times_ms != nullptr) {
            (*gpu_times_ms)[i] = window_ms + dft_ms;
        }
    }

    std::cout << "OpenCL batch window kernel total: "
              << total_window_kernel_ms << " ms\n";
    std::cout << "OpenCL batch DFT kernel total ("
              << (actual_mode == OpenCLKernelMode::Local ? "local" : "naive")
              << "): "
              << total_dft_kernel_ms << " ms\n";
}

}  // namespace

/*
 * Computes a spectrogram for one audio signal using the OpenCL pipeline.
 *
 * Execution flow:
 *   1. Validate parameters and compute output dimensions.
 *   2. Initialize OpenCL and choose the DFT kernel variant.
 *   3. Create kernels and allocate device buffers.
 *   4. Launch the windowing kernel.
 *   5. Launch the DFT kernel after windowing completes.
 *   6. Wait for completion, collect profiling times, and read results back.
 *   7. Release all OpenCL resources before returning.
 *
 * The returned SpectrogramResult stores power values in row-major order:
 * consecutive frames, each containing num_bins values.
 */
SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins,
    double* gpu_time_ms,
    OpenCLKernelMode kernel_mode
) {
    SpectrogramResult result{};
    if (!initialize_single_result(
            samples,
            window_size,
            hop_size,
            num_bins,
            result,
            gpu_time_ms)) {
        return result;
    }

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL.\n";
        return {};
    }

    const OpenCLKernelMode actual_mode = resolve_kernel_mode(
        cl_ctx.device,
        window_size,
        kernel_mode
    );

    cl_kernel window_kernel = nullptr;
    cl_kernel dft_kernel_naive = nullptr;
    cl_kernel dft_kernel_local = nullptr;

    if (!create_kernels(
            cl_ctx,
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local)) {
        cleanup_opencl(cl_ctx);
        return {};
    }

    const cl_kernel selected_dft_kernel =
        (actual_mode == OpenCLKernelMode::Local)
            ? dft_kernel_local
            : dft_kernel_naive;

    SingleRunResources res{};
    if (!create_single_buffers(
            cl_ctx,
            samples,
            result,
            window_size,
            res)) {
        release_single_resources(res);
        release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
        cleanup_opencl(cl_ctx);
        return {};
    }

    if (!set_single_kernel_args(
            window_kernel,
            selected_dft_kernel,
            actual_mode,
            res,
            static_cast<int>(samples.size()),
            window_size,
            hop_size,
            num_bins,
            result.num_frames)) {
        release_single_resources(res);
        release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
        cleanup_opencl(cl_ctx);
        return {};
    }

    const KernelLaunchConfig cfg = make_launch_config(
        result.num_frames,
        window_size,
        num_bins
    );

    if (!enqueue_single_pipeline(
            cl_ctx,
            window_kernel,
            selected_dft_kernel,
            cfg,
            res)) {
        release_single_resources(res);
        release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
        cleanup_opencl(cl_ctx);
        return {};
    }

    if (!finalize_single_run(
            cl_ctx,
            res,
            actual_mode,
            result,
            gpu_time_ms)) {
        release_single_resources(res);
        release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
        cleanup_opencl(cl_ctx);
        return {};
    }

    release_single_resources(res);
    release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
    cleanup_opencl(cl_ctx);
    return result;
}

/*
 * Computes spectrograms for multiple audio signals using a shared OpenCL
 * context and one parent output buffer partitioned into sub-buffers.
 *
 * Each batch item still receives its own input and intermediate buffers,
 * but all outputs are packed into one parent device buffer for simpler
 * management. Profiling times are optionally returned per batch item.
 */
std::vector<std::pair<std::string, SpectrogramResult>>
compute_spectrogram_opencl_batch(
    const std::vector<std::pair<std::string, std::vector<float>>>& batch_inputs,
    int window_size,
    int hop_size,
    int num_bins,
    std::vector<double>* gpu_times_ms,
    OpenCLKernelMode kernel_mode
) {
    std::vector<std::pair<std::string, SpectrogramResult>> results;
    std::vector<BatchItemInfo> infos;
    std::vector<BatchRunResources> resources;
    size_t total_output_bytes = 0;

    if (gpu_times_ms != nullptr) {
        gpu_times_ms->clear();
    }

    if (window_size <= 0 || hop_size <= 0 || num_bins <= 0) {
        std::cerr << "Invalid OpenCL batch spectrogram parameters.\n";
        return results;
    }

    resources.resize(batch_inputs.size());
    initialize_batch_layout(
        batch_inputs,
        window_size,
        hop_size,
        num_bins,
        results,
        infos,
        total_output_bytes
    );

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL for batch mode.\n";
        return {};
    }

    const OpenCLKernelMode actual_mode = resolve_kernel_mode(
        cl_ctx.device,
        window_size,
        kernel_mode
    );

    cl_kernel window_kernel = nullptr;
    cl_kernel dft_kernel_naive = nullptr;
    cl_kernel dft_kernel_local = nullptr;

    if (!create_kernels(
            cl_ctx,
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local)) {
        cleanup_opencl(cl_ctx);
        return {};
    }

    const cl_kernel selected_dft_kernel =
        (actual_mode == OpenCLKernelMode::Local)
            ? dft_kernel_local
            : dft_kernel_naive;

    cl_int err = CL_SUCCESS;
    cl_mem parent_output_buffer = create_parent_output_buffer(
        cl_ctx,
        total_output_bytes
    );
    if (parent_output_buffer == nullptr) {
        err = CL_OUT_OF_RESOURCES;
    }

    if (!check_opencl_error(err, "Failed to create parent output buffer")) {
        release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
        cleanup_opencl(cl_ctx);
        return {};
    }

    bool enqueue_failed = false;

    /*
     * Create per-item buffers, bind the correct kernel arguments, and enqueue
     * windowing plus DFT work for each valid input in the batch.
     */
    for (size_t i = 0; i < batch_inputs.size(); ++i) {
        const auto& samples = batch_inputs[i].second;
        auto& result = results[i].second;
        auto& res = resources[i];

        if (result.values.empty()) {
            continue;
        }

        if (!enqueue_batch_item(
                cl_ctx,
                window_kernel,
                selected_dft_kernel,
                actual_mode,
                parent_output_buffer,
                samples,
                infos[i],
                result,
                window_size,
                hop_size,
                num_bins,
                res)) {
            enqueue_failed = true;
            break;
        }
    }

    if (!enqueue_failed) {
        enqueue_failed = !enqueue_batch_reads(
            cl_ctx,
            infos,
            results,
            resources
        );
    }

    if (!enqueue_failed) {
        err = clFinish(cl_ctx.queue);
        enqueue_failed = !check_opencl_error(err, "Failed to finish batch queue");
    }

    if (!enqueue_failed) {
        collect_batch_timings(resources, actual_mode, gpu_times_ms);
    }

    release_batch_resources(resources);

    if (parent_output_buffer != nullptr) {
        clReleaseMemObject(parent_output_buffer);
    }

    release_kernel_set(window_kernel, dft_kernel_naive, dft_kernel_local);
    cleanup_opencl(cl_ctx);

    if (enqueue_failed) {
        return {};
    }

    return results;
}