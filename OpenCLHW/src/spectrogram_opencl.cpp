#include "spectrogram_opencl.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "opencl_utils.h"
#include "spectrogram_cpu.h"

namespace {

struct BatchItemInfo {
    int num_frames = 0;
    int num_bins = 0;
    size_t value_count = 0;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

struct BatchRunResources {
    cl_mem samples_buffer = nullptr;
    cl_mem windowed_frames_buffer = nullptr;
    cl_mem output_subbuffer = nullptr;
    cl_event window_event = nullptr;
    cl_event dft_event = nullptr;
    cl_event read_event = nullptr;
};

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

void print_event_time_ms(
    cl_event event,
    const std::string& label
) {
    const double ms = get_event_time_ms(event);
    if (ms > 0.0) {
        std::cout << label << ": " << ms << " ms\n";
    }
}

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

}  // namespace

SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins,
    double* gpu_time_ms,
    OpenCLKernelMode kernel_mode
) {
    SpectrogramResult result{};

    if (gpu_time_ms != nullptr) {
        *gpu_time_ms = 0.0;
    }

    if (window_size <= 0 || hop_size <= 0 || num_bins <= 0) {
        std::cerr << "Invalid OpenCL spectrogram parameters.\n";
        return result;
    }

    if (static_cast<int>(samples.size()) < window_size) {
        std::cerr << "Not enough samples for OpenCL spectrogram.\n";
        return result;
    }

    result.num_frames =
        1 + static_cast<int>((samples.size() - window_size) / hop_size);
    result.num_bins = num_bins;
    result.values.resize(
        static_cast<size_t>(result.num_frames) *
            static_cast<size_t>(result.num_bins),
        0.0f
    );

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL.\n";
        return {};
    }

    OpenCLKernelMode actual_mode = resolve_kernel_mode(
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

    cl_kernel selected_dft_kernel =
        (actual_mode == OpenCLKernelMode::Local)
            ? dft_kernel_local
            : dft_kernel_naive;

    cl_int err = CL_SUCCESS;
    const int num_samples = static_cast<int>(samples.size());

    cl_mem samples_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * samples.size(),
        const_cast<float*>(samples.data()),
        &err
    );
    if (!check_opencl_error(err, "Failed to create samples buffer")) {
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    cl_mem windowed_frames_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_WRITE,
        sizeof(float) * static_cast<size_t>(result.num_frames) *
            static_cast<size_t>(window_size),
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create windowed frames buffer")) {
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    cl_mem output_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * result.values.size(),
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create output buffer")) {
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    if (!set_window_kernel_args(
            window_kernel,
            samples_buffer,
            windowed_frames_buffer,
            num_samples,
            window_size,
            hop_size,
            result.num_frames)) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    if (!set_dft_kernel_args(
            selected_dft_kernel,
            actual_mode,
            windowed_frames_buffer,
            output_buffer,
            window_size,
            num_bins,
            result.num_frames)) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    const size_t window_local[2] = {1, 64};
    const size_t window_global[2] = {
        static_cast<size_t>(result.num_frames),
        round_up(static_cast<size_t>(window_size), window_local[1])
    };

    const size_t dft_local[2] = {1, 16};
    const size_t dft_global[2] = {
        static_cast<size_t>(result.num_frames),
        round_up(static_cast<size_t>(num_bins), dft_local[1])
    };

    cl_event window_event = nullptr;
    cl_event dft_event = nullptr;

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        window_kernel,
        2,
        nullptr,
        window_global,
        window_local,
        0,
        nullptr,
        &window_event
    );
    if (!check_opencl_error(err, "Failed to enqueue window kernel")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        selected_dft_kernel,
        2,
        nullptr,
        dft_global,
        dft_local,
        1,
        &window_event,
        &dft_event
    );
    if (!check_opencl_error(err, "Failed to enqueue DFT kernel")) {
        clReleaseEvent(window_event);
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    err = clFinish(cl_ctx.queue);
    if (!check_opencl_error(err, "Failed to finish OpenCL queue")) {
        clReleaseEvent(dft_event);
        clReleaseEvent(window_event);
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    print_event_time_ms(window_event, "OpenCL window kernel");
    print_event_time_ms(
        dft_event,
        actual_mode == OpenCLKernelMode::Local
            ? "OpenCL DFT kernel (local)"
            : "OpenCL DFT kernel (naive)"
    );

    const double total_gpu_kernel_ms =
        get_event_time_ms(window_event) + get_event_time_ms(dft_event);
    if (gpu_time_ms != nullptr) {
        *gpu_time_ms = total_gpu_kernel_ms;
    }

    err = clEnqueueReadBuffer(
        cl_ctx.queue,
        output_buffer,
        CL_TRUE,
        0,
        sizeof(float) * result.values.size(),
        result.values.data(),
        0,
        nullptr,
        nullptr
    );
    if (!check_opencl_error(err, "Failed to read OpenCL output buffer")) {
        clReleaseEvent(dft_event);
        clReleaseEvent(window_event);
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    clReleaseEvent(dft_event);
    clReleaseEvent(window_event);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(windowed_frames_buffer);
    clReleaseMemObject(samples_buffer);
    release_kernel_set(
        window_kernel,
        dft_kernel_naive,
        dft_kernel_local
    );
    cleanup_opencl(cl_ctx);

    return result;
}

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

    results.resize(batch_inputs.size());
    infos.resize(batch_inputs.size());
    resources.resize(batch_inputs.size());

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

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL for batch mode.\n";
        return {};
    }

    OpenCLKernelMode actual_mode = resolve_kernel_mode(
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

    cl_kernel selected_dft_kernel =
        (actual_mode == OpenCLKernelMode::Local)
            ? dft_kernel_local
            : dft_kernel_naive;

    cl_int err = CL_SUCCESS;

    cl_mem parent_output_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        total_output_bytes,
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create parent output buffer")) {
        release_kernel_set(
            window_kernel,
            dft_kernel_naive,
            dft_kernel_local
        );
        cleanup_opencl(cl_ctx);
        return {};
    }

    const size_t window_local[2] = {1, 64};
    const size_t dft_local[2] = {1, 16};

    bool enqueue_failed = false;

    for (size_t i = 0; i < batch_inputs.size(); ++i) {
        const auto& samples = batch_inputs[i].second;
        const auto& info = infos[i];
        auto& result = results[i].second;
        auto& res = resources[i];

        if (result.values.empty()) {
            continue;
        }

        const int num_samples = static_cast<int>(samples.size());

        res.samples_buffer = clCreateBuffer(
            cl_ctx.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * samples.size(),
            const_cast<float*>(samples.data()),
            &err
        );
        if (!check_opencl_error(err, "Failed to create batch samples buffer")) {
            enqueue_failed = true;
            break;
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
            enqueue_failed = true;
            break;
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
            enqueue_failed = true;
            break;
        }

        if (!set_window_kernel_args(
                window_kernel,
                res.samples_buffer,
                res.windowed_frames_buffer,
                num_samples,
                window_size,
                hop_size,
                result.num_frames)) {
            enqueue_failed = true;
            break;
        }

        if (!set_dft_kernel_args(
                selected_dft_kernel,
                actual_mode,
                res.windowed_frames_buffer,
                res.output_subbuffer,
                window_size,
                num_bins,
                result.num_frames)) {
            enqueue_failed = true;
            break;
        }

        const size_t window_global[2] = {
            static_cast<size_t>(result.num_frames),
            round_up(static_cast<size_t>(window_size), window_local[1])
        };

        const size_t dft_global[2] = {
            static_cast<size_t>(result.num_frames),
            round_up(static_cast<size_t>(num_bins), dft_local[1])
        };

        err = clEnqueueNDRangeKernel(
            cl_ctx.queue,
            window_kernel,
            2,
            nullptr,
            window_global,
            window_local,
            0,
            nullptr,
            &res.window_event
        );
        if (!check_opencl_error(err, "Failed to enqueue batch window kernel")) {
            enqueue_failed = true;
            break;
        }

        err = clEnqueueNDRangeKernel(
            cl_ctx.queue,
            selected_dft_kernel,
            2,
            nullptr,
            dft_global,
            dft_local,
            1,
            &res.window_event,
            &res.dft_event
        );
        if (!check_opencl_error(err, "Failed to enqueue batch DFT kernel")) {
            enqueue_failed = true;
            break;
        }
    }

    if (!enqueue_failed) {
        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            const auto& info = infos[i];
            auto& result = results[i].second;
            auto& res = resources[i];

            if (result.values.empty() || res.output_subbuffer == nullptr) {
                continue;
            }

            err = clEnqueueReadBuffer(
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
                enqueue_failed = true;
                break;
            }
        }
    }

    if (!enqueue_failed) {
        err = clFinish(cl_ctx.queue);
        check_opencl_error(err, "Failed to finish batch queue");
    }

    double total_window_kernel_ms = 0.0;
    double total_dft_kernel_ms = 0.0;

    if (gpu_times_ms != nullptr) {
        gpu_times_ms->resize(batch_inputs.size(), 0.0);
    }

    for (size_t i = 0; i < resources.size(); ++i) {
        auto& res = resources[i];

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

    release_batch_resources(resources);

    if (parent_output_buffer != nullptr) {
        clReleaseMemObject(parent_output_buffer);
    }

    release_kernel_set(
        window_kernel,
        dft_kernel_naive,
        dft_kernel_local
    );
    cleanup_opencl(cl_ctx);

    if (enqueue_failed) {
        return {};
    }

    return results;
}