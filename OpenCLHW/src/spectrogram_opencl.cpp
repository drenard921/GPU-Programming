#include "spectrogram_opencl.h"

#include <iostream>
#include <string>
#include <vector>

#include "opencl_utils.h"

namespace {

struct BatchItemInfo {
    int num_frames = 0;
    int num_bins = 0;
    size_t value_count = 0;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }
    size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    return value + multiple - remainder;
}

bool create_kernels(
    const OpenCLContext& cl_ctx,
    cl_kernel& window_kernel,
    cl_kernel& dft_kernel
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

    dft_kernel = clCreateKernel(
        cl_ctx.program,
        "dft_power_kernel",
        &err
    );
    if (!check_opencl_error(err, "Failed to create DFT kernel")) {
        clReleaseKernel(window_kernel);
        window_kernel = nullptr;
        return false;
    }

    return true;
}

void release_kernel_pair(cl_kernel& window_kernel, cl_kernel& dft_kernel) {
    if (window_kernel != nullptr) {
        clReleaseKernel(window_kernel);
        window_kernel = nullptr;
    }

    if (dft_kernel != nullptr) {
        clReleaseKernel(dft_kernel);
        dft_kernel = nullptr;
    }
}

void print_event_time_ms(
    cl_event event,
    const std::string& label
) {
    if (event == nullptr) {
        return;
    }

    cl_ulong start = 0;
    cl_ulong end = 0;

    cl_int err1 = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        nullptr
    );

    cl_int err2 = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        nullptr
    );

    if (err1 == CL_SUCCESS && err2 == CL_SUCCESS && end >= start) {
        double ms = static_cast<double>(end - start) / 1.0e6;
        std::cout << label << ": " << ms << " ms\n";
    }
}

}  // namespace

SpectrogramResult compute_spectrogram_opencl(
    const std::vector<float>& samples,
    int window_size,
    int hop_size,
    int num_bins
) {
    SpectrogramResult result{};

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
    result.values.resize(result.num_frames * result.num_bins, 0.0f);

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL.\n";
        return {};
    }

    cl_kernel window_kernel = nullptr;
    cl_kernel dft_kernel = nullptr;
    if (!create_kernels(cl_ctx, window_kernel, dft_kernel)) {
        cleanup_opencl(cl_ctx);
        return {};
    }

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
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    cl_mem windowed_frames_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_WRITE,
        sizeof(float) * result.num_frames * window_size,
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create windowed frames buffer")) {
        clReleaseMemObject(samples_buffer);
        release_kernel_pair(window_kernel, dft_kernel);
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
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

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
    err |= clSetKernelArg(window_kernel, 5, sizeof(int), &result.num_frames);

    if (!check_opencl_error(err, "Failed to set window kernel arguments")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

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
    err |= clSetKernelArg(dft_kernel, 5, sizeof(int), &result.num_frames);

    if (!check_opencl_error(err, "Failed to set DFT kernel arguments")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
        release_kernel_pair(window_kernel, dft_kernel);
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
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        dft_kernel,
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
        release_kernel_pair(window_kernel, dft_kernel);
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
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    print_event_time_ms(window_event, "OpenCL window kernel");
    print_event_time_ms(dft_event, "OpenCL DFT kernel");

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
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    clReleaseEvent(dft_event);
    clReleaseEvent(window_event);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(windowed_frames_buffer);
    clReleaseMemObject(samples_buffer);
    release_kernel_pair(window_kernel, dft_kernel);
    cleanup_opencl(cl_ctx);

    return result;
}

std::vector<SpectrogramResult> compute_spectrogram_opencl_batch(
    const std::vector<std::vector<float>>& batch_samples,
    int window_size,
    int hop_size,
    int num_bins
) {
    std::vector<SpectrogramResult> results;
    if (batch_samples.empty()) {
        return results;
    }

    results.resize(batch_samples.size());
    std::vector<BatchItemInfo> infos(batch_samples.size());

    size_t total_output_bytes = 0;

    for (size_t i = 0; i < batch_samples.size(); ++i) {
        const auto& samples = batch_samples[i];

        if (static_cast<int>(samples.size()) < window_size) {
            continue;
        }

        infos[i].num_frames =
            1 + static_cast<int>((samples.size() - window_size) / hop_size);
        infos[i].num_bins = num_bins;
        infos[i].value_count =
            static_cast<size_t>(infos[i].num_frames) *
            static_cast<size_t>(num_bins);
        infos[i].byte_offset = total_output_bytes;
        infos[i].byte_size = infos[i].value_count * sizeof(float);

        total_output_bytes += infos[i].byte_size;

        results[i].num_frames = infos[i].num_frames;
        results[i].num_bins = num_bins;
        results[i].values.resize(infos[i].value_count, 0.0f);
    }

    OpenCLContext cl_ctx{};
    if (!initialize_opencl(cl_ctx, "kernels/spectrogram.cl")) {
        std::cerr << "Failed to initialize OpenCL for batch mode.\n";
        return {};
    }

    cl_kernel window_kernel = nullptr;
    cl_kernel dft_kernel = nullptr;
    if (!create_kernels(cl_ctx, window_kernel, dft_kernel)) {
        cleanup_opencl(cl_ctx);
        return {};
    }

    cl_int err = CL_SUCCESS;

    cl_mem parent_output_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        total_output_bytes,
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create parent output buffer")) {
        release_kernel_pair(window_kernel, dft_kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    const size_t window_local[2] = {1, 64};
    const size_t dft_local[2] = {1, 16};

    double total_window_kernel_ms = 0.0;
    double total_dft_kernel_ms = 0.0;

    for (size_t i = 0; i < batch_samples.size(); ++i) {
        const auto& samples = batch_samples[i];
        const auto& info = infos[i];
        auto& result = results[i];

        if (result.values.empty()) {
            continue;
        }

        const int num_samples = static_cast<int>(samples.size());

        cl_mem samples_buffer = clCreateBuffer(
            cl_ctx.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * samples.size(),
            const_cast<float*>(samples.data()),
            &err
        );
        if (!check_opencl_error(err, "Failed to create batch samples buffer")) {
            continue;
        }

        cl_mem windowed_frames_buffer = clCreateBuffer(
            cl_ctx.context,
            CL_MEM_READ_WRITE,
            sizeof(float) * result.num_frames * window_size,
            nullptr,
            &err
        );
        if (!check_opencl_error(
                err,
                "Failed to create batch windowed frames buffer")) {
            clReleaseMemObject(samples_buffer);
            continue;
        }

        cl_buffer_region region{};
        region.origin = info.byte_offset;
        region.size = info.byte_size;

        cl_mem output_subbuffer = clCreateSubBuffer(
            parent_output_buffer,
            CL_MEM_WRITE_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err
        );
        if (!check_opencl_error(err, "Failed to create output sub-buffer")) {
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

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
        err |= clSetKernelArg(window_kernel, 5, sizeof(int), &result.num_frames);

        if (!check_opencl_error(
                err,
                "Failed to set batch window kernel arguments")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        err  = clSetKernelArg(
            dft_kernel,
            0,
            sizeof(cl_mem),
            &windowed_frames_buffer
        );
        err |= clSetKernelArg(dft_kernel, 1, sizeof(cl_mem), &output_subbuffer);
        err |= clSetKernelArg(
            dft_kernel,
            2,
            sizeof(float) * static_cast<size_t>(window_size),
            nullptr
        );
        err |= clSetKernelArg(dft_kernel, 3, sizeof(int), &window_size);
        err |= clSetKernelArg(dft_kernel, 4, sizeof(int), &num_bins);
        err |= clSetKernelArg(dft_kernel, 5, sizeof(int), &result.num_frames);

        if (!check_opencl_error(
                err,
                "Failed to set batch DFT kernel arguments")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        const size_t window_global[2] = {
            static_cast<size_t>(result.num_frames),
            round_up(static_cast<size_t>(window_size), window_local[1])
        };

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
        if (!check_opencl_error(err, "Failed to enqueue batch window kernel")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        err = clEnqueueNDRangeKernel(
            cl_ctx.queue,
            dft_kernel,
            2,
            nullptr,
            dft_global,
            dft_local,
            1,
            &window_event,
            &dft_event
        );
        if (!check_opencl_error(err, "Failed to enqueue batch DFT kernel")) {
            clReleaseEvent(window_event);
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        err = clFinish(cl_ctx.queue);
        if (!check_opencl_error(err, "Failed to finish batch queue")) {
            clReleaseEvent(dft_event);
            clReleaseEvent(window_event);
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(windowed_frames_buffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        cl_ulong start = 0;
        cl_ulong end = 0;

        if (clGetEventProfilingInfo(
                window_event,
                CL_PROFILING_COMMAND_START,
                sizeof(cl_ulong),
                &start,
                nullptr) == CL_SUCCESS &&
            clGetEventProfilingInfo(
                window_event,
                CL_PROFILING_COMMAND_END,
                sizeof(cl_ulong),
                &end,
                nullptr) == CL_SUCCESS &&
            end >= start) {
            total_window_kernel_ms +=
                static_cast<double>(end - start) / 1.0e6;
        }

        if (clGetEventProfilingInfo(
                dft_event,
                CL_PROFILING_COMMAND_START,
                sizeof(cl_ulong),
                &start,
                nullptr) == CL_SUCCESS &&
            clGetEventProfilingInfo(
                dft_event,
                CL_PROFILING_COMMAND_END,
                sizeof(cl_ulong),
                &end,
                nullptr) == CL_SUCCESS &&
            end >= start) {
            total_dft_kernel_ms +=
                static_cast<double>(end - start) / 1.0e6;
        }

        err = clEnqueueReadBuffer(
            cl_ctx.queue,
            output_subbuffer,
            CL_TRUE,
            0,
            info.byte_size,
            result.values.data(),
            0,
            nullptr,
            nullptr
        );
        check_opencl_error(err, "Failed to read batch sub-buffer");

        clReleaseEvent(dft_event);
        clReleaseEvent(window_event);
        clReleaseMemObject(output_subbuffer);
        clReleaseMemObject(windowed_frames_buffer);
        clReleaseMemObject(samples_buffer);
    }

    std::cout << "OpenCL batch window kernel total: "
              << total_window_kernel_ms << " ms\n";
    std::cout << "OpenCL batch DFT kernel total:    "
              << total_dft_kernel_ms << " ms\n";

    clReleaseMemObject(parent_output_buffer);
    release_kernel_pair(window_kernel, dft_kernel);
    cleanup_opencl(cl_ctx);

    return results;
}