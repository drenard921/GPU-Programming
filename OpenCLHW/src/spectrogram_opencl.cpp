#include "spectrogram_opencl.h"

#include <iostream>
#include <string>
#include <vector>

#include "opencl_utils.h"

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

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = clCreateKernel(
        cl_ctx.program,
        "spectrogram_kernel",
        &err
    );
    if (!check_opencl_error(err, "Failed to create OpenCL kernel")) {
        cleanup_opencl(cl_ctx);
        return {};
    }

    const int num_samples = static_cast<int>(samples.size());

    cl_mem samples_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * samples.size(),
        const_cast<float*>(samples.data()),
        &err
    );
    if (!check_opencl_error(err, "Failed to create samples buffer")) {
        clReleaseKernel(kernel);
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
        clReleaseMemObject(samples_buffer);
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &samples_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &num_samples);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &window_size);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &hop_size);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &num_bins);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &result.num_frames);

    if (!check_opencl_error(err, "Failed to set kernel arguments")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(samples_buffer);
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    const size_t global_size[2] = {
        static_cast<size_t>(result.num_frames),
        static_cast<size_t>(result.num_bins)
    };

    err = clEnqueueNDRangeKernel(
        cl_ctx.queue,
        kernel,
        2,
        nullptr,
        global_size,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    if (!check_opencl_error(err, "Failed to enqueue OpenCL kernel")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(samples_buffer);
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    err = clFinish(cl_ctx.queue);
    if (!check_opencl_error(err, "Failed to finish OpenCL queue")) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(samples_buffer);
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
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
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(samples_buffer);
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    clReleaseMemObject(output_buffer);
    clReleaseMemObject(samples_buffer);
    clReleaseKernel(kernel);
    cleanup_opencl(cl_ctx);

    return result;
}

struct BatchItemInfo {
    int num_frames = 0;
    int num_bins = 0;
    size_t value_count = 0;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

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

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(
        cl_ctx.program,
        "spectrogram_kernel",
        &err
    );
    if (!check_opencl_error(err, "Failed to create OpenCL kernel")) {
        cleanup_opencl(cl_ctx);
        return {};
    }

    cl_mem parent_output_buffer = clCreateBuffer(
        cl_ctx.context,
        CL_MEM_WRITE_ONLY,
        total_output_bytes,
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create parent output buffer")) {
        clReleaseKernel(kernel);
        cleanup_opencl(cl_ctx);
        return {};
    }

    for (size_t i = 0; i < batch_samples.size(); ++i) {
        const auto& samples = batch_samples[i];
        auto& info = infos[i];
        auto& result = results[i];

        if (result.values.empty()) {
            continue;
        }

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
            clReleaseMemObject(samples_buffer);
            continue;
        }

        const int num_samples = static_cast<int>(samples.size());

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &samples_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_subbuffer);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &num_samples);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &window_size);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &hop_size);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &num_bins);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &result.num_frames);

        if (!check_opencl_error(err, "Failed to set batch kernel arguments")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        const size_t global_size[2] = {
            static_cast<size_t>(result.num_frames),
            static_cast<size_t>(num_bins)
        };

        err = clEnqueueNDRangeKernel(
            cl_ctx.queue,
            kernel,
            2,
            nullptr,
            global_size,
            nullptr,
            0,
            nullptr,
            nullptr
        );
        if (!check_opencl_error(err, "Failed to enqueue batch kernel")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(samples_buffer);
            continue;
        }

        err = clFinish(cl_ctx.queue);
        if (!check_opencl_error(err, "Failed to finish batch queue")) {
            clReleaseMemObject(output_subbuffer);
            clReleaseMemObject(samples_buffer);
            continue;
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

        clReleaseMemObject(output_subbuffer);
        clReleaseMemObject(samples_buffer);
    }

    clReleaseMemObject(parent_output_buffer);
    clReleaseKernel(kernel);
    cleanup_opencl(cl_ctx);

    return results;
}