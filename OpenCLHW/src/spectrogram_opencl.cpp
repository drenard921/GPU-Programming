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