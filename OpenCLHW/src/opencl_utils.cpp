#include "opencl_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

bool check_opencl_error(
    cl_int err,
    const std::string& message
) {
    if (err != CL_SUCCESS) {
        std::cerr << message << " (OpenCL error " << err << ")\n";
        return false;
    }
    return true;
}

std::string load_kernel_source(const std::string& kernel_path) {
    std::ifstream file(kernel_path);
    if (!file) {
        std::cerr << "Failed to open kernel file: "
                  << kernel_path << "\n";
        return "";
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void print_build_log(
    cl_program program,
    cl_device_id device
) {
    size_t log_size = 0;
    cl_int err = clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_LOG,
        0,
        nullptr,
        &log_size
    );

    if (err != CL_SUCCESS || log_size == 0) {
        std::cerr << "Could not retrieve OpenCL build log.\n";
        return;
    }

    std::vector<char> log(log_size);
    err = clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_LOG,
        log_size,
        log.data(),
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr << "Could not read OpenCL build log.\n";
        return;
    }

    std::cerr << "\nOpenCL Build Log:\n";
    std::cerr << "------------------\n";
    std::cerr << log.data() << "\n";
}

bool initialize_opencl(
    OpenCLContext& cl_ctx,
    const std::string& kernel_path
) {
    cl_int err = CL_SUCCESS;

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (!check_opencl_error(err, "Failed to query OpenCL platforms")) {
        return false;
    }

    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(
        num_platforms,
        platforms.data(),
        nullptr
    );
    if (!check_opencl_error(err, "Failed to get OpenCL platform IDs")) {
        return false;
    }

    cl_ctx.platform = platforms[0];

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(
        cl_ctx.platform,
        CL_DEVICE_TYPE_GPU,
        0,
        nullptr,
        &num_devices
    );

    if (err != CL_SUCCESS || num_devices == 0) {
        err = clGetDeviceIDs(
            cl_ctx.platform,
            CL_DEVICE_TYPE_DEFAULT,
            0,
            nullptr,
            &num_devices
        );
        if (!check_opencl_error(
                err,
                "Failed to query OpenCL devices")) {
            return false;
        }
    }

    if (num_devices == 0) {
        std::cerr << "No OpenCL devices found.\n";
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(
        cl_ctx.platform,
        CL_DEVICE_TYPE_GPU,
        num_devices,
        devices.data(),
        nullptr
    );

    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(
            cl_ctx.platform,
            CL_DEVICE_TYPE_DEFAULT,
            num_devices,
            devices.data(),
            nullptr
        );
        if (!check_opencl_error(
                err,
                "Failed to get OpenCL device IDs")) {
            return false;
        }
    }

    cl_ctx.device = devices[0];

    cl_ctx.context = clCreateContext(
        nullptr,
        1,
        &cl_ctx.device,
        nullptr,
        nullptr,
        &err
    );
    if (!check_opencl_error(err, "Failed to create OpenCL context")) {
        cleanup_opencl(cl_ctx);
        return false;
    }

#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = {0};
    cl_ctx.queue = clCreateCommandQueueWithProperties(
        cl_ctx.context,
        cl_ctx.device,
        props,
        &err
    );
#else
    cl_ctx.queue = clCreateCommandQueue(
        cl_ctx.context,
        cl_ctx.device,
        0,
        &err
    );
#endif

    if (!check_opencl_error(
            err,
            "Failed to create OpenCL command queue")) {
        cleanup_opencl(cl_ctx);
        return false;
    }

    std::string source = load_kernel_source(kernel_path);
    if (source.empty()) {
        cleanup_opencl(cl_ctx);
        return false;
    }

    const char* source_ptr = source.c_str();
    size_t source_size = source.size();

    cl_ctx.program = clCreateProgramWithSource(
        cl_ctx.context,
        1,
        &source_ptr,
        &source_size,
        &err
    );
    if (!check_opencl_error(
            err,
            "Failed to create OpenCL program")) {
        cleanup_opencl(cl_ctx);
        return false;
    }

    err = clBuildProgram(
        cl_ctx.program,
        1,
        &cl_ctx.device,
        nullptr,
        nullptr,
        nullptr
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build OpenCL program.\n";
        print_build_log(cl_ctx.program, cl_ctx.device);
        cleanup_opencl(cl_ctx);
        return false;
    }

    return true;
}

void cleanup_opencl(OpenCLContext& cl_ctx) {
    if (cl_ctx.program != nullptr) {
        clReleaseProgram(cl_ctx.program);
        cl_ctx.program = nullptr;
    }

    if (cl_ctx.queue != nullptr) {
        clReleaseCommandQueue(cl_ctx.queue);
        cl_ctx.queue = nullptr;
    }

    if (cl_ctx.context != nullptr) {
        clReleaseContext(cl_ctx.context);
        cl_ctx.context = nullptr;
    }

    cl_ctx.device = nullptr;
    cl_ctx.platform = nullptr;
}