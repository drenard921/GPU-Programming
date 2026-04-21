#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <string>

struct OpenCLContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
};

bool initialize_opencl(
    OpenCLContext& cl_ctx,
    const std::string& kernel_path
);

void cleanup_opencl(OpenCLContext& cl_ctx);

std::string load_kernel_source(const std::string& kernel_path);

void print_build_log(
    cl_program program,
    cl_device_id device
);

bool check_opencl_error(
    cl_int err,
    const std::string& message
);

#endif