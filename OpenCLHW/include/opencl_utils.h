/*
 * opencl_utils.h
 *
 * Declarations for host-side OpenCL setup and teardown utilities used by
 * the spectrogram application.
 *
 * This header defines the OpenCLContext structure, which stores the core
 * OpenCL handles needed during execution, along with helper functions for
 * initialization, cleanup, kernel source loading, build-log reporting,
 * and basic error checking.
 */

#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <string>

/*
 * Stores the main OpenCL objects required by the application.
 * These handles are created during initialization and released
 * during cleanup.
 */
struct OpenCLContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
};

/*
 * Initializes the OpenCL runtime by selecting a device, creating the
 * context and command queue, loading the kernel source, and building
 * the OpenCL program.
 */
bool initialize_opencl(
    OpenCLContext& cl_ctx,
    const std::string& kernel_path
);

/*
 * Releases all OpenCL resources stored in the context structure and
 * resets the handles to nullptr.
 */
void cleanup_opencl(OpenCLContext& cl_ctx);

/* Loads the OpenCL kernel source code from the specified file path. */
std::string load_kernel_source(const std::string& kernel_path);

/* Prints the compiler build log for an OpenCL program. */
void print_build_log(
    cl_program program,
    cl_device_id device
);

/* Reports an OpenCL error code together with a descriptive message. */
bool check_opencl_error(
    cl_int err,
    const std::string& message
);

#endif