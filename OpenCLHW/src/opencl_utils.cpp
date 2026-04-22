/*
 * opencl_utils.cpp
 *
 * Helper functions for OpenCL platform discovery, device selection,
 * program compilation, command queue creation, and resource cleanup.
 *
 * This file centralizes the host-side OpenCL setup used by the
 * spectrogram application. It loads kernel source code from disk,
 * selects an available device, builds the OpenCL program, creates
 * a profiling-enabled command queue, and provides cleanup utilities
 * for releasing OpenCL resources safely.
 *
 * High-level responsibilities:
 *   1. Query available OpenCL platforms and devices.
 *   2. Select a usable device, preferring GPU when available.
 *   3. Load and compile the kernel source file.
 *   4. Create the OpenCL context and command queue.
 *   5. Report build errors and device information.
 *   6. Release OpenCL resources during shutdown or failure.
 */

#include "opencl_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

/*
 * Returns the first available device of the requested type for the
 * given platform. This is used to prefer GPU execution while still
 * allowing fallback to default or CPU devices.
 */
bool get_first_device_for_type(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_device_id& device_out
) {
    cl_uint num_devices = 0;
    cl_int err = clGetDeviceIDs(
        platform,
        device_type,
        0,
        nullptr,
        &num_devices
    );

    if (err != CL_SUCCESS || num_devices == 0) {
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(
        platform,
        device_type,
        num_devices,
        devices.data(),
        nullptr
    );

    if (err != CL_SUCCESS || devices.empty()) {
        return false;
    }

    device_out = devices[0];
    return true;
}

/*
 * Prints the vendor and device name of the selected OpenCL device
 * so the runtime configuration is visible to the user.
 */
void print_selected_device_info(cl_device_id device) {
    if (device == nullptr) {
        return;
    }

    char name[256] = {0};
    char vendor[256] = {0};

    if (clGetDeviceInfo(
            device,
            CL_DEVICE_NAME,
            sizeof(name),
            name,
            nullptr) == CL_SUCCESS &&
        clGetDeviceInfo(
            device,
            CL_DEVICE_VENDOR,
            sizeof(vendor),
            vendor,
            nullptr) == CL_SUCCESS) {
        std::cout << "OpenCL device: " << vendor
                  << " - " << name << "\n";
    }
}

}  // namespace

/* Prints an error message when an OpenCL API call fails. */
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

/* Reads the OpenCL kernel source file from disk into a single string. */
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

/*
 * Retrieves and prints the OpenCL compiler build log when program
 * compilation fails, which helps diagnose kernel syntax or device-specific issues.
 */
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

/*
 * Initializes the OpenCL runtime for the application.
 *
 * This function:
 *   1. Queries available OpenCL platforms.
 *   2. Selects the first platform and a usable device.
 *   3. Creates an OpenCL context for that device.
 *   4. Creates a command queue with profiling enabled.
 *   5. Loads kernel source code from disk.
 *   6. Builds the OpenCL program for the selected device.
 *
 * Device preference order:
 *   GPU -> default device -> CPU
 *
 * On failure, any partially created resources are released before returning.
 */
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
    cl_ctx.device = nullptr;

    if (!get_first_device_for_type(
            cl_ctx.platform,
            CL_DEVICE_TYPE_GPU,
            cl_ctx.device) &&
        !get_first_device_for_type(
            cl_ctx.platform,
            CL_DEVICE_TYPE_DEFAULT,
            cl_ctx.device) &&
        !get_first_device_for_type(
            cl_ctx.platform,
            CL_DEVICE_TYPE_CPU,
            cl_ctx.device)) {
        std::cerr << "No usable OpenCL devices found.\n";
        return false;
    }

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
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0
    };
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
        CL_QUEUE_PROFILING_ENABLE,
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

    print_selected_device_info(cl_ctx.device);
    return true;
}

/*
 * Releases OpenCL resources owned by the context structure and resets
 * all handles to null so repeated cleanup calls remain safe.
 */
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