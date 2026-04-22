/*
 * main.cpp
 *
 * Entry point and host-side orchestration for the spectrogram application.
 *
 * This file handles command-line argument parsing, runtime configuration,
 * WAV file input, CPU/OpenCL spectrogram execution, PNG output generation,
 * and runtime benchmarking. It supports both single-file processing and
 * batch directory processing, along with CPU-only, GPU-only, or side-by-side
 * comparison modes.
 *
 * High-level pipeline:
 *   1. Parse command-line arguments into an AppConfig structure.
 *   2. Read one WAV file or collect multiple WAV files from a directory.
 *   3. Compute a spectrogram using either the CPU or OpenCL implementation.
 *   4. Write the resulting spectrogram image(s) as PNG files.
 *   5. Report timing summaries for I/O, computation, and total runtime.
 *
 * Batch mode:
 *   If the input path is a directory, each .wav file in that directory is
 *   processed and written to the output folder. GPU batch mode can process
 *   multiple inputs through the OpenCL batch interface for comparison against
 *   a sequential CPU baseline.
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cctype>

#include "image_writer.h"
#include "spectrogram_cpu.h"
#include "spectrogram_opencl.h"
#include "wav_reader.h"

namespace fs = std::filesystem;

/* Stores command-line options and runtime settings for one program run. */
struct AppConfig {
    std::string input_path;
    std::string mode = "cpu";
    OpenCLKernelMode kernel_mode = OpenCLKernelMode::Local;
    int window_size = 1024;
    int hop_size = 512;
    int num_bins = 512;
    std::string output_prefix = "output/output";
};

/* Stores timing breakdowns for I/O, computation, and full pipeline runtime. */
struct TimingInfo {
    long long wav_read_ms = 0;
    long long spectrogram_ms = 0;
    long long image_write_ms = 0;
    long long total_ms = 0;
};

/* Returns true if the requested execution mode is supported. */
bool is_valid_mode(const std::string& mode) {
    return mode == "cpu" || mode == "gpu" || mode == "both";
}

/* Converts a command-line kernel mode string into the corresponding OpenCL kernel enum. */
bool parse_kernel_mode_arg(
    const std::string& text,
    OpenCLKernelMode& kernel_mode
) {
    if (text == "naive") {
        kernel_mode = OpenCLKernelMode::Naive;
        return true;
    }

    if (text == "local") {
        kernel_mode = OpenCLKernelMode::Local;
        return true;
    }

    return false;
}

/* Converts the kernel enum back into a human-readable label. */
std::string kernel_mode_to_string(OpenCLKernelMode kernel_mode) {
    return kernel_mode == OpenCLKernelMode::Naive ? "naive" : "local";
}

/* Prints command-line usage instructions and example invocations. */
void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name
        << " <wav_file_or_directory> [cpu|gpu|both] "
        << "[naive|local] [window_size] [hop_size] [num_bins]\n\n"
        << "Notes:\n"
        << "  - Kernel mode is only used for gpu or both modes.\n"
        << "  - If kernel mode is omitted, it defaults to local.\n\n"
        << "Examples:\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav cpu\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav gpu naive\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav both local "
        << "1024 512 512\n"
        << "  " << program_name
        << " Data/genres_original/blues gpu naive 1024 512 512\n";
}

/* Parses one positive integer command-line argument. */
bool parse_int_arg(
    const char* text,
    int& value,
    const std::string& name
) {
    try {
        value = std::stoi(text);
    } catch (...) {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        return false;
    }

    if (value <= 0) {
        std::cerr << name << " must be positive.\n";
        return false;
    }

    return true;
}

/*
 * Parses command-line arguments into the application configuration.
 *
 * Expected order:
 *   input_path [mode] [kernel_mode] [window_size] [hop_size] [num_bins]
 *
 * The parser accepts omitted optional arguments and falls back to defaults.
 * Kernel mode is only parsed when GPU execution is requested.
 */
bool parse_args(int argc, char** argv, AppConfig& config) {
    if (argc < 2) {
        print_usage(argv[0]);
        return false;
    }

    /* The input path may be either a single WAV file or a directory of WAV files. */
    config.input_path = argv[1];

    int next_arg = 2;

    /* Parse optional execution mode: cpu, gpu, or both. */
    if (argc > next_arg) {
        const std::string mode_text = argv[next_arg];
        if (is_valid_mode(mode_text)) {
            config.mode = mode_text;
            ++next_arg;
        }
    }

    /* Parse optional OpenCL kernel variant when GPU execution is enabled. */
    if (config.mode == "gpu" || config.mode == "both") {
        if (argc > next_arg) {
            OpenCLKernelMode parsed_mode = config.kernel_mode;
            if (parse_kernel_mode_arg(argv[next_arg], parsed_mode)) {
                config.kernel_mode = parsed_mode;
                ++next_arg;
            }
        }
    }

    /* Parse optional numeric parameters in order, using defaults when omitted. */
    if (argc > next_arg &&
        !parse_int_arg(argv[next_arg], config.window_size, "window_size")) {
        return false;
    }
    if (argc > next_arg) {
        ++next_arg;
    }

    if (argc > next_arg &&
        !parse_int_arg(argv[next_arg], config.hop_size, "hop_size")) {
        return false;
    }
    if (argc > next_arg) {
        ++next_arg;
    }

    if (argc > next_arg &&
        !parse_int_arg(argv[next_arg], config.num_bins, "num_bins")) {
        return false;
    }
    if (argc > next_arg) {
        ++next_arg;
    }

    if (argc > next_arg) {
        std::cerr << "Too many arguments.\n";
        print_usage(argv[0]);
        return false;
    }

    return true;
}

/* Ensures the output directory exists before PNG files are written. */
bool ensure_output_directory(const std::string& output_prefix) {
    const fs::path out_path(output_prefix);
    const fs::path out_dir = out_path.parent_path();

    if (out_dir.empty()) {
        return true;
    }

    std::error_code ec;
    fs::create_directories(out_dir, ec);

    if (ec) {
        std::cerr << "Failed to create output directory: "
                  << out_dir << "\n";
        return false;
    }

    return true;
}

/* Builds the PNG output path for a single-file run. */
std::string make_output_png_path(
    const AppConfig& config,
    const std::string& mode,
    OpenCLKernelMode kernel_mode
) {
    std::string out_path = config.output_prefix + "_" + mode;

    if (mode == "gpu") {
        out_path += "_" + kernel_mode_to_string(kernel_mode);
    }

    out_path += ".png";
    return out_path;
}

/* Builds the PNG output path for one file in batch mode. */
std::string make_batch_output_path(
    const fs::path& wav_path,
    const std::string& mode,
    OpenCLKernelMode kernel_mode
) {
    fs::path filename = wav_path.stem();
    std::string out_name = filename.string() + "_" + mode;

    if (mode == "gpu") {
        out_name += "_" + kernel_mode_to_string(kernel_mode);
    }

    out_name += ".png";
    return (fs::path("output") / out_name).string();
}

/* Reads a WAV file and records the elapsed read time in milliseconds. */
bool load_wav_with_timing(
    const std::string& input_path,
    WavData& wav,
    TimingInfo& timing
) {
    const auto start = std::chrono::high_resolution_clock::now();
    const bool ok = read_wav_file(input_path, wav);
    const auto end = std::chrono::high_resolution_clock::now();

    timing.wav_read_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    if (!ok) {
        std::cerr << "Failed to read WAV file: " << input_path << "\n";
        return false;
    }

    return true;
}

/*
 * Runs either the CPU or OpenCL spectrogram implementation and records
 * end-to-end compute time for that stage.
 */
bool compute_spectrogram_with_timing(
    const AppConfig& config,
    const WavData& wav,
    SpectrogramResult& spec,
    TimingInfo& timing
) {
    const auto start = std::chrono::high_resolution_clock::now();

    if (config.mode == "cpu") {
        spec = compute_spectrogram_cpu(
            wav.samples,
            config.window_size,
            config.hop_size,
            config.num_bins
        );
    } else {
        double gpu_kernel_ms = 0.0;
        spec = compute_spectrogram_opencl(
            wav.samples,
            config.window_size,
            config.hop_size,
            config.num_bins,
            &gpu_kernel_ms,
            config.kernel_mode
        );
    }

    const auto end = std::chrono::high_resolution_clock::now();
    timing.spectrogram_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    if (spec.values.empty()) {
        std::cerr << "Spectrogram computation failed.\n";
        return false;
    }

    return true;
}

/* Writes the spectrogram image to disk and records PNG write time. */
bool write_png_with_timing(
    const std::string& out_path,
    const SpectrogramResult& spec,
    TimingInfo& timing
) {
    const auto start = std::chrono::high_resolution_clock::now();
    const bool ok = write_png(
        out_path,
        spec.values,
        spec.num_frames,
        spec.num_bins
    );
    const auto end = std::chrono::high_resolution_clock::now();

    timing.image_write_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    if (!ok) {
        std::cerr << "Failed to write PNG: " << out_path << "\n";
        return false;
    }

    return true;
}

/* Prints a summary of the input audio file and selected runtime settings. */
void print_input_summary(const AppConfig& config, const WavData& wav) {
    std::cout << "Input path:            " << config.input_path << "\n";
    std::cout << "Mode:                  " << config.mode << "\n";

    if (config.mode == "gpu" || config.mode == "both") {
        std::cout << "Kernel mode:           "
                  << kernel_mode_to_string(config.kernel_mode) << "\n";
    }

    std::cout << "Sample rate:           " << wav.sample_rate << " Hz\n";
    std::cout << "Channels:              " << wav.channels << "\n";
    std::cout << "Samples:               " << wav.samples.size() << "\n";
    std::cout << "Window size:           " << config.window_size << "\n";
    std::cout << "Hop size:              " << config.hop_size << "\n";
    std::cout << "Number of bins:        " << config.num_bins << "\n";
}

/* Prints the dimensions of the computed spectrogram result. */
void print_spectrogram_summary(const SpectrogramResult& spec) {
    std::cout << "Frames:                " << spec.num_frames << "\n";
    std::cout << "Bins:                  " << spec.num_bins << "\n";
}

/* Prints a formatted timing summary for one execution mode. */
void print_timing_summary(
    const std::string& label,
    const TimingInfo& timing
) {
    std::cout << "\n" << label << "\n";
    std::cout << "-------------------------\n";
    std::cout << "WAV read time:         "
              << timing.wav_read_ms << " ms\n";
    std::cout << "Spectrogram time:      "
              << timing.spectrogram_ms << " ms\n";
    std::cout << "PNG write time:        "
              << timing.image_write_ms << " ms\n";
    std::cout << "Total runtime:         "
              << timing.total_ms << " ms\n";
}

/* Prints CPU versus GPU runtime speedup comparisons. */
void print_comparison(
    const TimingInfo& cpu,
    const TimingInfo& gpu
) {
    std::cout << "\nCPU vs GPU Comparison\n";
    std::cout << "---------------------\n";

    if (gpu.spectrogram_ms > 0) {
        const double compute_speedup =
            static_cast<double>(cpu.spectrogram_ms) /
            static_cast<double>(gpu.spectrogram_ms);
        std::cout << "Compute speedup:      "
                  << compute_speedup << "x\n";
    }

    if (gpu.total_ms > 0) {
        const double total_speedup =
            static_cast<double>(cpu.total_ms) /
            static_cast<double>(gpu.total_ms);
        std::cout << "Pipeline speedup:     "
                  << total_speedup << "x\n";
    }
}

/*
 * Executes the full single-input pipeline:
 *   create output directory,
 *   read WAV data,
 *   compute spectrogram,
 *   write PNG output,
 *   and report timing results.
 */
bool run_pipeline(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    const auto total_start = std::chrono::high_resolution_clock::now();

    WavData wav;
    if (!load_wav_with_timing(config.input_path, wav, timing)) {
        return false;
    }

    print_input_summary(config, wav);

    SpectrogramResult spec;
    if (!compute_spectrogram_with_timing(config, wav, spec, timing)) {
        return false;
    }

    print_spectrogram_summary(spec);

    const std::string out_path = make_output_png_path(
        config,
        config.mode,
        config.kernel_mode
    );

    if (!write_png_with_timing(out_path, spec, timing)) {
        return false;
    }

    const auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "Wrote image:           " << out_path << "\n";
    print_timing_summary(
        config.mode == "cpu" ? "CPU Runtime Summary"
                             : "OpenCL Runtime Summary",
        timing
    );

    return true;
}

/*
 * Runs both CPU and GPU pipelines on the same input so their runtimes
 * can be reported side by side.
 */
bool run_both(const AppConfig& base_config) {
    AppConfig cpu_config = base_config;
    cpu_config.mode = "cpu";

    AppConfig gpu_config = base_config;
    gpu_config.mode = "gpu";

    TimingInfo cpu_timing{};
    TimingInfo gpu_timing{};

    std::cout << "\n=== Running CPU ===\n";
    if (!run_pipeline(cpu_config, cpu_timing)) {
        return false;
    }

    std::cout << "\n=== Running GPU ("
              << kernel_mode_to_string(gpu_config.kernel_mode)
              << ") ===\n";
    if (!run_pipeline(gpu_config, gpu_timing)) {
        return false;
    }

    print_comparison(cpu_timing, gpu_timing);
    return true;
}

/* Collects and sorts all .wav files found in the input directory for batch processing. */
std::vector<fs::path> collect_wav_files(const std::string& directory_path) {
    std::vector<fs::path> wav_files;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(
                        ext.begin(),
                        ext.end(),
                        ext.begin(),
                        [](unsigned char c) {
                            return static_cast<char>(std::tolower(c));
                        }
                    );

        if (ext == ".wav") {
            wav_files.push_back(entry.path());
        }
    }

    std::sort(wav_files.begin(), wav_files.end());
    return wav_files;
}

/*
 * Loads all WAV files for GPU batch execution and stores each input as
 * a filename/sample-vector pair expected by the batch OpenCL interface.
 */
bool load_batch_wavs(
    const std::vector<fs::path>& wav_files,
    std::vector<std::pair<std::string, std::vector<float>>>& batch_inputs,
    long long& total_wav_read_ms
) {
    batch_inputs.clear();
    total_wav_read_ms = 0;

    for (const auto& wav_path : wav_files) {
        TimingInfo timing{};
        WavData wav;

        if (!load_wav_with_timing(wav_path.string(), wav, timing)) {
            std::cerr << "Failed to read WAV in batch mode: "
                      << wav_path << "\n";
            return false;
        }

        total_wav_read_ms += timing.wav_read_ms;
        batch_inputs.push_back(
            {wav_path.filename().string(), std::move(wav.samples)}
        );
    }

    return true;
}

/* Writes all batch spectrogram results to PNG files and accumulates write time. */
bool write_batch_pngs(
    const std::vector<fs::path>& wav_files,
    const std::vector<SpectrogramResult>& results,
    const std::string& mode,
    OpenCLKernelMode kernel_mode,
    long long& total_png_write_ms
) {
    total_png_write_ms = 0;

    if (wav_files.size() != results.size()) {
        std::cerr << "Batch output size mismatch.\n";
        return false;
    }

    for (size_t i = 0; i < wav_files.size(); ++i) {
        TimingInfo timing{};
        const std::string out_path =
            make_batch_output_path(wav_files[i], mode, kernel_mode);

        if (!write_png_with_timing(out_path, results[i], timing)) {
            std::cerr << "Failed to write batch PNG: "
                      << out_path << "\n";
            return false;
        }

        total_png_write_ms += timing.image_write_ms;
        std::cout << "Wrote image: " << out_path << "\n";
    }

    return true;
}

/* Returns false and prints an error if the batch input directory has no WAV files. */
bool get_batch_wav_files(
    const AppConfig& config,
    std::vector<fs::path>& wav_files
) {
    wav_files = collect_wav_files(config.input_path);
    if (wav_files.empty()) {
        std::cerr << "No WAV files found in directory: "
                  << config.input_path << "\n";
        return false;
    }

    return true;
}

/* Converts named batch results into a plain vector of spectrogram outputs. */
std::vector<SpectrogramResult> strip_batch_result_names(
    const std::vector<std::pair<std::string, SpectrogramResult>>& results_with_names
) {
    std::vector<SpectrogramResult> results;
    results.reserve(results_with_names.size());

    for (const auto& item : results_with_names) {
        results.push_back(item.second);
    }

    return results;
}

/* Sums per-item GPU kernel timings for batch reporting. */
long long sum_gpu_kernel_times_ms(const std::vector<double>& gpu_times_ms) {
    long long total_kernel_ms = 0;

    for (double ms : gpu_times_ms) {
        total_kernel_ms += static_cast<long long>(ms);
    }

    return total_kernel_ms;
}

/* Prints the runtime summary for CPU batch execution. */
void print_batch_cpu_summary(
    const std::vector<fs::path>& wav_files,
    const TimingInfo& timing
) {
    std::cout << "\nCPU Batch Runtime Summary\n";
    std::cout << "-------------------------\n";
    std::cout << "Files processed:       "
              << wav_files.size() << "\n";
    std::cout << "WAV read time:         "
              << timing.wav_read_ms << " ms\n";
    std::cout << "Spectrogram time:      "
              << timing.spectrogram_ms << " ms\n";
    std::cout << "PNG write time:        "
              << timing.image_write_ms << " ms\n";
    std::cout << "Total runtime:         "
              << timing.total_ms << " ms\n";
}

/* Prints the runtime summary for OpenCL batch execution. */
void print_batch_gpu_summary(
    const std::vector<fs::path>& wav_files,
    const TimingInfo& timing,
    OpenCLKernelMode kernel_mode,
    const std::vector<double>& gpu_times_ms
) {
    std::cout << "\nOpenCL Batch Runtime Summary ("
              << kernel_mode_to_string(kernel_mode)
              << ")\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Files processed:       "
              << wav_files.size() << "\n";
    std::cout << "WAV read time:         "
              << timing.wav_read_ms << " ms\n";
    std::cout << "Spectrogram time:      "
              << timing.spectrogram_ms << " ms\n";
    std::cout << "PNG write time:        "
              << timing.image_write_ms << " ms\n";
    std::cout << "Total runtime:         "
              << timing.total_ms << " ms\n";

    if (!gpu_times_ms.empty()) {
        std::cout << "Summed GPU kernel ms:  "
                  << sum_gpu_kernel_times_ms(gpu_times_ms)
                  << " ms\n";
    }
}

/* Runs one CPU spectrogram computation and PNG write for a single batch file. */
bool process_cpu_batch_file(
    const fs::path& wav_path,
    const AppConfig& config,
    long long& total_wav_read_ms,
    long long& total_spec_ms,
    long long& total_png_write_ms
) {
    TimingInfo file_timing{};
    WavData wav;

    if (!load_wav_with_timing(wav_path.string(), wav, file_timing)) {
        return false;
    }
    total_wav_read_ms += file_timing.wav_read_ms;

    const auto spec_start = std::chrono::high_resolution_clock::now();
    SpectrogramResult spec = compute_spectrogram_cpu(
        wav.samples,
        config.window_size,
        config.hop_size,
        config.num_bins
    );
    const auto spec_end = std::chrono::high_resolution_clock::now();

    if (spec.values.empty()) {
        std::cerr << "CPU batch spectrogram computation failed for: "
                  << wav_path << "\n";
        return false;
    }

    total_spec_ms += std::chrono::duration_cast<
        std::chrono::milliseconds>(spec_end - spec_start).count();

    const std::string out_path =
        make_batch_output_path(wav_path, "cpu", OpenCLKernelMode::Local);

    if (!write_png_with_timing(out_path, spec, file_timing)) {
        return false;
    }

    total_png_write_ms += file_timing.image_write_ms;
    std::cout << "Wrote image: " << out_path << "\n";
    return true;
}

/* Stores accumulated batch timing totals into the output timing structure. */
void finalize_batch_timing(
    long long wav_read_ms,
    long long spectrogram_ms,
    long long image_write_ms,
    const std::chrono::high_resolution_clock::time_point& total_start,
    TimingInfo& timing
) {
    const auto total_end = std::chrono::high_resolution_clock::now();

    timing.wav_read_ms = wav_read_ms;
    timing.spectrogram_ms = spectrogram_ms;
    timing.image_write_ms = image_write_ms;
    timing.total_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(total_end - total_start).count();
}

/* Runs the batch pipeline sequentially on the CPU for every WAV file in a directory. */
bool run_batch_cpu(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    std::vector<fs::path> wav_files;
    if (!get_batch_wav_files(config, wav_files)) {
        return false;
    }

    const auto total_start = std::chrono::high_resolution_clock::now();

    long long total_wav_read_ms = 0;
    long long total_spec_ms = 0;
    long long total_png_write_ms = 0;

    for (const auto& wav_path : wav_files) {
        if (!process_cpu_batch_file(
                wav_path,
                config,
                total_wav_read_ms,
                total_spec_ms,
                total_png_write_ms)) {
            return false;
        }
    }

    finalize_batch_timing(
        total_wav_read_ms,
        total_spec_ms,
        total_png_write_ms,
        total_start,
        timing
    );

    print_batch_cpu_summary(wav_files, timing);
    return true;
}

/*
 * Runs the batch pipeline using the OpenCL batch interface, then writes
 * one PNG per returned spectrogram result.
 */
bool run_batch_gpu(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    std::vector<fs::path> wav_files;
    if (!get_batch_wav_files(config, wav_files)) {
        return false;
    }

    std::vector<std::pair<std::string, std::vector<float>>> batch_inputs;
    std::vector<double> gpu_times_ms;

    const auto total_start = std::chrono::high_resolution_clock::now();

    if (!load_batch_wavs(
            wav_files,
            batch_inputs,
            timing.wav_read_ms)) {
        return false;
    }

    const auto spec_start = std::chrono::high_resolution_clock::now();
    const auto results_with_names = compute_spectrogram_opencl_batch(
        batch_inputs,
        config.window_size,
        config.hop_size,
        config.num_bins,
        &gpu_times_ms,
        config.kernel_mode
    );
    const auto spec_end = std::chrono::high_resolution_clock::now();

    if (results_with_names.size() != wav_files.size()) {
        std::cerr << "Batch OpenCL computation failed.\n";
        return false;
    }

    const std::vector<SpectrogramResult> results =
        strip_batch_result_names(results_with_names);

    timing.spectrogram_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            spec_end - spec_start).count();

    if (!write_batch_pngs(
            wav_files,
            results,
            "gpu",
            config.kernel_mode,
            timing.image_write_ms)) {
        return false;
    }

    finalize_batch_timing(
        timing.wav_read_ms,
        timing.spectrogram_ms,
        timing.image_write_ms,
        total_start,
        timing
    );

    print_batch_gpu_summary(
        wav_files,
        timing,
        config.kernel_mode,
        gpu_times_ms
    );
    return true;
}

/* Runs both batch CPU and batch GPU modes for comparison on the same directory input. */
bool run_batch_both(const AppConfig& base_config) {
    AppConfig cpu_config = base_config;
    cpu_config.mode = "cpu";

    AppConfig gpu_config = base_config;
    gpu_config.mode = "gpu";

    TimingInfo cpu_timing{};
    TimingInfo gpu_timing{};

    std::cout << "\n=== Running Batch CPU ===\n";
    if (!run_batch_cpu(cpu_config, cpu_timing)) {
        return false;
    }

    std::cout << "\n=== Running Batch GPU ("
              << kernel_mode_to_string(gpu_config.kernel_mode)
              << ") ===\n";
    if (!run_batch_gpu(gpu_config, gpu_timing)) {
        return false;
    }

    print_comparison(cpu_timing, gpu_timing);
    return true;
}

/* Dispatches batch execution based on CPU, GPU, or comparison mode. */
bool run_batch_mode(const AppConfig& config) {
    if (config.mode == "cpu") {
        TimingInfo timing{};
        return run_batch_cpu(config, timing);
    }

    if (config.mode == "gpu") {
        TimingInfo timing{};
        return run_batch_gpu(config, timing);
    }

    return run_batch_both(config);
}

/* Dispatches single-file execution based on CPU, GPU, or comparison mode. */
bool run_single_mode(const AppConfig& config) {
    if (config.mode == "both") {
        return run_both(config);
    }

    TimingInfo timing{};
    return run_pipeline(config, timing);
}

/*
 * Program entry point.
 * Dispatches to single-file or batch processing based on whether the input path
 * is a file or directory, and then selects CPU, GPU, or comparison mode.
 */
int main(int argc, char** argv) {
    AppConfig config{};
    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    const bool ok = fs::is_directory(config.input_path)
        ? run_batch_mode(config)
        : run_single_mode(config);

    return ok ? 0 : 1;
}