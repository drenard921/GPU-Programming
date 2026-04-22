#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "image_writer.h"
#include "spectrogram_cpu.h"
#include "spectrogram_opencl.h"
#include "wav_reader.h"

namespace fs = std::filesystem;

struct AppConfig {
    std::string input_path;
    std::string mode = "cpu";
    OpenCLKernelMode kernel_mode = OpenCLKernelMode::Local;
    int window_size = 1024;
    int hop_size = 512;
    int num_bins = 512;
    std::string output_prefix = "output/output";
};

struct TimingInfo {
    long long wav_read_ms = 0;
    long long spectrogram_ms = 0;
    long long image_write_ms = 0;
    long long total_ms = 0;
};

bool is_valid_mode(const std::string& mode) {
    return mode == "cpu" || mode == "gpu" || mode == "both";
}

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

std::string kernel_mode_to_string(OpenCLKernelMode kernel_mode) {
    return kernel_mode == OpenCLKernelMode::Naive ? "naive" : "local";
}

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
        << " Data/genres_original/blues/blues.00000.wav gpu local\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav gpu naive\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav both local\n"
        << "  " << program_name
        << " Data/genres_original/blues gpu naive 1024 512 512\n"
        << "  " << program_name
        << " Data/genres_original/blues local\n";
}

bool parse_int_arg(const char* text, int& value) {
    try {
        value = std::stoi(text);
        return value > 0;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char** argv, AppConfig& config) {
    if (argc < 2) {
        print_usage(argv[0]);
        return false;
    }

    config.input_path = argv[1];

    int arg_index = 2;

    if (argc > arg_index) {
        const std::string candidate = argv[arg_index];
        if (is_valid_mode(candidate)) {
            config.mode = candidate;
            ++arg_index;
        }
    }

    if (config.mode == "gpu" || config.mode == "both") {
        if (argc > arg_index) {
            const std::string candidate = argv[arg_index];
            OpenCLKernelMode parsed_mode = OpenCLKernelMode::Local;

            if (parse_kernel_mode_arg(candidate, parsed_mode)) {
                config.kernel_mode = parsed_mode;
                ++arg_index;
            }
        }
    }

    if (argc > arg_index &&
        !parse_int_arg(argv[arg_index], config.window_size)) {
        std::cerr << "Invalid window_size: " << argv[arg_index] << "\n";
        return false;
    }
    if (argc > arg_index) {
        ++arg_index;
    }

    if (argc > arg_index &&
        !parse_int_arg(argv[arg_index], config.hop_size)) {
        std::cerr << "Invalid hop_size: " << argv[arg_index] << "\n";
        return false;
    }
    if (argc > arg_index) {
        ++arg_index;
    }

    if (argc > arg_index &&
        !parse_int_arg(argv[arg_index], config.num_bins)) {
        std::cerr << "Invalid num_bins: " << argv[arg_index] << "\n";
        return false;
    }
    if (argc > arg_index) {
        ++arg_index;
    }

    if (argc > arg_index) {
        std::cerr << "Too many arguments.\n";
        print_usage(argv[0]);
        return false;
    }

    return true;
}

bool ensure_output_directory(const std::string& output_prefix) {
    fs::path out_path(output_prefix);
    fs::path out_dir = out_path.parent_path();

    if (out_dir.empty()) {
        out_dir = "output";
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

std::string make_output_png_path(const AppConfig& config) {
    if (config.mode == "gpu") {
        return config.output_prefix + "_gpu_" +
               kernel_mode_to_string(config.kernel_mode) + ".png";
    }

    return config.output_prefix + "_" + config.mode + ".png";
}

std::string make_batch_output_path(
    const fs::path& input_file,
    const std::string& mode,
    OpenCLKernelMode kernel_mode
) {
    if (mode == "gpu") {
        return "output/" + input_file.stem().string() + "_" + mode + "_" +
               kernel_mode_to_string(kernel_mode) + ".png";
    }

    return "output/" + input_file.stem().string() + "_" + mode + ".png";
}

void print_input_summary(const AppConfig& config, const WavData& wav) {
    const double duration_seconds =
        static_cast<double>(wav.samples.size()) / wav.sample_rate;

    std::cout << "Input: " << config.input_path << "\n";
    std::cout << "Mode: " << config.mode << "\n";
    if (config.mode == "gpu" || config.mode == "both") {
        std::cout << "OpenCL kernel mode: "
                  << kernel_mode_to_string(config.kernel_mode) << "\n";
    }
    std::cout << "Sample rate: " << wav.sample_rate << "\n";
    std::cout << "Channels: " << wav.channels << "\n";
    std::cout << "Samples: " << wav.samples.size() << "\n";
    std::cout << "Duration (s): " << duration_seconds << "\n";
    std::cout << "Window size: " << config.window_size << "\n";
    std::cout << "Hop size: " << config.hop_size << "\n";
    std::cout << "Bins: " << config.num_bins << "\n";
}

void print_spectrogram_summary(const SpectrogramResult& spec) {
    std::cout << "Frames: " << spec.num_frames << "\n";
    std::cout << "Bins: " << spec.num_bins << "\n";
    std::cout << "Spectrogram values: " << spec.values.size() << "\n";
}

void print_timing_summary(
    const std::string& label,
    const TimingInfo& timing
) {
    std::cout << "\n" << label << " Runtime Summary\n";
    std::cout << "-------------------------\n";
    std::cout << "WAV read time:        "
              << timing.wav_read_ms << " ms\n";
    std::cout << "Spectrogram time:     "
              << timing.spectrogram_ms << " ms\n";
    std::cout << "PNG write time:       "
              << timing.image_write_ms << " ms\n";
    std::cout << "Total runtime:        "
              << timing.total_ms << " ms\n";
}

void print_comparison(
    const TimingInfo& cpu,
    const TimingInfo& gpu
) {
    std::cout << "\nComparison\n";
    std::cout << "----------\n";

    if (gpu.spectrogram_ms > 0) {
        double compute_speedup =
            static_cast<double>(cpu.spectrogram_ms) /
            static_cast<double>(gpu.spectrogram_ms);

        std::cout << "Compute speedup:      "
                  << compute_speedup << "x\n";
    }

    if (gpu.total_ms > 0) {
        double pipeline_speedup =
            static_cast<double>(cpu.total_ms) /
            static_cast<double>(gpu.total_ms);

        std::cout << "Pipeline speedup:     "
                  << pipeline_speedup << "x\n";
    }
}

bool load_wav_with_timing(
    const std::string& input_path,
    WavData& wav,
    TimingInfo& timing
) {
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = read_wav_file(input_path, wav);
    auto end = std::chrono::high_resolution_clock::now();

    timing.wav_read_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    return ok;
}

bool compute_spectrogram_with_timing(
    const AppConfig& config,
    const WavData& wav,
    SpectrogramResult& spec,
    TimingInfo& timing
) {
    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();

    timing.spectrogram_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    if (spec.values.empty() || spec.num_frames <= 0 || spec.num_bins <= 0) {
        std::cerr << "Spectrogram computation failed.\n";
        return false;
    }

    return true;
}

bool write_png_with_timing(
    const std::string& output_path,
    const SpectrogramResult& spec,
    TimingInfo& timing
) {
    auto start = std::chrono::high_resolution_clock::now();

    bool ok = write_png(
        output_path,
        spec.values,
        spec.num_frames,
        spec.num_bins
    );

    auto end = std::chrono::high_resolution_clock::now();

    timing.image_write_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    return ok;
}

bool run_pipeline(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

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

    const std::string output_png = make_output_png_path(config);
    if (!write_png_with_timing(output_png, spec, timing)) {
        std::cerr << "Failed to write PNG: " << output_png << "\n";
        return false;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "Wrote image: " << output_png << "\n";

    std::string label = "CPU";
    if (config.mode == "gpu") {
        label = "OpenCL (" + kernel_mode_to_string(config.kernel_mode) + ")";
    }

    print_timing_summary(label, timing);
    return true;
}

bool run_both(const AppConfig& base_config) {
    AppConfig cpu_config = base_config;
    cpu_config.mode = "cpu";
    cpu_config.output_prefix = "output/output";

    AppConfig gpu_config = base_config;
    gpu_config.mode = "gpu";
    gpu_config.output_prefix = "output/output";

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

std::vector<fs::path> collect_wav_files(const std::string& directory_path) {
    std::vector<fs::path> wav_files;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        if (entry.path().extension() == ".wav") {
            wav_files.push_back(entry.path());
        }
    }

    std::sort(wav_files.begin(), wav_files.end());
    return wav_files;
}

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
        batch_inputs.push_back({wav_path.filename().string(), std::move(wav.samples)});
    }

    return true;
}

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

bool run_batch_cpu(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    const auto wav_files = collect_wav_files(config.input_path);
    if (wav_files.empty()) {
        std::cerr << "No WAV files found in directory: "
                  << config.input_path << "\n";
        return false;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    long long total_wav_read_ms = 0;
    long long total_spec_ms = 0;
    long long total_png_write_ms = 0;

    for (const auto& wav_path : wav_files) {
        TimingInfo file_timing{};
        WavData wav;

        if (!load_wav_with_timing(wav_path.string(), wav, file_timing)) {
            return false;
        }
        total_wav_read_ms += file_timing.wav_read_ms;

        auto spec_start = std::chrono::high_resolution_clock::now();
        SpectrogramResult spec = compute_spectrogram_cpu(
            wav.samples,
            config.window_size,
            config.hop_size,
            config.num_bins
        );
        auto spec_end = std::chrono::high_resolution_clock::now();

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
    }

    auto total_end = std::chrono::high_resolution_clock::now();

    timing.wav_read_ms = total_wav_read_ms;
    timing.spectrogram_ms = total_spec_ms;
    timing.image_write_ms = total_png_write_ms;
    timing.total_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(total_end - total_start).count();

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

    return true;
}

bool run_batch_gpu(const AppConfig& config, TimingInfo& timing) {
    if (!ensure_output_directory(config.output_prefix)) {
        return false;
    }

    const auto wav_files = collect_wav_files(config.input_path);
    if (wav_files.empty()) {
        std::cerr << "No WAV files found in directory: "
                  << config.input_path << "\n";
        return false;
    }

    std::vector<std::pair<std::string, std::vector<float>>> batch_inputs;

    auto total_start = std::chrono::high_resolution_clock::now();

    if (!load_batch_wavs(
            wav_files,
            batch_inputs,
            timing.wav_read_ms)) {
        return false;
    }

    std::vector<double> gpu_times_ms;

    auto spec_start = std::chrono::high_resolution_clock::now();
    auto results_with_names = compute_spectrogram_opencl_batch(
        batch_inputs,
        config.window_size,
        config.hop_size,
        config.num_bins,
        &gpu_times_ms,
        config.kernel_mode
    );
    auto spec_end = std::chrono::high_resolution_clock::now();

    if (results_with_names.size() != wav_files.size()) {
        std::cerr << "Batch OpenCL computation failed.\n";
        return false;
    }

    std::vector<SpectrogramResult> results;
    results.reserve(results_with_names.size());
    for (const auto& item : results_with_names) {
        results.push_back(item.second);
    }

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

    auto total_end = std::chrono::high_resolution_clock::now();

    timing.total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start).count();

    std::cout << "\nOpenCL Batch Runtime Summary ("
              << kernel_mode_to_string(config.kernel_mode)
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
        long long total_kernel_ms = 0;
        for (double ms : gpu_times_ms) {
            total_kernel_ms += static_cast<long long>(ms);
        }

        std::cout << "Summed GPU kernel ms:  "
                  << total_kernel_ms << " ms\n";
    }

    return true;
}

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

int main(int argc, char** argv) {
    AppConfig config{};
    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    if (fs::is_directory(config.input_path)) {
        if (config.mode == "cpu") {
            TimingInfo timing{};
            if (!run_batch_cpu(config, timing)) {
                return 1;
            }
        } else if (config.mode == "gpu") {
            TimingInfo timing{};
            if (!run_batch_gpu(config, timing)) {
                return 1;
            }
        } else if (config.mode == "both") {
            if (!run_batch_both(config)) {
                return 1;
            }
        }
        return 0;
    }

    if (config.mode == "both") {
        if (!run_both(config)) {
            return 1;
        }
    } else {
        TimingInfo timing{};
        if (!run_pipeline(config, timing)) {
            return 1;
        }
    }

    return 0;
}