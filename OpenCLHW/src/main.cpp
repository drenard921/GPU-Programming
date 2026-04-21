#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "image_writer.h"
#include "spectrogram_cpu.h"
#include "spectrogram_opencl.h"
#include "wav_reader.h"

namespace fs = std::filesystem;

struct AppConfig {
    std::string input_path;
    std::string mode = "cpu";
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

void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name
        << " <wav_file> [cpu|gpu] [window_size] [hop_size] [num_bins]\n\n"
        << "Examples:\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav cpu\n"
        << "  " << program_name
        << " Data/genres_original/blues/blues.00000.wav gpu 1024 512 512\n";
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

    if (argc >= 3) {
        config.mode = argv[2];
        if (!is_valid_mode(config.mode)) {
            std::cerr << "Invalid mode: " << config.mode << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (argc >= 4 && !parse_int_arg(argv[3], config.window_size)) {
        std::cerr << "Invalid window_size: " << argv[3] << "\n";
        return false;
    }

    if (argc >= 5 && !parse_int_arg(argv[4], config.hop_size)) {
        std::cerr << "Invalid hop_size: " << argv[4] << "\n";
        return false;
    }

    if (argc >= 6 && !parse_int_arg(argv[5], config.num_bins)) {
        std::cerr << "Invalid num_bins: " << argv[5] << "\n";
        return false;
    }

    return true;
}

bool ensure_output_directory(const std::string& output_prefix) {
    fs::path out_path(output_prefix);
    fs::path out_dir = out_path.parent_path();

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

std::string make_output_png_path(const AppConfig& config) {
    return config.output_prefix + "_" + config.mode + ".png";
}

void print_input_summary(const AppConfig& config, const WavData& wav) {
    const double duration_seconds =
        static_cast<double>(wav.samples.size()) / wav.sample_rate;

    std::cout << "Input: " << config.input_path << "\n";
    std::cout << "Mode: " << config.mode << "\n";
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
        spec = compute_spectrogram_opencl(
            wav.samples,
            config.window_size,
            config.hop_size,
            config.num_bins
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
        return false;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "Wrote image: " << output_png << "\n";
    print_timing_summary(
        config.mode == "cpu" ? "CPU" : "OpenCL",
        timing
    );

    return true;
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
        double total_speedup =
            static_cast<double>(cpu.total_ms) /
            static_cast<double>(gpu.total_ms);

        std::cout << "Pipeline speedup:     "
                  << total_speedup << "x\n";
    }
}

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

    std::cout << "\n=== Running GPU ===\n";
    if (!run_pipeline(gpu_config, gpu_timing)) {
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