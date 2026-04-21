#include "wav_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct WavHeader {
    char riff[4];
    std::uint32_t chunk_size;
    char wave[4];
};

struct ChunkHeader {
    char id[4];
    std::uint32_t size;
};

bool matches_id(const char id[4], const char* expected) {
    return id[0] == expected[0] &&
           id[1] == expected[1] &&
           id[2] == expected[2] &&
           id[3] == expected[3];
}

std::string chunk_id_to_string(const char id[4]) {
    return std::string(id, id + 4);
}

}  // namespace

bool read_wav_file(const std::string& path, WavData& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open WAV file: " << path << "\n";
        return false;
    }

    WavHeader wav_header{};
    file.read(reinterpret_cast<char*>(&wav_header), sizeof(wav_header));
    if (!file) {
        std::cerr << "Failed to read WAV header: " << path << "\n";
        return false;
    }

    if (!matches_id(wav_header.riff, "RIFF") ||
        !matches_id(wav_header.wave, "WAVE")) {
        std::cerr << "Invalid WAV file (missing RIFF/WAVE): "
                  << path << "\n";
        return false;
    }

    std::uint16_t audio_format = 0;
    std::uint16_t num_channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<std::int16_t> pcm_samples;

    bool found_fmt = false;
    bool found_data = false;

    while (file && (!found_fmt || !found_data)) {
        ChunkHeader chunk{};
        file.read(reinterpret_cast<char*>(&chunk), sizeof(chunk));
        if (!file) {
            break;
        }

        if (matches_id(chunk.id, "fmt ")) {
            found_fmt = true;

            std::uint16_t block_align = 0;
            std::uint32_t byte_rate = 0;

            file.read(reinterpret_cast<char*>(&audio_format),
                      sizeof(audio_format));
            file.read(reinterpret_cast<char*>(&num_channels),
                      sizeof(num_channels));
            file.read(reinterpret_cast<char*>(&sample_rate),
                      sizeof(sample_rate));
            file.read(reinterpret_cast<char*>(&byte_rate),
                      sizeof(byte_rate));
            file.read(reinterpret_cast<char*>(&block_align),
                      sizeof(block_align));
            file.read(reinterpret_cast<char*>(&bits_per_sample),
                      sizeof(bits_per_sample));

            if (!file) {
                std::cerr << "Failed reading fmt chunk in: "
                          << path << "\n";
                return false;
            }

            const std::uint32_t bytes_read = 16;
            if (chunk.size > bytes_read) {
                file.seekg(chunk.size - bytes_read, std::ios::cur);
            }
        } else if (matches_id(chunk.id, "data")) {
            found_data = true;

            if (bits_per_sample != 16) {
                std::cerr << "Only 16-bit PCM WAV is supported. File: "
                          << path << "\n";
                return false;
            }

            const std::size_t sample_count =
                chunk.size / sizeof(std::int16_t);
            pcm_samples.resize(sample_count);

            file.read(reinterpret_cast<char*>(pcm_samples.data()),
                      static_cast<std::streamsize>(chunk.size));
            if (!file) {
                std::cerr << "Failed reading data chunk in: "
                          << path << "\n";
                return false;
            }
        } else {
            file.seekg(chunk.size, std::ios::cur);
        }

        if (chunk.size % 2 == 1) {
            file.seekg(1, std::ios::cur);
        }
    }

    if (!found_fmt) {
        std::cerr << "Missing fmt chunk in WAV file: " << path << "\n";
        return false;
    }

    if (!found_data) {
        std::cerr << "Missing data chunk in WAV file: " << path << "\n";
        return false;
    }

    if (audio_format != 1) {
        std::cerr << "Only PCM WAV is supported. File: "
                  << path << "\n";
        return false;
    }

    if (num_channels == 0 || sample_rate == 0) {
        std::cerr << "Invalid WAV metadata in file: " << path << "\n";
        return false;
    }

    out.sample_rate = static_cast<int>(sample_rate);
    out.channels = static_cast<int>(num_channels);
    out.samples.clear();

    if (num_channels == 1) {
        out.samples.reserve(pcm_samples.size());
        for (std::int16_t s : pcm_samples) {
            out.samples.push_back(static_cast<float>(s) / 32768.0f);
        }
    } else {
        const std::size_t frame_count = pcm_samples.size() / num_channels;
        out.samples.reserve(frame_count);

        for (std::size_t i = 0; i < frame_count; ++i) {
            float sum = 0.0f;
            for (std::size_t ch = 0; ch < num_channels; ++ch) {
                const std::size_t idx = i * num_channels + ch;
                sum += static_cast<float>(pcm_samples[idx]) / 32768.0f;
            }
            out.samples.push_back(sum / static_cast<float>(num_channels));
        }

        out.channels = 1;
    }

    return true;
}