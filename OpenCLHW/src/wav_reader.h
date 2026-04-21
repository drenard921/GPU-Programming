#ifndef WAV_READER_H
#define WAV_READER_H

#include <string>
#include <vector>

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

bool read_wav_file(const std::string& path, WavData& out);

#endif