#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include <string>
#include <vector>

// Writes a grayscale PNG image from float data
// values: spectrogram data (size = width * height)
// width: number of columns (frames)
// height: number of rows (frequency bins)
bool write_png(
    const std::string& filename,
    const std::vector<float>& values,
    int width,
    int height
);

#endif