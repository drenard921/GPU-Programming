#include "image_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

bool write_png(
    const std::string& filename,
    const std::vector<float>& values,
    int width,
    int height
) {
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid image dimensions.\n";
        return false;
    }

    if (static_cast<int>(values.size()) != width * height) {
        std::cerr << "Image data size does not match dimensions.\n";
        return false;
    }

    std::vector<std::uint8_t> image(width * height, 0);

    const float min_db = -60.0f;
    const float max_db = -10.0f;


    

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int src_index = x * height + y;
            int dst_index = (height - 1 - y) * width + x;

            float v = values[src_index];
            float db = 10.0f * std::log10(v + 1e-6f);

            db = std::clamp(db, min_db, max_db);

            float norm = (db - min_db) / (max_db - min_db);
            norm = std::clamp(norm, 0.0f, 1.0f);

            // boost contrast
            norm = std::pow(norm, 0.7f);

            // invert so background is dark
            // norm = 1.0f - norm;

            image[dst_index] =
                static_cast<std::uint8_t>(norm * 255.0f);
        }
    }

    const int success = stbi_write_png(
        filename.c_str(),
        width,
        height,
        1,
        image.data(),
        width
    );

    if (!success) {
        std::cerr << "Failed to write PNG: " << filename << "\n";
        return false;
    }

    return true;
}