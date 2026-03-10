// Dylan Renard
// EN.605.617 Introduction to GPU Programming (JHU)
// Professor Chance Pascale
// March 9th, 2026
//
// CUDA Streams and Events Assignment
// Dual GBA Emulator Frame Processing with Concurrent CUDA Streams
//
// Overview:
// This program demonstrates practical usage of CUDA Streams and CUDA Events by
// processing video frames from two Game Boy Advance emulator instances in parallel.
// Each emulator produces RGB565 frames which are transferred to the GPU where
// multiple CUDA kernels perform image upscaling, visual comparison, and metric
// extraction in real time.
//
// The application renders a 2×3 visualization panel showing the GPU-processed
// results while simultaneously collecting performance measurements for analysis.
//
// CUDA Concepts Demonstrated:
//   • CUDA Streams      - Two independent streams process FireRed and LeafGreen
//                         frames concurrently to illustrate asynchronous execution.
//   • CUDA Events       - Kernel timing is measured using events to record
//                         execution durations for key GPU operations.
//   • Global Memory     - Frame buffers, intermediate images, and composite
//                         display outputs reside in device global memory.
//   • Registers         - Per-thread computations inside image processing kernels.
//   • Atomic Operations - Used for reduction metrics (heatmap statistics).
//
// GPU Processing Pipeline:
//   1. Two libretro emulator instances produce RGB565 video frames.
//   2. Frames are copied Host → Device asynchronously on separate CUDA streams.
//   3. Bilinear upscaling kernels run concurrently on each stream.
//   4. Nearest-neighbor upscales are computed as control comparisons.
//   5. Heatmap kernels compute visual differences between:
//        • Bilinear vs nearest (self comparison)
//        • FireRed vs LeafGreen frames (cross comparison)
//   6. Reduction kernels compute summary metrics such as mean heatmap intensity
//      and nonzero difference counts.
//   7. A CUDA kernel composes the six processed panels into a single output frame.
//   8. The composite image is copied Device → Host and rendered using SDL.
//
// Visualization Layout (2×3 Panel):
//
//   Row 1 – GPU Bilinear Upscaling
//      [ FireRed Bilinear ]   [ LeafGreen Bilinear ]
//
//   Row 2 – Control Comparison (Nearest Neighbor)
//      [ FireRed Nearest ]    [ LeafGreen Nearest ]
//
//   Row 3 – Data Analysis Views
//      [ Bilinear vs Nearest Heatmap ]   [ FireRed vs LeafGreen Heatmap ]
//
// Performance Instrumentation:
//   CUDA events measure kernel execution time for:
//      • FireRed bilinear upscale
//      • LeafGreen bilinear upscale
//      • Self comparison heatmap
//      • Cross-game difference heatmap
//
//   Metrics are logged per frame and written to a CSV file including:
//      frame index, thread/block configuration,
//      kernel timings, heatmap statistics,
//      and pixel-level difference counts.
//
// Emulator Controls:
//   Keyboard inputs are mapped to Game Boy Advance controls and applied
//   simultaneously to both emulator instances so gameplay remains synchronized.
//
//      W  = D-Pad Up
//      A  = D-Pad Left
//      S  = D-Pad Down
//      D  = D-Pad Right
//
//      Keypad 4 = A Button
//      Keypad 5 = B Button
//      Keypad 2 = Start
//      Keypad 3 = Select
//
//      Keypad 7 = L Shoulder
//      Keypad 9 = R Shoulder
//
//      ESC = Exit program
//
// Command-Line Usage:
//   ./streams_hw
//        [--core <mgba_libretro.so>]
//        [--fr <FireRed.gba>]
//        [--lg <LeafGreen.gba>]
//        [--scale N]
//        [--threads N]
//        [--blocks N]
//        [--frames N]
//        [--print-every N]
//        [--csv output.csv]
//
// Example:
//   ./streams_hw --scale 3 --threads 256 --frames 1000
//
// Output:
//   • Real-time SDL window displaying GPU processed frames.
//   • CSV performance log for analysis of CUDA stream concurrency
//     and kernel timing behavior.

#include <fstream>
#include <SDL2/SDL.h>
#include <dlfcn.h>
#include <link.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#include "libretro.h"

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
  } \
} while(0)

static constexpr int GBA_WIDTH = 240;
static constexpr int GBA_HEIGHT = 160;
static constexpr int DEFAULT_SCALE = 3;
static constexpr int MIN_THREADS = 64;
static constexpr int MAX_THREADS = 1024;
static constexpr int DEFAULT_THREADS = 256;
static constexpr int DEFAULT_PRINT_EVERY = 60;
static constexpr int AUDIO_BUFFER_SAMPLES = 1024;
static constexpr int DIFF_THRESHOLD = 8;
static constexpr int HEAT_GAIN = 6;
static constexpr int BLOCK2D_X = 16;
static constexpr int BLOCK2D_Y = 16;


// ---------------- Controls --------------------
static uint32_t g_keys = 0;

static constexpr uint32_t KEY_A      = (1u << 0);
static constexpr uint32_t KEY_B      = (1u << 1);
static constexpr uint32_t KEY_SELECT = (1u << 2);
static constexpr uint32_t KEY_START  = (1u << 3);
static constexpr uint32_t KEY_RIGHT  = (1u << 4);
static constexpr uint32_t KEY_LEFT   = (1u << 5);
static constexpr uint32_t KEY_UP     = (1u << 6);
static constexpr uint32_t KEY_DOWN   = (1u << 7);
static constexpr uint32_t KEY_R      = (1u << 8);
static constexpr uint32_t KEY_L      = (1u << 9);

static uint32_t build_synced_keymask_from_sdl() {
    SDL_PumpEvents();
    const Uint8* st = SDL_GetKeyboardState(nullptr);
    uint32_t keys = 0;

    if (st[SDL_SCANCODE_W]) keys |= KEY_UP;
    if (st[SDL_SCANCODE_S]) keys |= KEY_DOWN;
    if (st[SDL_SCANCODE_A]) keys |= KEY_LEFT;
    if (st[SDL_SCANCODE_D]) keys |= KEY_RIGHT;

    if (st[SDL_SCANCODE_KP_4]) keys |= KEY_A;
    if (st[SDL_SCANCODE_KP_5]) keys |= KEY_B;
    if (st[SDL_SCANCODE_KP_2]) keys |= KEY_START;
    if (st[SDL_SCANCODE_KP_3]) keys |= KEY_SELECT;
    if (st[SDL_SCANCODE_KP_7]) keys |= KEY_L;
    if (st[SDL_SCANCODE_KP_9]) keys |= KEY_R;

    return keys;
}

// ---------------- Libretro API --------------------
struct RetroAPI {
    void (*retro_init)();
    void (*retro_deinit)();
    unsigned (*retro_api_version)();

    void (*retro_set_environment)(retro_environment_t);
    void (*retro_set_video_refresh)(retro_video_refresh_t);
    void (*retro_set_audio_sample)(retro_audio_sample_t);
    void (*retro_set_audio_sample_batch)(retro_audio_sample_batch_t);
    void (*retro_set_input_poll)(retro_input_poll_t);
    void (*retro_set_input_state)(retro_input_state_t);

    void (*retro_get_system_av_info)(retro_system_av_info*);
    bool (*retro_load_game)(const retro_game_info*);
    void (*retro_unload_game)();
    void (*retro_run)();
};

struct Instance {
    void* so = nullptr;
    RetroAPI api{};

    std::vector<uint16_t> frame565;
    unsigned w = 0, h = 0;
    size_t pitch = 0;

    std::string rom_path;
};

static thread_local Instance* tls_current = nullptr;
static Instance* g_audio_source = nullptr;

// ---------------- Audio --------------------
static SDL_AudioDeviceID g_audio_dev = 0;
static std::vector<int16_t> g_audio_ring;
static size_t g_audio_r = 0, g_audio_w = 0;

static void audio_ring_write(const int16_t* src, size_t count) {
    if (g_audio_ring.empty() || !src || count == 0) return;
    size_t cap = g_audio_ring.size();
    for (size_t i = 0; i < count; i++) {
        g_audio_ring[g_audio_w] = src[i];
        g_audio_w = (g_audio_w + 1) % cap;
        if (g_audio_w == g_audio_r) g_audio_r = (g_audio_r + 1) % cap;
    }
}

static void sdl_audio_cb(void*, Uint8* stream, int len_bytes) {
    std::memset(stream, 0, (size_t)len_bytes);
    if (g_audio_ring.empty()) return;

    int16_t* out = (int16_t*)stream;
    size_t want = (size_t)len_bytes / sizeof(int16_t);
    size_t cap = g_audio_ring.size();

    for (size_t i = 0; i < want; i++) {
        if (g_audio_r == g_audio_w) break;
        out[i] = g_audio_ring[g_audio_r];
        g_audio_r = (g_audio_r + 1) % cap;
    }
}

// ---------------- Libretro callbacks --------------------
/**
 * @brief Video frame callback used by the emulator to deliver rendered frames.
 *
 * This function is invoked by the emulator core every time a new video frame
 * is produced. It copies the raw frame buffer into a thread-local structure
 * (`tls_current`) so that it can later be processed or transferred to GPU
 * memory for CUDA-based post-processing (e.g., scaling or filtering).
 *
 * The incoming frame is stored in RGB565 format. Because the emulator may
 * include padding between rows, the function copies the frame row-by-row
 * using the provided pitch value.
 *
 * @param data   Pointer to the raw frame buffer produced by the emulator.
 * @param width  Width of the frame in pixels.
 * @param height Height of the frame in pixels.
 * @param pitch  Number of bytes between the start of successive rows in the
 *               source buffer. This may be larger than width * sizeof(uint16_t)
 *               due to row padding.
 *
 * @note The frame data is copied into `tls_current->frame565`, which is resized
 *       to match the current frame dimensions.
 *
 * @warning The function safely exits if:
 *          - No thread-local context (`tls_current`) is available.
 *          - The input frame pointer is null.
 *          - The frame dimensions are invalid.
 */
static void video_refresh_cb(const void* data, unsigned width, unsigned height, size_t pitch) {
    if (!tls_current) return;
    if (!data || width == 0 || height == 0) return;

    tls_current->w = width;
    tls_current->h = height;
    tls_current->pitch = pitch;

    tls_current->frame565.resize((size_t)width * (size_t)height);
    const uint8_t* src = (const uint8_t*)data;
    for (unsigned y = 0; y < height; y++) {
        std::memcpy(&tls_current->frame565[(size_t)y * width],
                    src + (size_t)y * pitch,
                    (size_t)width * sizeof(uint16_t));
    }
}

static void audio_sample_cb(int16_t, int16_t) {}

static size_t audio_batch_cb(const int16_t* data, size_t frames) {
    if (!data) return frames;
    if (tls_current != g_audio_source) return frames;
    audio_ring_write(data, frames * 2);
    return frames;
}

static void input_poll_cb() {}

static int16_t input_state_cb(unsigned, unsigned device, unsigned, unsigned id) {
    if (device != RETRO_DEVICE_JOYPAD) return 0;
    switch (id) {
        case RETRO_DEVICE_ID_JOYPAD_A:      return (g_keys & KEY_A) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_B:      return (g_keys & KEY_B) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_SELECT: return (g_keys & KEY_SELECT) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_START:  return (g_keys & KEY_START) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_RIGHT:  return (g_keys & KEY_RIGHT) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_LEFT:   return (g_keys & KEY_LEFT) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_UP:     return (g_keys & KEY_UP) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_DOWN:   return (g_keys & KEY_DOWN) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_R:      return (g_keys & KEY_R) ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_L:      return (g_keys & KEY_L) ? 1 : 0;
        default: return 0;
    }
}

static bool environment_cb(unsigned cmd, void* data) {
    if (cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT) {
        auto* fmt = (retro_pixel_format*)data;
        *fmt = RETRO_PIXEL_FORMAT_RGB565;
        return true;
    }
    return false;
}

// ---------------- dl helpers --------------------
static void* must_dlsym(void* so, const char* name) {
    void* p = dlsym(so, name);
    if (!p) throw std::runtime_error(std::string("dlsym failed: ") + name);
    return p;
}

static RetroAPI load_retro_api(void* so) {
    RetroAPI a{};
    a.retro_init = (void(*)())must_dlsym(so, "retro_init");
    a.retro_deinit = (void(*)())must_dlsym(so, "retro_deinit");
    a.retro_api_version = (unsigned(*)())must_dlsym(so, "retro_api_version");

    a.retro_set_environment = (void(*)(retro_environment_t))must_dlsym(so, "retro_set_environment");
    a.retro_set_video_refresh = (void(*)(retro_video_refresh_t))must_dlsym(so, "retro_set_video_refresh");
    a.retro_set_audio_sample = (void(*)(retro_audio_sample_t))must_dlsym(so, "retro_set_audio_sample");
    a.retro_set_audio_sample_batch = (void(*)(retro_audio_sample_batch_t))must_dlsym(so, "retro_set_audio_sample_batch");
    a.retro_set_input_poll = (void(*)(retro_input_poll_t))must_dlsym(so, "retro_set_input_poll");
    a.retro_set_input_state = (void(*)(retro_input_state_t))must_dlsym(so, "retro_set_input_state");

    a.retro_get_system_av_info = (void(*)(retro_system_av_info*))must_dlsym(so, "retro_get_system_av_info");
    a.retro_load_game = (bool(*)(const retro_game_info*))must_dlsym(so, "retro_load_game");
    a.retro_unload_game = (void(*)())must_dlsym(so, "retro_unload_game");
    a.retro_run = (void(*)())must_dlsym(so, "retro_run");
    return a;
}


/**
 * @brief Creates and initializes a new emulator instance for a given ROM.
 *
 * This function dynamically loads a libretro emulator core and configures it
 * to run a specific ROM. A new dynamic linker namespace is created using
 * `dlmopen`, which allows multiple emulator cores to run simultaneously
 * without symbol conflicts. This is important for scenarios where multiple
 * emulation instances are executed concurrently (e.g., parallel execution
 * using CUDA streams).
 *
 * After loading the core, the function retrieves the libretro API and registers
 * all required callback functions for environment setup, video output, audio
 * output, and input handling.
 *
 * The thread-local pointer `tls_current` is temporarily set to the newly created
 * instance so that callback functions can correctly associate emulator events
 * with the proper instance.
 *
 * Finally, the emulator core is initialized and the specified ROM is loaded.
 *
 * @param core_so  Path to the libretro core shared library (.so).
 * @param rom_path Path to the ROM file that will be executed by the emulator.
 *
 * @return Instance A fully initialized emulator instance ready to run frames.
 *
 * @throws std::runtime_error If:
 *         - The emulator core fails to load via `dlmopen`
 *         - The ROM fails to load via `retro_load_game`
 *
 * @note `dlmopen` with `LM_ID_NEWLM` creates an isolated linker namespace,
 *       allowing multiple emulator cores to be loaded independently in the
 *       same process.
 *
 * @warning The thread-local pointer `tls_current` must be correctly set before
 *          calling libretro initialization functions so callbacks operate on
 *          the correct instance.
 */
static Instance make_instance(const std::string& core_so, const std::string& rom_path) {
    Instance inst;
    inst.rom_path = rom_path;

    inst.so = dlmopen(LM_ID_NEWLM, core_so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!inst.so) throw std::runtime_error(std::string("dlmopen failed: ") + dlerror());

    inst.api = load_retro_api(inst.so);

    inst.api.retro_set_environment(environment_cb);
    inst.api.retro_set_video_refresh(video_refresh_cb);
    inst.api.retro_set_audio_sample(audio_sample_cb);
    inst.api.retro_set_audio_sample_batch(audio_batch_cb);
    inst.api.retro_set_input_poll(input_poll_cb);
    inst.api.retro_set_input_state(input_state_cb);

    tls_current = &inst;
    inst.api.retro_init();

    retro_game_info info{};
    info.path = inst.rom_path.c_str();
    if (!inst.api.retro_load_game(&info)) {
        throw std::runtime_error("retro_load_game failed: " + inst.rom_path);
    }

    tls_current = nullptr;
    return inst;
}

static void destroy_instance(Instance& inst) {
    if (!inst.so) return;
    tls_current = &inst;
    inst.api.retro_unload_game();
    inst.api.retro_deinit();
    tls_current = nullptr;
    dlclose(inst.so);
    inst.so = nullptr;
}

// ---------------- Args --------------------
struct Args {
    std::string core_so = "./libretro-super/dist/unix/mgba_libretro.so";
    std::string fr = "./GBAROMS/FireRed.gba";
    std::string lg = "./GBAROMS/LeafGreen.gba";
    std::string csv = "streams_metrics.csv";
    int frames = -1;
    int scale = DEFAULT_SCALE;
    int threads = DEFAULT_THREADS;
    int blocks = 0;
    int print_every = DEFAULT_PRINT_EVERY;
};



/**
 * @brief Parses command-line arguments for the emulator CUDA processing program.
 *
 * This function processes command-line parameters and stores them in an `Args`
 * structure used to configure the emulator execution and CUDA kernel behavior.
 * Supported options allow users to specify ROM paths, emulator cores, runtime
 * parameters, CUDA execution settings, and output logging.
 *
 * Each argument that requires a value is validated using a helper lambda (`need`)
 * which ensures a parameter follows the option flag.
 *
 * After parsing, several sanity checks are applied to enforce valid ranges for
 * scaling factors, thread counts, and printing intervals.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 *
 * @return Args Structure containing all parsed runtime configuration options.
 *
 * @throws std::runtime_error If:
 *         - An argument requiring a value is missing one
 *         - An unknown command-line argument is encountered
 *
 * @note Supported command-line arguments:
 *       --core <file>        Path to the libretro core shared library
 *       --fr / --rom1        Path to FireRed ROM
 *       --lg / --rom2        Path to LeafGreen ROM
 *       --frames <N>         Number of frames to process
 *       --scale <S>          Upscaling factor applied by CUDA kernels
 *       --threads <T>        Number of CUDA threads per block
 *       --blocks <B>         Number of CUDA blocks (0 = automatic)
 *       --print-every <N>    Interval for printing performance statistics
 *       --csv <file>         Output CSV file for logging performance metrics
 *       --help / -h          Display usage information
 *
 * @warning Thread counts are clamped between MIN_THREADS and MAX_THREADS to
 *          prevent invalid CUDA kernel launch configurations.
 */
static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (k == "--core") a.core_so = need("--core");
        else if (k == "--fr" || k == "--rom1") a.fr = need(k.c_str());
        else if (k == "--lg" || k == "--rom2") a.lg = need(k.c_str());
        else if (k == "--csv") a.csv = need("--csv");
        else if (k == "--frames") a.frames = std::stoi(need("--frames"));
        else if (k == "--scale") a.scale = std::stoi(need("--scale"));
        else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--blocks") a.blocks = std::stoi(need("--blocks"));
        else if (k == "--print-every") a.print_every = std::stoi(need("--print-every"));
        else if (k == "--help" || k == "-h") {
            std::cout
                << "Usage:\n  " << argv[0]
                << " [--core <mgba_libretro.so>] [--fr <FireRed.gba>] [--lg <LeafGreen.gba>]\n"
                << "       [--scale S] [--frames N] [--threads T] [--blocks B(0=auto)]\n"
                << "       [--print-every N] [--csv output.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown arg: " + k);
        }
    }
    if (a.scale < 1) a.scale = 1;
    if (a.threads < MIN_THREADS) a.threads = MIN_THREADS;
    if (a.threads > MAX_THREADS) a.threads = MAX_THREADS;
    if (a.print_every < 1) a.print_every = DEFAULT_PRINT_EVERY;
    return a;
}

// ---------------- RGB565 device utils --------------------
__device__ __forceinline__ void unpack565(uint16_t p, float& r, float& g, float& b) {
    int R = (p >> 11) & 31;
    int G = (p >> 5)  & 63;
    int B = (p >> 0)  & 31;
    r = (float)R / 31.0f;
    g = (float)G / 63.0f;
    b = (float)B / 31.0f;
}

__device__ __forceinline__ uint16_t pack565(float r, float g, float b) {
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    int R = (int)lrintf(r * 31.0f);
    int G = (int)lrintf(g * 63.0f);
    int B = (int)lrintf(b * 31.0f);
    return (uint16_t)((R << 11) | (G << 5) | B);
}

__device__ __forceinline__ uint8_t luma8_from565(uint16_t p) {
    float r, g, b;
    unpack565(p, r, g, b);
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    int yi = (int)lrintf(y * 255.0f);
    yi = yi < 0 ? 0 : yi;
    yi = yi > 255 ? 255 : yi;
    return (uint8_t)yi;
}

// ---------------- CUDA kernels --------------------

/**
 * @brief CUDA kernel that performs bilinear upscaling of an RGB565 image.
 *
 * Each GPU thread computes one pixel of the output image by sampling four
 * neighboring pixels from the input image and interpolating their color values
 * using bilinear interpolation. The interpolation is performed in floating
 * point space after unpacking the RGB565 color components.
 *
 * The kernel maps output pixel coordinates to corresponding fractional
 * coordinates in the source image. The four nearest source pixels are then
 * fetched, and horizontal and vertical interpolation weights are applied
 * to compute the final color.
 *
 * The interpolated RGB values are finally packed back into RGB565 format and
 * written to the destination image buffer.
 *
 * @param src Pointer to the source image in RGB565 format.
 * @param iw  Width of the source image.
 * @param ih  Height of the source image.
 * @param dst Pointer to the destination (upscaled) image buffer in RGB565 format.
 * @param ow  Width of the output image.
 * @param oh  Height of the output image.
 *
 * @note Each thread processes exactly one output pixel using a 2D thread grid.
 *       Threads that fall outside the output bounds exit immediately.
 *
 * @note RGB565 pixels are unpacked into floating-point RGB components for
 *       interpolation and then repacked into RGB565 before storage.
 *
 * @warning Boundary coordinates are clamped to valid source image ranges to
 *          prevent out-of-bounds memory accesses.
 */
__global__ void k_upscale_bilinear_565(const uint16_t* src, int iw, int ih,
                                       uint16_t* dst, int ow, int oh) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= ow || y >= oh) return;

    float gx = ((x + 0.5f) * (float)iw / (float)ow) - 0.5f;
    float gy = ((y + 0.5f) * (float)ih / (float)oh) - 0.5f;

    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = gx - (float)x0;
    float ty = gy - (float)y0;

    x0 = max(0, min(iw - 1, x0));
    x1 = max(0, min(iw - 1, x1));
    y0 = max(0, min(ih - 1, y0));
    y1 = max(0, min(ih - 1, y1));

    float r00,g00,b00; unpack565(src[y0 * iw + x0], r00,g00,b00);
    float r10,g10,b10; unpack565(src[y0 * iw + x1], r10,g10,b10);
    float r01,g01,b01; unpack565(src[y1 * iw + x0], r01,g01,b01);
    float r11,g11,b11; unpack565(src[y1 * iw + x1], r11,g11,b11);

    float r0 = r00 + tx * (r10 - r00);
    float g0 = g00 + tx * (g10 - g00);
    float b0 = b00 + tx * (b10 - b00);

    float r1 = r01 + tx * (r11 - r01);
    float g1 = g01 + tx * (g11 - g01);
    float b1 = b01 + tx * (b11 - b01);

    float r = r0 + ty * (r1 - r0);
    float g = g0 + ty * (g1 - g0);
    float b = b0 + ty * (b1 - b0);

    dst[y * ow + x] = pack565(r, g, b);
}

/**
 * @brief CUDA kernel that performs nearest-neighbor upscaling of an RGB565 image.
 *
 * Each thread computes a single output pixel by mapping its coordinates to the
 * corresponding location in the source image and copying the nearest pixel value.
 * Unlike bilinear interpolation, this method does not blend neighboring pixels
 * and therefore preserves the exact original pixel values.
 *
 * This kernel is primarily used as a control or reference scaling method to
 * compare against higher-quality interpolation methods such as bilinear scaling.
 *
 * @param src Pointer to the source image in RGB565 format.
 * @param iw  Width of the source image.
 * @param ih  Height of the source image.
 * @param dst Pointer to the destination (upscaled) image buffer in RGB565 format.
 * @param ow  Width of the output image.
 * @param oh  Height of the output image.
 *
 * @note Each thread computes exactly one output pixel using a 2D thread grid.
 *       Threads that fall outside the image bounds exit immediately.
 *
 * @note Source coordinates are computed using integer scaling, ensuring
 *       deterministic nearest-pixel selection.
 */
__global__ void k_upscale_nearest_565(const uint16_t* src, int iw, int ih,
                                      uint16_t* dst, int ow, int oh) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= ow || y >= oh) return;
    int sx = (int)((long long)x * iw / ow);
    int sy = (int)((long long)y * ih / oh);
    dst[y * ow + x] = src[sy * iw + sx];
}

__device__ __forceinline__ uint16_t heat565(uint8_t d) {
    float t = (float)d / 255.0f;
    float r = fminf(1.0f, 2.0f * t);
    float g = (t < 0.5f) ? (2.0f * t) : (2.0f * (1.0f - t));
    float b = fminf(1.0f, 2.0f * (1.0f - t));
    r = powf(r, 0.8f);
    g = powf(g, 0.8f);
    b = powf(b, 0.8f);
    return pack565(r, g, b);
}


/**
 * @brief CUDA kernel that computes a per-pixel difference heatmap between two RGB565 images.
 *
 * Each thread processes one pixel index from two input images of equal size.
 * The kernel converts both RGB565 pixels to luma values, computes the absolute
 * luminance difference, amplifies that difference using a heatmap gain factor,
 * and maps the result to a false-color RGB565 heatmap for visualization.
 *
 * This kernel is used for two comparison modes in the pipeline:
 * - self comparison: bilinear upscale vs nearest-neighbor upscale
 * - cross comparison: FireRed frame vs LeafGreen frame
 *
 * @param a   Pointer to the first input RGB565 image.
 * @param b   Pointer to the second input RGB565 image.
 * @param n   Number of pixels in each input image.
 * @param out Pointer to the output RGB565 heatmap image.
 *
 * @note Each thread computes exactly one output pixel using a 1D launch layout.
 *       Threads with index >= n exit immediately.
 *
 * @note The luminance difference is scaled by `HEAT_GAIN` and clamped to 255
 *       before conversion to a false-color heatmap.
 *
 * @warning This kernel assumes both input images contain at least `n` pixels.
 */
__global__ void k_heatmap_diff_native(const uint16_t* a, const uint16_t* b, int n, uint16_t* out) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    uint8_t ya = luma8_from565(a[i]);
    uint8_t yb = luma8_from565(b[i]);
    int dd = abs((int)ya - (int)yb);
    dd = min(255, dd * HEAT_GAIN);
    out[i] = heat565((uint8_t)dd);
}

/**
 * @brief Sums the luminance values of an RGB565 image on the GPU.
 *
 * Each thread converts one RGB565 pixel to an 8-bit luminance value and adds it
 * to a device-wide accumulator using an atomic operation.
 *
 * @param img Pointer to the input RGB565 image.
 * @param n   Number of pixels in the image.
 * @param sum Pointer to the device accumulator storing the luminance sum.
 */
__global__ void k_sum_luma565(const uint16_t* img, int n, unsigned long long* sum) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    atomicAdd(sum, (unsigned long long)luma8_from565(img[i]));
}


/**
 * @brief Counts pixels in an RGB565 image whose luminance exceeds a threshold.
 *
 * Each thread converts one RGB565 pixel to luminance and increments a device
 * counter if the luminance is greater than the specified threshold.
 *
 * @param img       Pointer to the input RGB565 image.
 * @param n         Number of pixels in the image.
 * @param threshold Luminance threshold for counting a pixel as nonzero/significant.
 * @param count     Pointer to the device counter.
 */
__global__ void k_count_nonzero_diff(const uint16_t* img, int n, uint8_t threshold, unsigned int* count) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    uint8_t y = luma8_from565(img[i]);
    if (y > threshold) atomicAdd(count, 1u);
}

/**
 * @brief CUDA kernel that composes six image panels into a single 2×3 grid output image.
 *
 * Each thread computes one pixel in the final composite image. The output frame
 * is organized as a grid containing six panels arranged in two columns and three
 * rows. The kernel determines which source panel the current output pixel belongs
 * to, computes the local coordinates within that panel, and copies the
 * corresponding RGB565 pixel value into the output buffer.
 *
 * Panel layout:
 *   Row 0: FireRed bilinear upscale | LeafGreen bilinear upscale
 *   Row 1: FireRed nearest upscale  | LeafGreen nearest upscale
 *   Row 2: Self-difference heatmap  | Cross-game difference heatmap
 *
 * @param p00 Pointer to the first panel (row 0, column 0).
 * @param p10 Pointer to the second panel (row 0, column 1).
 * @param p01 Pointer to the third panel (row 1, column 0).
 * @param p11 Pointer to the fourth panel (row 1, column 1).
 * @param p02 Pointer to the fifth panel (row 2, column 0).
 * @param p12 Pointer to the sixth panel (row 2, column 1).
 * @param ow  Width of each individual panel.
 * @param oh  Height of each individual panel.
 * @param out Pointer to the final composite RGB565 output buffer.
 *
 * @note The output resolution is (2 * ow) × (3 * oh).
 *       Each thread writes exactly one output pixel.
 *
 * @warning Threads outside the composite image bounds exit immediately
 *          to avoid invalid memory access.
 */
__global__ void k_compose_2x3(const uint16_t* p00, const uint16_t* p10,
                              const uint16_t* p01, const uint16_t* p11,
                              const uint16_t* p02, const uint16_t* p12,
                              int ow, int oh,
                              uint16_t* out) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int W = 2 * ow;
    int H = 3 * oh;
    if (x >= W || y >= H) return;

    int col = (x < ow) ? 0 : 1;
    int row = (y < oh) ? 0 : (y < 2 * oh ? 1 : 2);
    int lx = (col == 0) ? x : (x - ow);
    int ly = (row == 0) ? y : (row == 1 ? (y - oh) : (y - 2 * oh));

    const uint16_t* src = nullptr;
    if (row == 0 && col == 0) src = p00;
    if (row == 0 && col == 1) src = p10;
    if (row == 1 && col == 0) src = p01;
    if (row == 1 && col == 1) src = p11;
    if (row == 2 && col == 0) src = p02;
    if (row == 2 && col == 1) src = p12;

    out[y * W + x] = src[ly * ow + lx];
}

// ---------------- Main helpers --------------------

struct Geometry {
    int raw_w = GBA_WIDTH;
    int raw_h = GBA_HEIGHT;
    int scale = DEFAULT_SCALE;
    int up_w = 0;
    int up_h = 0;
    int win_w = 0;
    int win_h = 0;
};

struct FrameMetrics {
    float fr_up_ms = 0.0f;
    float lg_up_ms = 0.0f;
    float self_diff_ms = 0.0f;
    float cross_diff_ms = 0.0f;
    double total_gpu_ms = 0.0;
    double heat_self_mean = 0.0;
    double heat_cross_mean = 0.0;
    unsigned int nonzero_diff_pixels = 0;
};

struct SdlResources {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
};

struct CudaResources {
    uint16_t *fr_raw = nullptr, *lg_raw = nullptr;
    uint16_t *fr_up = nullptr, *lg_up = nullptr;
    uint16_t *fr_nearest = nullptr, *lg_nearest = nullptr;
    uint16_t *heat_cross_native = nullptr;
    uint16_t *heat_self = nullptr, *heat_cross = nullptr;
    uint16_t *composite = nullptr;

    unsigned long long *sum_self = nullptr, *sum_cross = nullptr;
    unsigned int *count_cross = nullptr;

    cudaStream_t fr_stream = nullptr;
    cudaStream_t lg_stream = nullptr;

    cudaEvent_t ev_fr_up_start = nullptr, ev_fr_up_end = nullptr;
    cudaEvent_t ev_lg_up_start = nullptr, ev_lg_up_end = nullptr;
    cudaEvent_t ev_self_start = nullptr, ev_self_end = nullptr;
    cudaEvent_t ev_cross_start = nullptr, ev_cross_end = nullptr;
    cudaEvent_t ev_frame_done = nullptr;
};

static Geometry make_geometry(int scale) {
    Geometry g;
    g.scale = scale;
    g.up_w = g.raw_w * g.scale;
    g.up_h = g.raw_h * g.scale;
    g.win_w = 2 * g.up_w;
    g.win_h = 3 * g.up_h;
    return g;
}

static void write_csv_header(std::ofstream& csv) {
    csv << "frame,threads,blocks,fr_up_ms,lg_up_ms,self_diff_ms,cross_diff_ms,"
           "total_gpu_ms,heat_self_mean,heat_cross_mean,nonzero_diff_pixels\n";
}

static void write_csv_row(std::ofstream& csv,
                          int frame_idx,
                          int threads,
                          int blocks,
                          const FrameMetrics& metrics) {
    csv << frame_idx << ","
        << threads << ","
        << blocks << ","
        << metrics.fr_up_ms << ","
        << metrics.lg_up_ms << ","
        << metrics.self_diff_ms << ","
        << metrics.cross_diff_ms << ","
        << metrics.total_gpu_ms << ","
        << metrics.heat_self_mean << ","
        << metrics.heat_cross_mean << ","
        << metrics.nonzero_diff_pixels << "\n";
}

static void print_metrics(int frame_idx,
                          int threads,
                          int blocks,
                          const FrameMetrics& metrics) {
    std::cout << "[frame " << frame_idx << "] "
              << "threads=" << threads
              << " blocks=" << blocks
              << " fr_up_ms=" << metrics.fr_up_ms
              << " lg_up_ms=" << metrics.lg_up_ms
              << " self_diff_ms=" << metrics.self_diff_ms
              << " cross_diff_ms=" << metrics.cross_diff_ms
              << " total_gpu_ms=" << metrics.total_gpu_ms
              << " heat_self_mean=" << metrics.heat_self_mean
              << " heat_cross_mean=" << metrics.heat_cross_mean
              << " nonzero_diff_pixels=" << metrics.nonzero_diff_pixels
              << "\n";
}

static void init_audio(const retro_system_av_info& av) {
    SDL_AudioSpec want{};
    want.freq = (int)av.timing.sample_rate;
    want.format = AUDIO_S16SYS;
    want.channels = 2;
    want.samples = AUDIO_BUFFER_SAMPLES;
    want.callback = sdl_audio_cb;

    SDL_AudioSpec have{};
    g_audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (!g_audio_dev) {
        std::cerr << "WARN: SDL_OpenAudioDevice failed: " << SDL_GetError() << "\n";
        return;
    }

    size_t ring_samples = (size_t)have.freq * (size_t)have.channels / 2;
    g_audio_ring.assign(ring_samples, 0);
    g_audio_r = g_audio_w = 0;
    SDL_PauseAudioDevice(g_audio_dev, 0);
}

static void shutdown_audio() {
    if (g_audio_dev) {
        SDL_CloseAudioDevice(g_audio_dev);
        g_audio_dev = 0;
    }
}

static void init_sdl_resources(const Geometry& geom, SdlResources& sdl) {
    sdl.window = SDL_CreateWindow(
        "StreamsHW CUDA (2x3 panels, RGB565)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        geom.win_w, geom.win_h,
        SDL_WINDOW_SHOWN
    );
    if (!sdl.window) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    sdl.renderer = SDL_CreateRenderer(sdl.window, -1, SDL_RENDERER_ACCELERATED);
    if (!sdl.renderer) {
        throw std::runtime_error(std::string("SDL_CreateRenderer failed: ") + SDL_GetError());
    }

    sdl.texture = SDL_CreateTexture(
        sdl.renderer,
        SDL_PIXELFORMAT_RGB565,
        SDL_TEXTUREACCESS_STREAMING,
        geom.win_w,
        geom.win_h
    );
    if (!sdl.texture) {
        throw std::runtime_error(std::string("SDL_CreateTexture failed: ") + SDL_GetError());
    }
}

static void destroy_sdl_resources(SdlResources& sdl) {
    if (sdl.texture) SDL_DestroyTexture(sdl.texture);
    if (sdl.renderer) SDL_DestroyRenderer(sdl.renderer);
    if (sdl.window) SDL_DestroyWindow(sdl.window);
    sdl.texture = nullptr;
    sdl.renderer = nullptr;
    sdl.window = nullptr;
}


/**
 * @brief Allocates GPU memory, CUDA streams, and CUDA events required for frame processing.
 *
 * This function initializes all CUDA resources needed to process emulator frames
 * on the GPU. Memory buffers are allocated for raw emulator output, upscaled
 * frames, intermediate processing results (e.g., nearest-neighbor scaling and
 * heatmaps), and the final composited output frame.
 *
 * Device-side accumulators used for statistical analysis are also allocated.
 * These track values such as self-similarity and cross-frame differences that
 * may be used for performance metrics or divergence analysis between emulator
 * instances.
 *
 * In addition to memory allocation, two CUDA streams are created to allow
 * concurrent GPU execution for separate emulator instances (e.g., FireRed
 * and LeafGreen). CUDA events are also initialized to measure kernel timing
 * and coordinate execution between stages of the GPU pipeline.
 *
 * @param gpu  Structure that stores all CUDA device pointers, streams, and events.
 * @param geom Structure describing the geometry of the frame buffers, including
 *             raw frame dimensions, upscaled frame dimensions, and window size.
 *
 * @note Memory buffers are sized according to three pixel resolutions:
 *       - raw_pixels: native emulator resolution
 *       - up_pixels:  resolution after GPU upscaling
 *       - win_pixels: resolution of the final composited display window
 *
 * @warning All allocations use CUDA_CHECK to ensure runtime errors are caught
 *          immediately if device memory allocation fails.
 */
static void allocate_cuda_resources(CudaResources& gpu, const Geometry& geom) {
    const size_t raw_pixels = (size_t)geom.raw_w * (size_t)geom.raw_h;
    const size_t up_pixels  = (size_t)geom.up_w * (size_t)geom.up_h;
    const size_t win_pixels = (size_t)geom.win_w * (size_t)geom.win_h;

    CUDA_CHECK(cudaMalloc(&gpu.fr_raw, raw_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.lg_raw, raw_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.fr_up, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.lg_up, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.fr_nearest, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.lg_nearest, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.heat_cross_native, raw_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.heat_self, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.heat_cross, up_pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&gpu.composite, win_pixels * sizeof(uint16_t)));

    CUDA_CHECK(cudaMalloc(&gpu.sum_self, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&gpu.sum_cross, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&gpu.count_cross, sizeof(unsigned int)));

    CUDA_CHECK(cudaStreamCreate(&gpu.fr_stream));
    CUDA_CHECK(cudaStreamCreate(&gpu.lg_stream));

    CUDA_CHECK(cudaEventCreate(&gpu.ev_fr_up_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_fr_up_end));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_lg_up_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_lg_up_end));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_self_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_self_end));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_cross_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_cross_end));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_frame_done));
}

static void destroy_cuda_resources(CudaResources& gpu) {
    if (gpu.ev_fr_up_start) CUDA_CHECK(cudaEventDestroy(gpu.ev_fr_up_start));
    if (gpu.ev_fr_up_end)   CUDA_CHECK(cudaEventDestroy(gpu.ev_fr_up_end));
    if (gpu.ev_lg_up_start) CUDA_CHECK(cudaEventDestroy(gpu.ev_lg_up_start));
    if (gpu.ev_lg_up_end)   CUDA_CHECK(cudaEventDestroy(gpu.ev_lg_up_end));
    if (gpu.ev_self_start)  CUDA_CHECK(cudaEventDestroy(gpu.ev_self_start));
    if (gpu.ev_self_end)    CUDA_CHECK(cudaEventDestroy(gpu.ev_self_end));
    if (gpu.ev_cross_start) CUDA_CHECK(cudaEventDestroy(gpu.ev_cross_start));
    if (gpu.ev_cross_end)   CUDA_CHECK(cudaEventDestroy(gpu.ev_cross_end));
    if (gpu.ev_frame_done)  CUDA_CHECK(cudaEventDestroy(gpu.ev_frame_done));

    if (gpu.fr_stream) CUDA_CHECK(cudaStreamDestroy(gpu.fr_stream));
    if (gpu.lg_stream) CUDA_CHECK(cudaStreamDestroy(gpu.lg_stream));

    if (gpu.fr_raw) CUDA_CHECK(cudaFree(gpu.fr_raw));
    if (gpu.lg_raw) CUDA_CHECK(cudaFree(gpu.lg_raw));
    if (gpu.fr_up) CUDA_CHECK(cudaFree(gpu.fr_up));
    if (gpu.lg_up) CUDA_CHECK(cudaFree(gpu.lg_up));
    if (gpu.fr_nearest) CUDA_CHECK(cudaFree(gpu.fr_nearest));
    if (gpu.lg_nearest) CUDA_CHECK(cudaFree(gpu.lg_nearest));
    if (gpu.heat_cross_native) CUDA_CHECK(cudaFree(gpu.heat_cross_native));
    if (gpu.heat_self) CUDA_CHECK(cudaFree(gpu.heat_self));
    if (gpu.heat_cross) CUDA_CHECK(cudaFree(gpu.heat_cross));
    if (gpu.composite) CUDA_CHECK(cudaFree(gpu.composite));

    if (gpu.sum_self) CUDA_CHECK(cudaFree(gpu.sum_self));
    if (gpu.sum_cross) CUDA_CHECK(cudaFree(gpu.sum_cross));
    if (gpu.count_cross) CUDA_CHECK(cudaFree(gpu.count_cross));

    gpu = CudaResources{};
}

static bool instances_have_valid_frames(const Instance& fire_red,
                                        const Instance& leaf_green,
                                        const Geometry& geom) {
    const size_t raw_pixels = (size_t)geom.raw_w * (size_t)geom.raw_h;
    return fire_red.w == (unsigned)geom.raw_w &&
           fire_red.h == (unsigned)geom.raw_h &&
           fire_red.frame565.size() >= raw_pixels &&
           leaf_green.w == (unsigned)geom.raw_w &&
           leaf_green.h == (unsigned)geom.raw_h &&
           leaf_green.frame565.size() >= raw_pixels;
}

static void render_composite_texture(SDL_Texture* texture,
                                     SDL_Renderer* renderer,
                                     const std::vector<uint16_t>& host_composite_rgb565,
                                     const Geometry& geom) {
    void* pixels = nullptr;
    int pitch_bytes = 0;
    if (SDL_LockTexture(texture, nullptr, &pixels, &pitch_bytes) == 0) {
        for (int y = 0; y < geom.win_h; y++) {
            std::memcpy(
                (uint8_t*)pixels + (size_t)y * (size_t)pitch_bytes,
                &host_composite_rgb565[(size_t)y * (size_t)geom.win_w],
                (size_t)geom.win_w * sizeof(uint16_t)
            );
        }
        SDL_UnlockTexture(texture);
    }

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
}


/**
 * @brief Collects timing and statistical metrics from GPU processing for a single frame.
 *
 * This function gathers performance measurements and computed statistics from the
 * GPU after a frame has completed processing. CUDA events are used to measure the
 * execution time of different kernel stages, including upscaling for each emulator
 * instance and difference computations between frames.
 *
 * In addition to timing metrics, device-side accumulators are copied back to the
 * host to compute summary statistics such as average self-difference, average
 * cross-difference between emulator frames, and the number of pixels with
 * non-zero differences.
 *
 * These metrics are used for runtime monitoring, performance analysis, and
 * potential logging to CSV for experimental evaluation.
 *
 * @param gpu  Structure containing CUDA device pointers, streams, and timing events.
 * @param geom Geometry structure describing the dimensions of the processed frames.
 *
 * @return FrameMetrics Structure containing kernel execution times, heatmap
 *         statistics, total GPU time for the frame, and divergence metrics.
 *
 * @note CUDA events measure elapsed time in milliseconds between recorded start
 *       and end points for each kernel stage.
 *
 * @warning Device accumulators must be copied back to host memory using
 *          cudaMemcpy before computing aggregate statistics.
 */
static FrameMetrics collect_frame_metrics(const CudaResources& gpu,
                                          const Geometry& geom) {
    FrameMetrics metrics{};

    CUDA_CHECK(cudaEventElapsedTime(&metrics.fr_up_ms, gpu.ev_fr_up_start, gpu.ev_fr_up_end));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.lg_up_ms, gpu.ev_lg_up_start, gpu.ev_lg_up_end));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.self_diff_ms, gpu.ev_self_start, gpu.ev_self_end));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.cross_diff_ms, gpu.ev_cross_start, gpu.ev_cross_end));

    unsigned long long h_sum_self = 0;
    unsigned long long h_sum_cross = 0;
    unsigned int h_count_cross = 0;

    CUDA_CHECK(cudaMemcpy(&h_sum_self, gpu.sum_self, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_cross, gpu.sum_cross, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count_cross, gpu.count_cross, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    const double up_pixels = (double)geom.up_w * (double)geom.up_h;
    metrics.heat_self_mean = (double)h_sum_self / up_pixels;
    metrics.heat_cross_mean = (double)h_sum_cross / up_pixels;
    metrics.total_gpu_ms = std::max(
        std::max((double)metrics.fr_up_ms, (double)metrics.lg_up_ms),
        std::max((double)metrics.self_diff_ms, (double)metrics.cross_diff_ms)
    );
    metrics.nonzero_diff_pixels = h_count_cross;

    return metrics;
}


/**
 * @brief Executes the full GPU processing pipeline for one synchronized emulator frame pair.
 *
 * This function transfers the current FireRed and LeafGreen emulator frames to
 * device memory, launches CUDA kernels to upscale and compare them, computes
 * frame-level difference metrics, composes a multi-panel visualization, and
 * copies the final composite image back to host memory for display.
 *
 * The pipeline uses two CUDA streams so that the FireRed and LeafGreen frame
 * upscaling stages can run concurrently. Subsequent analysis and composition
 * stages are coordinated with CUDA events to preserve dependencies while still
 * exposing overlap where possible.
 *
 * Processing stages include:
 * - asynchronous host-to-device copies of both raw RGB565 emulator frames
 * - bilinear upscaling of both games on separate CUDA streams
 * - nearest-neighbor upscaling for visual comparison
 * - heatmap generation for bilinear-vs-nearest self-difference
 * - heatmap generation for FireRed-vs-LeafGreen cross-difference
 * - reduction kernels for mean-difference and changed-pixel statistics
 * - composition of six output panels into a single display image
 * - asynchronous device-to-host copy of the final composite frame
 *
 * @param fire_red  Emulator instance containing the current FireRed frame.
 * @param leaf_green Emulator instance containing the current LeafGreen frame.
 * @param geom      Geometry information for raw, upscaled, and output frame sizes.
 * @param args      Runtime parameters controlling thread/block configuration.
 * @param gpu       CUDA buffers, streams, and events used by the processing pipeline.
 * @param host_composite_rgb565 Host-side output buffer that receives the final
 *                              composited RGB565 frame.
 *
 * @note The function uses:
 *       - 2D launch geometry for image-space kernels
 *       - 1D launch geometry for reduction and per-pixel metric kernels
 *       - CUDA events for timing and stream coordination
 *
 * @warning This function assumes all device memory, CUDA streams, and CUDA
 *          events in `gpu` have already been allocated and initialized.
 */
static void process_gpu_frame(const Instance& fire_red,
                              const Instance& leaf_green,
                              const Geometry& geom,
                              const Args& args,
                              CudaResources& gpu,
                              std::vector<uint16_t>& host_composite_rgb565) {
    auto grid2 = [](int w, int h, dim3 block) {
        return dim3(
            (w + (int)block.x - 1) / (int)block.x,
            (h + (int)block.y - 1) / (int)block.y,
            1
        );
    };

    const dim3 block2(BLOCK2D_X, BLOCK2D_Y, 1);
    const int threads1 = args.threads;
    const int blocks1_native = args.blocks > 0 ? args.blocks : (geom.raw_w * geom.raw_h + threads1 - 1) / threads1;
    const int blocks1_scaled = args.blocks > 0 ? args.blocks : (geom.up_w * geom.up_h + threads1 - 1) / threads1;

    // Copy raw emulator frames to GPU.
    CUDA_CHECK(cudaMemcpyAsync(
        gpu.fr_raw, fire_red.frame565.data(),
        (size_t)geom.raw_w * (size_t)geom.raw_h * sizeof(uint16_t),
        cudaMemcpyHostToDevice, gpu.fr_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        gpu.lg_raw, leaf_green.frame565.data(),
        (size_t)geom.raw_w * (size_t)geom.raw_h * sizeof(uint16_t),
        cudaMemcpyHostToDevice, gpu.lg_stream
    ));

    // Timed region 1: FireRed bilinear upscale on its own CUDA stream.
    CUDA_CHECK(cudaEventRecord(gpu.ev_fr_up_start, gpu.fr_stream));
    k_upscale_bilinear_565<<<grid2(geom.up_w, geom.up_h, block2), block2, 0, gpu.fr_stream>>>(
        gpu.fr_raw, geom.raw_w, geom.raw_h, gpu.fr_up, geom.up_w, geom.up_h
    );
    CUDA_CHECK(cudaEventRecord(gpu.ev_fr_up_end, gpu.fr_stream));

    // Timed region 2: LeafGreen bilinear upscale on its own CUDA stream.
    CUDA_CHECK(cudaEventRecord(gpu.ev_lg_up_start, gpu.lg_stream));
    k_upscale_bilinear_565<<<grid2(geom.up_w, geom.up_h, block2), block2, 0, gpu.lg_stream>>>(
        gpu.lg_raw, geom.raw_w, geom.raw_h, gpu.lg_up, geom.up_w, geom.up_h
    );
    CUDA_CHECK(cudaEventRecord(gpu.ev_lg_up_end, gpu.lg_stream));

    // Control row: nearest-neighbor upscales for comparison.
    k_upscale_nearest_565<<<grid2(geom.up_w, geom.up_h, block2), block2, 0, gpu.fr_stream>>>(
        gpu.fr_raw, geom.raw_w, geom.raw_h, gpu.fr_nearest, geom.up_w, geom.up_h
    );
    k_upscale_nearest_565<<<grid2(geom.up_w, geom.up_h, block2), block2, 0, gpu.lg_stream>>>(
        gpu.lg_raw, geom.raw_w, geom.raw_h, gpu.lg_nearest, geom.up_w, geom.up_h
    );

    // Timed region 3: FireRed bilinear vs nearest heatmap in display space.
    CUDA_CHECK(cudaEventRecord(gpu.ev_self_start, gpu.fr_stream));
    k_heatmap_diff_native<<<blocks1_scaled, threads1, 0, gpu.fr_stream>>>(
        gpu.fr_up, gpu.fr_nearest, geom.up_w * geom.up_h, gpu.heat_self
    );
    CUDA_CHECK(cudaEventRecord(gpu.ev_self_end, gpu.fr_stream));

    // Timed region 4: FireRed vs LeafGreen source difference heatmap.
    CUDA_CHECK(cudaStreamWaitEvent(gpu.fr_stream, gpu.ev_lg_up_end, 0));
    CUDA_CHECK(cudaEventRecord(gpu.ev_cross_start, gpu.fr_stream));
    k_heatmap_diff_native<<<blocks1_native, threads1, 0, gpu.fr_stream>>>(
        gpu.fr_raw, gpu.lg_raw, geom.raw_w * geom.raw_h, gpu.heat_cross_native
    );
    k_upscale_nearest_565<<<grid2(geom.up_w, geom.up_h, block2), block2, 0, gpu.fr_stream>>>(
        gpu.heat_cross_native, geom.raw_w, geom.raw_h, gpu.heat_cross, geom.up_w, geom.up_h
    );
    CUDA_CHECK(cudaEventRecord(gpu.ev_cross_end, gpu.fr_stream));

    // Metric reductions.
    CUDA_CHECK(cudaMemsetAsync(gpu.sum_self, 0, sizeof(unsigned long long), gpu.fr_stream));
    CUDA_CHECK(cudaMemsetAsync(gpu.sum_cross, 0, sizeof(unsigned long long), gpu.fr_stream));
    CUDA_CHECK(cudaMemsetAsync(gpu.count_cross, 0, sizeof(unsigned int), gpu.fr_stream));

    k_sum_luma565<<<blocks1_scaled, threads1, 0, gpu.fr_stream>>>(
        gpu.heat_self, geom.up_w * geom.up_h, gpu.sum_self
    );
    k_sum_luma565<<<blocks1_scaled, threads1, 0, gpu.fr_stream>>>(
        gpu.heat_cross, geom.up_w * geom.up_h, gpu.sum_cross
    );
    k_count_nonzero_diff<<<blocks1_scaled, threads1, 0, gpu.fr_stream>>>(
        gpu.heat_cross, geom.up_w * geom.up_h, DIFF_THRESHOLD, gpu.count_cross
    );

    // Compose the six panels into one RGB565 output image.
    k_compose_2x3<<<grid2(geom.win_w, geom.win_h, block2), block2, 0, gpu.fr_stream>>>(
        gpu.fr_up, gpu.lg_up,
        gpu.fr_nearest, gpu.lg_nearest,
        gpu.heat_self, gpu.heat_cross,
        geom.up_w, geom.up_h,
        gpu.composite
    );

    CUDA_CHECK(cudaMemcpyAsync(
        host_composite_rgb565.data(), gpu.composite,
        (size_t)geom.win_w * (size_t)geom.win_h * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, gpu.fr_stream
    ));

    CUDA_CHECK(cudaEventRecord(gpu.ev_frame_done, gpu.fr_stream));
    CUDA_CHECK(cudaEventSynchronize(gpu.ev_frame_done));
}

// ---------------- Main --------------------
int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        Instance fire_red = make_instance(args.core_so, args.fr);
        Instance leaf_green = make_instance(args.core_so, args.lg);
        g_audio_source = &fire_red;

        retro_system_av_info av{};
        tls_current = &fire_red;
        fire_red.api.retro_get_system_av_info(&av);
        tls_current = nullptr;

        const double target_fps = (av.timing.fps > 1.0) ? av.timing.fps : 59.7275;
        const double frame_ms = 1000.0 / target_fps;
        uint64_t next_tick = SDL_GetTicks64();

        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_AUDIO) != 0) {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }

        init_audio(av);

        Geometry geom = make_geometry(args.scale);

        SdlResources sdl{};
        init_sdl_resources(geom, sdl);

        std::vector<uint16_t> host_composite_rgb565(
            (size_t)geom.win_w * (size_t)geom.win_h, 0
        );

        std::ofstream csv(args.csv);
        if (!csv) {
            throw std::runtime_error("Failed to open CSV output file: " + args.csv);
        }
        write_csv_header(csv);

        CudaResources gpu{};
        allocate_cuda_resources(gpu, geom);

        const int threads_used = args.threads;
        const int blocks_used = args.blocks > 0
            ? args.blocks
            : (geom.raw_w * geom.raw_h + threads_used - 1) / threads_used;

        bool running = true;
        int frame_idx = 0;

        while (running) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) running = false;
                if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            }

            g_keys = build_synced_keymask_from_sdl();

            tls_current = &fire_red;
            fire_red.api.retro_run();
            tls_current = &leaf_green;
            leaf_green.api.retro_run();
            tls_current = nullptr;

            if (!instances_have_valid_frames(fire_red, leaf_green, geom)) {
                continue;
            }

            process_gpu_frame(fire_red, leaf_green, geom, args, gpu, host_composite_rgb565);

            FrameMetrics metrics = collect_frame_metrics(gpu, geom);
            write_csv_row(csv, frame_idx, threads_used, blocks_used, metrics);

            if (args.print_every > 0 && (frame_idx % args.print_every) == 0) {
                print_metrics(frame_idx, threads_used, blocks_used, metrics);
            }

            render_composite_texture(sdl.texture, sdl.renderer, host_composite_rgb565, geom);

            frame_idx++;
            if (args.frames >= 0 && frame_idx >= args.frames) {
                running = false;
            }

            next_tick += (uint64_t)(frame_ms + 0.5);
            uint64_t now = SDL_GetTicks64();
            if (next_tick > now) {
                SDL_Delay((Uint32)(next_tick - now));
            } else {
                next_tick = now;
            }
        }

        csv.close();
        destroy_cuda_resources(gpu);
        destroy_sdl_resources(sdl);
        shutdown_audio();
        SDL_Quit();

        destroy_instance(leaf_green);
        destroy_instance(fire_red);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}