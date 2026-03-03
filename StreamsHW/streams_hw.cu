// streams_hw_cuda.cu
// 2×3 panel viewer (RGB565 end-to-end)
// Row 1 (GPU gamer upscale):   [FR up] [LG up]    (bilinear, CUDA streams)
// Row 2 (control / raw):       [FR raw] [LG raw]  (nearest upscale for "pixel perfect")
// Row 3 (data science stream): [FR self diff] [FR vs LG diff] (heatmaps)
//
// Controls:
//   WASD = D-pad
//   KP_4=A, KP_5=B
//   KP_2=Start, KP_3=Select
//   KP_7=L, KP_9=R
//   ESC quits
//
// Build (example):
// nvcc -O2 -std=c++17 streams_hw_cuda.cu -o streams_hw_cuda \
//   $(sdl2-config --cflags) -I./mgba/src/platform/libretro \
//   $(sdl2-config --libs) -ldl
//4
// Run:
// ./streams_hw_cuda --scale 3 --threads 256 --frames 600 --print-every 60

#include <SDL2/SDL.h>
#include <dlfcn.h>
#include <link.h>   // dlmopen, LM_ID_NEWLM

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

// ---------------- CUDA helpers ----------------
#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
  } \
} while(0)

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

    // D-pad (WASD)
    if (st[SDL_SCANCODE_W]) keys |= KEY_UP;
    if (st[SDL_SCANCODE_S]) keys |= KEY_DOWN;
    if (st[SDL_SCANCODE_A]) keys |= KEY_LEFT;
    if (st[SDL_SCANCODE_D]) keys |= KEY_RIGHT;

    // Numpad
    if (st[SDL_SCANCODE_KP_4]) keys |= KEY_A;
    if (st[SDL_SCANCODE_KP_5]) keys |= KEY_B;
    if (st[SDL_SCANCODE_KP_2]) keys |= KEY_START;
    if (st[SDL_SCANCODE_KP_3]) keys |= KEY_SELECT;
    if (st[SDL_SCANCODE_KP_7]) keys |= KEY_L;
    if (st[SDL_SCANCODE_KP_9]) keys |= KEY_R;

    return keys;
}

// ---------------- Libretro API table ----------------
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

// ---------------- Instance + TLS --------------------
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

// ---------------- Audio ring buffer (SDL) ----------------
static SDL_AudioDeviceID g_audio_dev = 0;
static std::vector<int16_t> g_audio_ring;
static size_t g_audio_r = 0, g_audio_w = 0;

static void audio_ring_write(const int16_t* src, size_t count) {
    if (g_audio_ring.empty() || !src || count == 0) return;
    size_t cap = g_audio_ring.size();
    for (size_t i = 0; i < count; i++) {
        g_audio_ring[g_audio_w] = src[i];
        g_audio_w = (g_audio_w + 1) % cap;
        if (g_audio_w == g_audio_r) g_audio_r = (g_audio_r + 1) % cap; // drop oldest
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
    if (tls_current != g_audio_source) return frames; // only one audible
    audio_ring_write(data, frames * 2); // stereo
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
    int frames = -1;
    int scale = 3;

    int threads = 256;
    int blocks = 0; // 0 => auto
    int print_every = 60;
};

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
        else if (k == "--frames") a.frames = std::stoi(need("--frames"));
        else if (k == "--scale") a.scale = std::stoi(need("--scale"));
        else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--blocks") a.blocks = std::stoi(need("--blocks"));
        else if (k == "--print-every") a.print_every = std::stoi(need("--print-every"));
        else if (k == "--help" || k == "-h") {
            std::cout <<
              "Usage:\n  " << argv[0]
              << " [--core <mgba_libretro.so>] [--fr <FireRed.gba>] [--lg <LeafGreen.gba>]\n"
              << "       [--scale S] [--frames N] [--threads T] [--blocks B(0=auto)] [--print-every N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown arg: " + k);
        }
    }
    if (a.scale < 1) a.scale = 1;
    if (a.threads < 64) a.threads = 64;
    if (a.threads > 1024) a.threads = 1024;
    if (a.print_every < 1) a.print_every = 60;
    return a;
}

// ---------------- RGB565 utils (device) ----------------
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
    return (uint16_t)((R << 11) | (G << 5) | (B));
}

__device__ __forceinline__ uint8_t luma8_from565(uint16_t p) {
    float r,g,b; unpack565(p,r,g,b);
    float y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    int yi = (int)lrintf(y * 255.0f);
    if (yi < 0) yi = 0;
    if (yi > 255) yi = 255;
    return (uint8_t)yi;
}

// Bilinear upscale RGB565: dst (ow×oh), src (iw×ih)
__global__ void k_upscale_bilinear_565(const uint16_t* src, int iw, int ih,
                                       uint16_t* dst, int ow, int oh) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= ow || y >= oh) return;

    // map dst pixel center -> src space
    float gx = ((x + 0.5f) * (float)iw / (float)ow) - 0.5f;
    float gy = ((y + 0.5f) * (float)ih / (float)oh) - 0.5f;

    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = gx - (float)x0;
    float ty = gy - (float)y0;

    x0 = max(0, min(iw-1, x0));
    x1 = max(0, min(iw-1, x1));
    y0 = max(0, min(ih-1, y0));
    y1 = max(0, min(ih-1, y1));

    float r00,g00,b00; unpack565(src[y0*iw + x0], r00,g00,b00);
    float r10,g10,b10; unpack565(src[y0*iw + x1], r10,g10,b10);
    float r01,g01,b01; unpack565(src[y1*iw + x0], r01,g01,b01);
    float r11,g11,b11; unpack565(src[y1*iw + x1], r11,g11,b11);

    float r0 = r00 + tx*(r10 - r00);
    float g0 = g00 + tx*(g10 - g00);
    float b0 = b00 + tx*(b10 - b00);

    float r1 = r01 + tx*(r11 - r01);
    float g1 = g01 + tx*(g11 - g01);
    float b1 = b01 + tx*(b11 - b01);

    float r = r0 + ty*(r1 - r0);
    float g = g0 + ty*(g1 - g0);
    float b = b0 + ty*(b1 - b0);

    dst[y*ow + x] = pack565(r,g,b);
}

// Nearest upscale "pixel perfect" control panel
__global__ void k_upscale_nearest_565(const uint16_t* src, int iw, int ih,
                                      uint16_t* dst, int ow, int oh) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= ow || y >= oh) return;
    int sx = (int)((long long)x * iw / ow);
    int sy = (int)((long long)y * ih / oh);
    dst[y*ow + x] = src[sy*iw + sx];
}

// Heatmap: abs(luma(a)-luma(b)) -> colored RGB565 (blue->red)
__device__ __forceinline__ uint16_t heat565(uint8_t d) {
    // d in [0,255]
    float t = (float)d / 255.0f;
    // simple gradient: blue -> cyan -> green -> yellow -> red
    float r = fminf(1.0f, 2.0f * t);
    float g = (t < 0.5f) ? (2.0f*t) : (2.0f*(1.0f - t));
    float b = fminf(1.0f, 2.0f*(1.0f - t));
    // boost contrast a bit
    r = powf(r, 0.8f);
    g = powf(g, 0.8f);
    b = powf(b, 0.8f);
    return pack565(r,g,b);
}

__global__ void k_heatmap_diff_native(const uint16_t* a, const uint16_t* b, int w, int h, uint16_t* out) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = w*h;
    if (i >= n) return;
    uint8_t ya = luma8_from565(a[i]);
    uint8_t yb = luma8_from565(b[i]);
    uint8_t d = (uint8_t)abs((int)ya - (int)yb);
    out[i] = heat565(d);
}

// Compose 6 panels (all are ow×oh RGB565) into composite (2*ow × 3*oh)
__global__ void k_compose_2x3(const uint16_t* p00, const uint16_t* p10,
                              const uint16_t* p01, const uint16_t* p11,
                              const uint16_t* p02, const uint16_t* p12,
                              int ow, int oh,
                              uint16_t* out) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int W = 2*ow;
    int H = 3*oh;
    if (x >= W || y >= H) return;

    int col = (x < ow) ? 0 : 1;
    int row = (y < oh) ? 0 : (y < 2*oh ? 1 : 2);
    int lx = (col == 0) ? x : (x - ow);
    int ly = (row == 0) ? y : (row == 1 ? (y - oh) : (y - 2*oh));

    const uint16_t* src = nullptr;
    if (row == 0 && col == 0) src = p00;
    if (row == 0 && col == 1) src = p10;
    if (row == 1 && col == 0) src = p01;
    if (row == 1 && col == 1) src = p11;
    if (row == 2 && col == 0) src = p02;
    if (row == 2 && col == 1) src = p12;

    out[y*W + x] = src[ly*ow + lx];
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        // Libretro instances
        Instance A = make_instance(args.core_so, args.fr);
        Instance B = make_instance(args.core_so, args.lg);

        // Audio from FireRed
        g_audio_source = &A;

        // Query FPS + sample rate
        retro_system_av_info av{};
        tls_current = &A;
        A.api.retro_get_system_av_info(&av);
        tls_current = nullptr;

        const double target_fps = (av.timing.fps > 1.0) ? av.timing.fps : 59.7275;
        const double frame_ms = 1000.0 / target_fps;
        uint64_t next_tick = SDL_GetTicks64();

        // SDL init
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_AUDIO) != 0)
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

        // Audio device
        SDL_AudioSpec want{};
        want.freq = (int)av.timing.sample_rate;
        want.format = AUDIO_S16SYS;
        want.channels = 2;
        want.samples = 1024;
        want.callback = sdl_audio_cb;

        SDL_AudioSpec have{};
        g_audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (!g_audio_dev) {
            std::cerr << "WARN: SDL_OpenAudioDevice failed: " << SDL_GetError() << "\n";
        } else {
            size_t ring_samples = (size_t)have.freq * (size_t)have.channels / 2; // ~0.5s
            g_audio_ring.assign(ring_samples, 0);
            g_audio_r = g_audio_w = 0;
            SDL_PauseAudioDevice(g_audio_dev, 0);
        }

        // Geometry
        const int W = 240, H = 160;
        const int S = args.scale;
        const int OW = W * S;
        const int OH = H * S;

        const int WIN_W = 2 * OW;
        const int WIN_H = 3 * OH;

        SDL_Window* win = SDL_CreateWindow("StreamsHW CUDA (2x3 panels, RGB565)",
                                           SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                           WIN_W, WIN_H,
                                           SDL_WINDOW_SHOWN);
        if (!win) throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());

        SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
        if (!ren) throw std::runtime_error(std::string("SDL_CreateRenderer failed: ") + SDL_GetError());

        SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB565,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             WIN_W, WIN_H);
        if (!tex) throw std::runtime_error(std::string("SDL_CreateTexture failed: ") + SDL_GetError());

        // Host composite
        std::vector<uint16_t> h_composite((size_t)WIN_W * (size_t)WIN_H, 0);

        // CUDA buffers
        uint16_t *d_fr_raw = nullptr, *d_lg_raw = nullptr;
        uint16_t *d_fr_up = nullptr, *d_lg_up = nullptr;
        uint16_t *d_fr_ctl = nullptr, *d_lg_ctl = nullptr;
        uint16_t *d_fr_down = nullptr; // W×H
        uint16_t *d_heat_self_native = nullptr; // W×H
        uint16_t *d_heat_cross_native = nullptr; // W×H
        uint16_t *d_heat_self = nullptr, *d_heat_cross = nullptr; // OW×OH
        uint16_t *d_composite = nullptr;

        CUDA_CHECK(cudaMalloc(&d_fr_raw, (size_t)W*H*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lg_raw, (size_t)W*H*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_fr_up,  (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lg_up,  (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_fr_ctl, (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_lg_ctl, (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_fr_down, (size_t)W*H*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_heat_self_native, (size_t)W*H*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_heat_cross_native, (size_t)W*H*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_heat_self, (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_heat_cross, (size_t)OW*OH*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_composite, (size_t)WIN_W*WIN_H*sizeof(uint16_t)));

        // Streams + events
        cudaStream_t sFR, sLG;
        CUDA_CHECK(cudaStreamCreate(&sFR));
        CUDA_CHECK(cudaStreamCreate(&sLG));

        cudaEvent_t evFR0, evFR1, evLG0, evLG1, evAll;
        CUDA_CHECK(cudaEventCreate(&evFR0));
        CUDA_CHECK(cudaEventCreate(&evFR1));
        CUDA_CHECK(cudaEventCreate(&evLG0));
        CUDA_CHECK(cudaEventCreate(&evLG1));
        CUDA_CHECK(cudaEventCreate(&evAll));

        auto grid2 = [&](int w, int h, dim3 block) {
            return dim3((w + (int)block.x - 1) / (int)block.x,
                        (h + (int)block.y - 1) / (int)block.y, 1);
        };

        // Kernel launch params (2D kernels)
        dim3 block2(16, 16, 1);

        // For 1D heatmap kernels
        int threads1 = args.threads;
        int blocks1_native = args.blocks > 0 ? args.blocks : (W*H + threads1 - 1) / threads1;

        bool running = true;
        int frame_idx = 0;

        while (running) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) running = false;
                if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            }

            g_keys = build_synced_keymask_from_sdl();

            // Run emulation one frame each
            tls_current = &A; A.api.retro_run();
            tls_current = &B; B.api.retro_run();
            tls_current = nullptr;

            // Validate frame buffers
            if (A.w != (unsigned)W || A.h != (unsigned)H || A.frame565.size() < (size_t)W*H) continue;
            if (B.w != (unsigned)W || B.h != (unsigned)H || B.frame565.size() < (size_t)W*H) continue;

            // ---- STREAM FR: copy + upscale + control + downscale + self heatmap ----
            CUDA_CHECK(cudaEventRecord(evFR0, sFR));

            CUDA_CHECK(cudaMemcpyAsync(d_fr_raw, A.frame565.data(), (size_t)W*H*sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, sFR));

            // Row1 TL: gamer upscale (bilinear)
            k_upscale_bilinear_565<<<grid2(OW, OH, block2), block2, 0, sFR>>>(d_fr_raw, W, H, d_fr_up, OW, OH);

            // Row2 ML: control raw (nearest upscale)
            k_upscale_nearest_565<<<grid2(OW, OH, block2), block2, 0, sFR>>>(d_fr_raw, W, H, d_fr_ctl, OW, OH);

            // Downscale (bilinear) back to native: d_fr_up (OW×OH) -> d_fr_down (W×H)
            k_upscale_bilinear_565<<<grid2(W, H, block2), block2, 0, sFR>>>(d_fr_up, OW, OH, d_fr_down, W, H);

            // Heat self native: FR raw vs FR down
            k_heatmap_diff_native<<<blocks1_native, threads1, 0, sFR>>>(d_fr_raw, d_fr_down, W, H, d_heat_self_native);

            // Upscale self heatmap native -> display size (nearest so it looks crisp)
            k_upscale_nearest_565<<<grid2(OW, OH, block2), block2, 0, sFR>>>(d_heat_self_native, W, H, d_heat_self, OW, OH);

            CUDA_CHECK(cudaEventRecord(evFR1, sFR));

            // ---- STREAM LG: copy + upscale + control ----
            CUDA_CHECK(cudaEventRecord(evLG0, sLG));

            CUDA_CHECK(cudaMemcpyAsync(d_lg_raw, B.frame565.data(), (size_t)W*H*sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, sLG));

            // Row1 TR: gamer upscale (bilinear)
            k_upscale_bilinear_565<<<grid2(OW, OH, block2), block2, 0, sLG>>>(d_lg_raw, W, H, d_lg_up, OW, OH);

            // Row2 MR: control raw (nearest)
            k_upscale_nearest_565<<<grid2(OW, OH, block2), block2, 0, sLG>>>(d_lg_raw, W, H, d_lg_ctl, OW, OH);

            CUDA_CHECK(cudaEventRecord(evLG1, sLG));

            // ---- Cross heatmap (FR raw vs LG raw) ----
            // Wait for both raw copies to complete
            CUDA_CHECK(cudaStreamWaitEvent(sFR, evLG1, 0));
            CUDA_CHECK(cudaStreamWaitEvent(sFR, evFR1, 0));

            // Heat cross native -> upscale to display
            k_heatmap_diff_native<<<blocks1_native, threads1, 0, sFR>>>(d_fr_raw, d_lg_raw, W, H, d_heat_cross_native);
            k_upscale_nearest_565<<<grid2(OW, OH, block2), block2, 0, sFR>>>(d_heat_cross_native, W, H, d_heat_cross, OW, OH);

            // ---- Compose 2×3 panels on GPU ----
            // p00=fr_up, p10=lg_up
            // p01=fr_ctl, p11=lg_ctl
            // p02=heat_self, p12=heat_cross
            k_compose_2x3<<<grid2(WIN_W, WIN_H, block2), block2, 0, sFR>>>(
                d_fr_up, d_lg_up,
                d_fr_ctl, d_lg_ctl,
                d_heat_self, d_heat_cross,
                OW, OH,
                d_composite
            );

            // Copy composite back
            CUDA_CHECK(cudaMemcpyAsync(h_composite.data(), d_composite,
                                       (size_t)WIN_W*WIN_H*sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, sFR));

            CUDA_CHECK(cudaEventRecord(evAll, sFR));
            CUDA_CHECK(cudaEventSynchronize(evAll));

            // ---- Timing print ----
            if (args.print_every > 0 && (frame_idx % args.print_every) == 0) {
                float msFR = 0.f, msLG = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&msFR, evFR0, evFR1));
                CUDA_CHECK(cudaEventElapsedTime(&msLG, evLG0, evLG1));
                std::cout << "[frame " << frame_idx << "] "
                          << "FR stream=" << msFR << " ms, "
                          << "LG stream=" << msLG << " ms, "
                          << "threads=" << threads1
                          << "\n";
            }

            // ---- SDL render ----
            void* pixels = nullptr;
            int pitchBytes = 0;
            if (SDL_LockTexture(tex, nullptr, &pixels, &pitchBytes) == 0) {
                for (int y = 0; y < WIN_H; y++) {
                    std::memcpy((uint8_t*)pixels + (size_t)y * (size_t)pitchBytes,
                                &h_composite[(size_t)y * (size_t)WIN_W],
                                (size_t)WIN_W * sizeof(uint16_t));
                }
                SDL_UnlockTexture(tex);
            }

            SDL_RenderClear(ren);
            SDL_RenderCopy(ren, tex, nullptr, nullptr);
            SDL_RenderPresent(ren);

            frame_idx++;
            if (args.frames >= 0 && frame_idx >= args.frames) running = false;

            // Frame pacing
            next_tick += (uint64_t)(frame_ms + 0.5);
            uint64_t now = SDL_GetTicks64();
            if (next_tick > now) SDL_Delay((Uint32)(next_tick - now));
            else next_tick = now;
        }

        // Cleanup
        if (g_audio_dev) SDL_CloseAudioDevice(g_audio_dev);

        SDL_DestroyTexture(tex);
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        SDL_Quit();

        CUDA_CHECK(cudaEventDestroy(evFR0));
        CUDA_CHECK(cudaEventDestroy(evFR1));
        CUDA_CHECK(cudaEventDestroy(evLG0));
        CUDA_CHECK(cudaEventDestroy(evLG1));
        CUDA_CHECK(cudaEventDestroy(evAll));
        CUDA_CHECK(cudaStreamDestroy(sFR));
        CUDA_CHECK(cudaStreamDestroy(sLG));

        CUDA_CHECK(cudaFree(d_fr_raw));
        CUDA_CHECK(cudaFree(d_lg_raw));
        CUDA_CHECK(cudaFree(d_fr_up));
        CUDA_CHECK(cudaFree(d_lg_up));
        CUDA_CHECK(cudaFree(d_fr_ctl));
        CUDA_CHECK(cudaFree(d_lg_ctl));
        CUDA_CHECK(cudaFree(d_fr_down));
        CUDA_CHECK(cudaFree(d_heat_self_native));
        CUDA_CHECK(cudaFree(d_heat_cross_native));
        CUDA_CHECK(cudaFree(d_heat_self));
        CUDA_CHECK(cudaFree(d_heat_cross));
        CUDA_CHECK(cudaFree(d_composite));

        destroy_instance(B);
        destroy_instance(A);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}