#include <SDL2/SDL.h>
#include <dlfcn.h>
#include <link.h>   // dlmopen, LM_ID_NEWLM

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "libretro.h"

// -------------------- Controls --------------------
// WASD = D-pad, 4=A, 5=B, 2=Start, 3=Select, Q=L, R=R
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

    // D-pad (WASD stays the same)
    if (st[SDL_SCANCODE_W]) keys |= KEY_UP;
    if (st[SDL_SCANCODE_S]) keys |= KEY_DOWN;
    if (st[SDL_SCANCODE_A]) keys |= KEY_LEFT;
    if (st[SDL_SCANCODE_D]) keys |= KEY_RIGHT;

    // Numpad controls
    if (st[SDL_SCANCODE_KP_4]) keys |= KEY_A;       // A
    if (st[SDL_SCANCODE_KP_5]) keys |= KEY_B;       // B

    if (st[SDL_SCANCODE_KP_2]) keys |= KEY_START;   // Start
    if (st[SDL_SCANCODE_KP_3]) keys |= KEY_SELECT;  // Select

    if (st[SDL_SCANCODE_KP_7]) keys |= KEY_L;       // L
    if (st[SDL_SCANCODE_KP_9]) keys |= KEY_R;       // R

    return keys;
}

// -------------------- Libretro API table --------------------
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

// -------------------- Instance + TLS current --------------------
struct Instance {
    void* so = nullptr;
    RetroAPI api{};
    std::vector<uint16_t> frame565;
    unsigned w = 0, h = 0;
    size_t pitch = 0;
    std::string rom_path;
};

static thread_local Instance* tls_current = nullptr;
static Instance* g_audio_source = nullptr; // which instance drives audio

// -------------------- Video callback --------------------
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

// -------------------- Audio (SDL ring buffer) --------------------
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

static void audio_cb(void* /*userdata*/, Uint8* stream, int len_bytes) {
    std::memset(stream, 0, (size_t)len_bytes);
    if (g_audio_ring.empty()) return;

    int16_t* out = (int16_t*)stream;
    size_t want = (size_t)len_bytes / sizeof(int16_t);
    size_t cap = g_audio_ring.size();

    for (size_t i = 0; i < want; i++) {
        if (g_audio_r == g_audio_w) break; // underrun => keep zeros
        out[i] = g_audio_ring[g_audio_r];
        g_audio_r = (g_audio_r + 1) % cap;
    }
}

// Libretro audio callbacks
static void audio_sample_cb(int16_t, int16_t) {}

static size_t audio_batch_cb(const int16_t* data, size_t frames) {
    // data is interleaved stereo; frames = number of stereo frames
    if (!data) return frames;
    if (tls_current != g_audio_source) return frames; // only one instance is audible
    audio_ring_write(data, frames * 2); // stereo => 2 samples per frame
    return frames;
}

static void input_poll_cb() {}

// -------------------- Input mapping --------------------
static int16_t input_state_cb(unsigned /*port*/, unsigned device, unsigned /*index*/, unsigned id) {
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

// Force RGB565 (fixed your cursed pixels)
static bool environment_cb(unsigned cmd, void* data) {
    if (cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT) {
        auto* fmt = (retro_pixel_format*)data;
        *fmt = RETRO_PIXEL_FORMAT_RGB565;
        return true;
    }
    return false;
}

// -------------------- dl helpers --------------------
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

// -------------------- instance lifecycle --------------------
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

// -------------------- Args --------------------
struct Args {
    std::string core_so;
    std::string rom1;
    std::string rom2;
    int frames = -1;
    int scale = 3;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    a.core_so = "./libretro-super/dist/unix/mgba_libretro.so";
    a.rom1 = "./GBAROMS/FireRed.gba";
    a.rom2 = "./GBAROMS/LeafGreen.gba";

    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (k == "--core") a.core_so = need("--core");
        else if (k == "--rom1" || k == "--fr" || k == "--rom") a.rom1 = need(k.c_str());
        else if (k == "--rom2" || k == "--lg") a.rom2 = need(k.c_str());
        else if (k == "--frames") a.frames = std::stoi(need("--frames"));
        else if (k == "--scale") a.scale = std::stoi(need("--scale"));
        else if (k == "--help" || k == "-h") {
            std::cout <<
                "Usage:\n  " << argv[0] <<
                " [--core <mgba_libretro.so>] [--fr <FireRed.gba>] [--lg <LeafGreen.gba>]"
                " [--frames N] [--scale S]\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown arg: " + k);
    }

    if (a.scale < 1) a.scale = 1;
    return a;
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        Instance A = make_instance(args.core_so, args.rom1); // FireRed
        Instance B = make_instance(args.core_so, args.rom2); // LeafGreen

        // Audio comes from instance A for MVP
        g_audio_source = &A;

        // Query sample rate from the core
        retro_system_av_info av{};
        tls_current = &A;
        A.api.retro_get_system_av_info(&av);
        tls_current = nullptr;

        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_AUDIO) != 0)
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

        // Open SDL audio
        SDL_AudioSpec want{};
        want.freq = (int)av.timing.sample_rate;   // typically 48000
        want.format = AUDIO_S16SYS;
        want.channels = 2;
        want.samples = 1024;
        want.callback = audio_cb;
        const double target_fps = av.timing.fps > 1.0 ? av.timing.fps : 59.7275;
        const double frame_ms = 1000.0 / target_fps;
        uint64_t next_tick = SDL_GetTicks64();

        SDL_AudioSpec have{};
        g_audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (!g_audio_dev) {
            std::cerr << "WARN: SDL_OpenAudioDevice failed: " << SDL_GetError() << "\n";
        } else {
            // ~0.5 seconds ring buffer
            size_t ring_samples = (size_t)have.freq * (size_t)have.channels / 2;
            g_audio_ring.assign(ring_samples, 0);
            g_audio_r = g_audio_w = 0;
            SDL_PauseAudioDevice(g_audio_dev, 0);
        }

        const int tileW = 240, tileH = 160;
        const int winW = 2 * tileW, winH = tileH;

        SDL_Window* win = SDL_CreateWindow("mGBA libretro (2 instances)",
                                           SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                           winW * args.scale, winH * args.scale,
                                           SDL_WINDOW_SHOWN);
        if (!win) throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());

        SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
        if (!ren) throw std::runtime_error(std::string("SDL_CreateRenderer failed: ") + SDL_GetError());

        SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB565,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             winW, winH);
        if (!tex) throw std::runtime_error(std::string("SDL_CreateTexture failed: ") + SDL_GetError());

        std::vector<uint16_t> composite((size_t)winW * (size_t)winH, 0);

        bool running = true;
        int frame_idx = 0;

        while (running) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) running = false;
                if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            }

            g_keys = build_synced_keymask_from_sdl();

            tls_current = &A;
            A.api.retro_run();
            tls_current = &B;
            B.api.retro_run();
            tls_current = nullptr;

            auto blit = [&](const Instance& inst, int xoff) {
                if (inst.w != 240 || inst.h != 160) return;
                if (inst.frame565.size() < (size_t)240 * 160) return;

                for (int y = 0; y < 160; y++) {
                    uint16_t* dst = &composite[(size_t)y * (size_t)winW + (size_t)xoff];
                    const uint16_t* src = &inst.frame565[(size_t)y * 240];
                    std::memcpy(dst, src, 240 * sizeof(uint16_t));
                }
            };

            std::fill(composite.begin(), composite.end(), 0);
            blit(A, 0);
            blit(B, 240);

            void* pixels = nullptr;
            int pitchBytes = 0;
            if (SDL_LockTexture(tex, nullptr, &pixels, &pitchBytes) == 0) {
                for (int y = 0; y < winH; y++) {
                    std::memcpy((uint8_t*)pixels + (size_t)y * (size_t)pitchBytes,
                                &composite[(size_t)y * (size_t)winW],
                                (size_t)winW * sizeof(uint16_t));
                }
                SDL_UnlockTexture(tex);
            }

            SDL_RenderClear(ren);
            SDL_RenderCopy(ren, tex, nullptr, nullptr);
            SDL_RenderPresent(ren);

            frame_idx++;
            if (args.frames >= 0 && frame_idx >= args.frames) running = false;

            // Frame pacing
            next_tick += (uint64_t)(frame_ms + 0.5); // round to nearest ms
            uint64_t now = SDL_GetTicks64();
            if (next_tick > now) {
                SDL_Delay((Uint32)(next_tick - now));
            } else {
                // If we're behind, don't sleep; also avoid runaway drift
                next_tick = now;
            }
        }

        if (g_audio_dev) SDL_CloseAudioDevice(g_audio_dev);
        g_audio_dev = 0;

        SDL_DestroyTexture(tex);
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        SDL_Quit();

        destroy_instance(B);
        destroy_instance(A);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}