// =============================================================================
// simulation.cu — CUDA implementation for Rotea-inspired CFC simulation
//
// Provides:
//   1. Persistent CUDA particle buffer for live OpenGL CUDA mode
//   2. Standalone CUDA frame generator for replay mode
//   3. Replay binary format compatible with main.cpp reader:
//        magic[8] = "ROTEAFRM"
//        version
//        particleCount
//        frameCount
//        dt
//
// Important:
//   - No cudaSetDevice() call.
//   - Uses cudaGetDeviceCount() only.
//   - One thread per particle using grid-stride loop.
//   - Live CUDA now accepts a phase argument:
//        0 = load
//        1 = concentrate
//        2 = wash
//        3 = harvest
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "simulation.h"
#include "types.h"

namespace {

constexpr float PI_F = 3.14159265358979323846f;

// Must match replay reader in main.cpp.
struct ReplayHeader {
    char magic[8];               // "ROTEAFRM"
    std::uint32_t version;       // 1
    std::uint32_t particleCount;
    std::uint32_t frameCount;
    float dt;
};

struct SimParams {
    int count;
    float dt;
    float time;

    float height;
    float outerBase;
    float outerTip;
    float innerBase;
    float innerTip;

    float gForce;
    float flowRate;
    int phase;
};

// Persistent device state for live CUDA mode.
static Particle* g_deviceParticles = nullptr;
static int g_deviceParticleCount = 0;
static int g_frameCounter = 0;

// -----------------------------------------------------------------------------
// CUDA helpers
// -----------------------------------------------------------------------------

static bool checkCuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        const char* msg = cudaGetErrorString(err);
        if (msg == nullptr) msg = "unknown CUDA error";
        if (where == nullptr) where = "unknown CUDA location";

        std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", where, msg);
        return false;
    }

    return true;
}

static bool checkCudaAvailable() {
    int count = 0;

    if (!checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) {
        return false;
    }

    if (count <= 0) {
        std::fprintf(stderr, "[CUDA ERROR] No CUDA devices found.\n");
        return false;
    }

    return true;
}

static int chooseBlocks(int particleCount, int threadsPerBlock) {
    const int needed = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
    const int maxBlocks = 256;
    return std::max(1, std::min(maxBlocks, needed));
}

static int clampPhase(int phase) {
    if (phase < 0) return 0;
    if (phase > 3) return 3;
    return phase;
}

// -----------------------------------------------------------------------------
// CLI helpers for standalone mode
// -----------------------------------------------------------------------------

static int getIntArg(int argc, char** argv, const char* name, int defaultValue) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] != nullptr &&
            argv[i + 1] != nullptr &&
            std::strcmp(argv[i], name) == 0) {
            return std::max(1, std::atoi(argv[i + 1]));
        }
    }

    return std::max(1, defaultValue);
}

static float getFloatArg(int argc, char** argv, const char* name, float defaultValue) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] != nullptr &&
            argv[i + 1] != nullptr &&
            std::strcmp(argv[i], name) == 0) {
            return std::max(0.000001f, std::strtof(argv[i + 1], nullptr));
        }
    }

    return std::max(0.000001f, defaultValue);
}

static std::string getStringArg(
    int argc,
    char** argv,
    const char* name,
    const char* defaultValue
) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] != nullptr &&
            argv[i + 1] != nullptr &&
            std::strcmp(argv[i], name) == 0) {
            return std::string(argv[i + 1]);
        }
    }

    return std::string(defaultValue != nullptr ? defaultValue : "frames.bin");
}

// -----------------------------------------------------------------------------
// CPU-side initialization for standalone frame generation
// -----------------------------------------------------------------------------

static float clampCpu(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static float radiusAtYCpu(float y, float height, float bottomRadius, float topRadius) {
    const float t = clampCpu((y + 0.5f * height) / height, 0.0f, 1.0f);
    return bottomRadius + (topRadius - bottomRadius) * t;
}

static std::uint32_t lcgNext(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static float rand01(std::uint32_t& state) {
    return static_cast<float>(lcgNext(state) & 0x00FFFFFFu) /
           static_cast<float>(0x01000000u);
}

static void initializeParticlesHost(
    std::vector<Particle>& particles,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip
) {
    std::uint32_t rng = 9212026u;

    const int n = static_cast<int>(particles.size());
    const int cellStart = (n <= 1) ? 0 : static_cast<int>(0.90f * static_cast<float>(n));

    for (int i = 0; i < n; ++i) {
        const bool isCell = (i >= cellStart);

        const float y = -0.5f * height + rand01(rng) * 0.08f * height;

        const float rInner = radiusAtYCpu(y, height, innerBase, innerTip);
        const float rOuter = radiusAtYCpu(y, height, outerBase, outerTip);
        const float safeOuter = std::max(rOuter, rInner + 1.0e-4f);

        const float u = rand01(rng);
        const float r = std::sqrt(
            rInner * rInner + u * (safeOuter * safeOuter - rInner * rInner)
        );

        const float theta = 2.0f * PI_F * rand01(rng);

        Particle p{};
        p.x = r * std::cos(theta);
        p.y = y;
        p.z = r * std::sin(theta);

        p.vx = 0.0f;
        p.vy = 0.10f + 0.05f * rand01(rng);
        p.vz = 0.0f;

        p.type = isCell ? 1 : 0;
        p.diameter = isCell ? 0.12f : 0.055f;
        p.density = isCell ? 1.08f : 1.00f;

        particles[static_cast<std::size_t>(i)] = p;
    }
}

static bool writeReplayHeader(FILE* out, int particles, int frames, float dt) {
    if (out == nullptr || particles <= 0 || frames <= 0) {
        return false;
    }

    ReplayHeader header{};
    const char magic[8] = {'R', 'O', 'T', 'E', 'A', 'F', 'R', 'M'};

    std::memcpy(header.magic, magic, sizeof(header.magic));
    header.version = 1;
    header.particleCount = static_cast<std::uint32_t>(particles);
    header.frameCount = static_cast<std::uint32_t>(frames);
    header.dt = dt;

    return std::fwrite(&header, sizeof(header), 1, out) == 1;
}

static void protocolForFrame(
    int frame,
    int totalFrames,
    float& gForce,
    float& flowRate,
    int& phase
) {
    const float t =
        static_cast<float>(frame) / static_cast<float>(std::max(1, totalFrames - 1));

    if (t < 0.25f) {
        // Load / establish bed.
        phase = 0;
        gForce = 900.0f;
        flowRate = 24.0f;
    } else if (t < 0.55f) {
        // Concentrate / retain cells.
        phase = 1;
        gForce = 3000.0f;
        flowRate = 12.0f;
    } else if (t < 0.82f) {
        // Wash / buffer exchange.
        phase = 2;
        gForce = 2400.0f;
        flowRate = 18.0f;
    } else {
        // Harvest-like reverse recovery behavior.
        phase = 3;
        gForce = 700.0f;
        flowRate = 24.0f;
    }
}

} // namespace

// =============================================================================
// Device helpers
// =============================================================================

__device__ static float clampDevice(float v, float lo, float hi) {
    return fmaxf(lo, fminf(v, hi));
}

__device__ static float radiusAtYDevice(
    float y,
    float height,
    float bottomRadius,
    float topRadius
) {
    const float t = clampDevice((y + 0.5f * height) / height, 0.0f, 1.0f);
    return bottomRadius + (topRadius - bottomRadius) * t;
}

__device__ static bool saneDevice(float v) {
    return (v == v) && fabsf(v) < 1.0e6f;
}

__device__ static std::uint32_t xorshift32(std::uint32_t& state) {
    std::uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ static float randSignedDevice(std::uint32_t& state) {
    return 2.0f * static_cast<float>(xorshift32(state) & 0x00FFFFFFu) /
           16777216.0f - 1.0f;
}

// =============================================================================
// CUDA kernel
// =============================================================================

__global__ void updateParticlesKernel(
    Particle* __restrict__ particles,
    SimParams params,
    int frame
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < params.count; i += stride) {
        constexpr float eps = 1.0e-6f;

        Particle p = particles[i];

        float x = p.x;
        float y = p.y;
        float z = p.z;

        float vx = p.vx;
        float vy = p.vy;
        float vz = p.vz;

        float r = sqrtf(x * x + z * z) + eps;
        float rx = x / r;
        float rz = z / r;

        const float rOuter = radiusAtYDevice(
            y,
            params.height,
            params.outerBase,
            params.outerTip
        );

        const float rInner = radiusAtYDevice(
            y,
            params.height,
            params.innerBase,
            params.innerTip
        );

        const float gap = fmaxf(rOuter - rInner, eps);
        const float wallFrac = clampDevice((r - rInner) / gap, 0.0f, 1.0f);

        // ---------------------------------------------------------------------
        // Normalized Rotea-inspired physics model
        //
        // Real CFC behavior is governed mainly by opposing effects:
        //   1. density/diameter-dependent centrifugal migration
        //   2. Stokes-like drag from local fluid flow
        //
        // This model keeps normalized visualization units but preserves the main
        // dependencies:
        //   - larger particles respond differently than small particles
        //   - denser particles experience stronger centrifugal retention
        //   - higher flow rate increases drag-driven elutriation
        //   - higher G force increases retention/concentration
        //   - harvest reverses the pump-driven flow direction
        // ---------------------------------------------------------------------

        const float mediaDensity = 1.00f;
        const float mediaViscosity = 1.00f;

        const float d = fmaxf(p.diameter, 0.01f);
        const float densityDelta = fmaxf(p.density - mediaDensity, 0.0f);

        // Convert user-facing G-force into a normalized angular velocity.
        // This is not a calibrated unit conversion; it is a stable visual scaling.
        const float omega = 0.00032f * params.gForce;

        // Harvest/concentrate recovery reverses flow direction.
        float signedFlowRate = params.flowRate;
        if (params.phase == 3) {
            signedFlowRate = -signedFlowRate;
        }

        // Phase-level flow shaping. The main phase distinction still comes from
        // the protocol gForce/flowRate values passed in from main.cpp.
        float phaseFlowScale = 1.0f;
        if (params.phase == 1) {
            // Concentrate: lower through-flow relative to retention.
            phaseFlowScale = 0.75f;
        } else if (params.phase == 2) {
            // Wash: stronger through-flow through the bed.
            phaseFlowScale = 1.15f;
        } else if (params.phase == 3) {
            // Harvest: stronger reverse pull toward the tip/output.
            phaseFlowScale = 1.25f;
        }

        const float flow = 0.010f * signedFlowRate * phaseFlowScale;
        const float absFlow = fabsf(flow);

        // Local cross-section proxy. The narrow tip has higher local velocity,
        // while wider regions have lower local velocity.
        const float localArea = fmaxf(
            rOuter * rOuter - rInner * rInner,
            eps
        );

        const float tipArea = fmaxf(
            params.outerBase * params.outerBase -
            params.innerBase * params.innerBase,
            eps
        );

        const float velocityGain = clampDevice(tipArea / localArea, 0.35f, 3.5f);

        // Inlet/tip jet: strongest near the narrow tip.
        const float tipY = -0.5f * params.height;
        const float dyTip = y - tipY;

        const float inletFalloff =
            1.0f / (1.0f + 18.0f * (r * r + dyTip * dyTip));

        // Near-wall return/recirculation effect.
        const float returnFlow =
            clampDevice((wallFrac - 0.70f) / 0.30f, 0.0f, 1.0f);

        // Swirl from the rotating chamber. This is visualized as tangential
        // motion, not used as the main retention term.
        const float swirlVx = -rz * omega * 0.35f;
        const float swirlVz =  rx * omega * 0.35f;

        // Local fluid velocity field.
        float fluidVx =
            swirlVx +
            rx * absFlow * inletFalloff * 0.60f -
            rx * absFlow * returnFlow * 0.55f;

        float fluidVy =
            flow * velocityGain;

        float fluidVz =
            swirlVz +
            rz * absFlow * inletFalloff * 0.60f -
            rz * absFlow * returnFlow * 0.55f;

        // Stokes-like drag coefficient:
        //   F_drag ∝ 3πηvd
        //
        // For acceleration response, smaller particles should be more influenced
        // by fluid drag than larger particles, so the response is scaled with
        // approximate particle mass ∝ density * d^3.
        const float stokesCoeff = 3.0f * PI_F * mediaViscosity * d;
        const float pseudoMass = fmaxf(p.density * d * d * d, 0.0001f);

        float dragResponse =
            0.0015f * stokesCoeff / pseudoMass;

        dragResponse = clampDevice(dragResponse, 0.020f, 0.180f);

        // Density-dependent centrifugal migration:
        //   F_cent ∝ (rho_p - rho_m) * pi * d^3 * omega^2 * r / 6
        //
        // This is scaled into normalized visual acceleration units.
        const float centrifugal =
            densityDelta *
            PI_F *
            d * d * d *
            omega * omega *
            r / 6.0f;

        const float centrifugalScale = 5000.0f;

        float axCent = rx * centrifugal * centrifugalScale;
        float ayCent = 0.0f;
        float azCent = rz * centrifugal * centrifugalScale;

        // During harvest, reverse flow dominates recovery from the chamber.
        // Keep the centrifugal effect but reduce radial bed tightening slightly
        // so cells can be pulled out instead of only retained.
        if (params.phase == 3) {
            axCent *= 0.45f;
            azCent *= 0.45f;
        }

        // Drag pulls particle velocity toward local fluid velocity.
        float axDrag = (fluidVx - vx) * dragResponse;
        float ayDrag = (fluidVy - vy) * dragResponse;
        float azDrag = (fluidVz - vz) * dragResponse;

        // Mild bed-stabilizing bias for cell-like particles during concentrate
        // and wash. This preserves the visible fluidized-bed behavior without
        // replacing the force model.
        if (p.type == 1 && params.phase == 1) {
            ayDrag -= 0.0025f;
        }

        if (p.type == 1 && params.phase == 2) {
            ayDrag -= 0.0010f;
        }

        // Small deterministic perturbation prevents perfectly laminar visuals.
        // Kept small so it does not dominate the force model.
        std::uint32_t rng =
            0x9E3779B9u ^
            static_cast<std::uint32_t>(i * 747796405u) ^
            static_cast<std::uint32_t>(frame * 2891336453u);

        const float noiseScale = 0.0008f;
        axDrag += noiseScale * randSignedDevice(rng);
        ayDrag += noiseScale * randSignedDevice(rng);
        azDrag += noiseScale * randSignedDevice(rng);

        // Integrate velocity.
        vx += (axDrag + axCent) * params.dt;
        vy += (ayDrag + ayCent) * params.dt;
        vz += (azDrag + azCent) * params.dt;

        // Viscous damping for numerical stability.
        vx *= 0.992f;
        vy *= 0.992f;
        vz *= 0.992f;

        // Integrate position.
        x += vx * params.dt;
        y += vy * params.dt;
        z += vz * params.dt;

        // ---------------------------------------------------------------------
        // Boundary behavior
        // ---------------------------------------------------------------------

        const float topY = 0.5f * params.height;
        const float bottomY = -0.5f * params.height;

        if (params.phase == 3) {
            // Harvest / reverse flow: material exits toward the tip/bottom.
            if (y < bottomY) {
                y = topY;
                vx *= 0.25f;
                vy = -fabsf(vy);
                vz *= 0.25f;
            }

            if (y > topY) {
                y = topY;
                vy = -fabsf(vy);
            }
        } else {
            // Forward flow: material exits upward and re-enters at the tip.
            if (y > topY) {
                y = bottomY;
                vx *= 0.25f;
                vy = fabsf(vy) + 0.03f;
                vz *= 0.25f;
            }

            if (y < bottomY) {
                y = bottomY;
                vy = fabsf(vy) + 0.03f;
            }
        }

        // Clamp to cone annulus after movement.
        const float newOuter = radiusAtYDevice(
            y,
            params.height,
            params.outerBase,
            params.outerTip
        );

        const float newInner = radiusAtYDevice(
            y,
            params.height,
            params.innerBase,
            params.innerTip
        );

        float newR = sqrtf(x * x + z * z) + eps;

        if (newR > newOuter) {
            const float scale = newOuter / newR;
            x *= scale;
            z *= scale;

            // Wall collision / shear loss.
            vx *= 0.45f;
            vz *= 0.45f;

            newR = newOuter;
        }

        if (newR < newInner) {
            const float scale = newInner / fmaxf(newR, eps);
            x *= scale;
            z *= scale;

            vx *= 0.45f;
            vz *= 0.45f;
        }

        // Last-resort numerical guard.
        if (!saneDevice(x)  ||
            !saneDevice(y)  ||
            !saneDevice(z)  ||
            !saneDevice(vx) ||
            !saneDevice(vy) ||
            !saneDevice(vz)) {
            x = 0.20f;
            y = -0.45f * params.height;
            z = 0.0f;
            vx = 0.0f;
            vy = 0.10f;
            vz = 0.0f;
        }

        p.x = x;
        p.y = y;
        p.z = z;

        p.vx = vx;
        p.vy = vy;
        p.vz = vz;

        particles[i] = p;
    }
}

// Launches the particle update kernel using a fixed thread block size and a
// capped block count. The kernel itself uses a grid-stride loop, so all
// particles are covered even when particleCount exceeds blocks * threads.
static bool launchUpdate(Particle* deviceParticles, const SimParams& params, int frame) {
    if (deviceParticles == nullptr || params.count <= 0) {
        std::fprintf(stderr, "[CUDA ERROR] launchUpdate received invalid input.\n");
        return false;
    }

    constexpr int threads = 256;
    const int blocks = chooseBlocks(params.count, threads);

    updateParticlesKernel<<<blocks, threads>>>(deviceParticles, params, frame);

    return checkCuda(cudaGetLastError(), "updateParticlesKernel launch");
}

// =============================================================================
// Public C API for live CUDA mode
// =============================================================================

// Allocates the persistent device-side particle buffer used by live CUDA mode.
// The allocation is reused across frames to avoid cudaMalloc/cudaFree inside
// the visual or headless update loop.
extern "C" bool initCudaParticleBuffer(int count) {
    if (count <= 0) {
        std::fprintf(stderr, "[CUDA ERROR] initCudaParticleBuffer count <= 0\n");
        return false;
    }

    if (!checkCudaAvailable()) {
        return false;
    }

    if (g_deviceParticles != nullptr) {
        cudaFree(g_deviceParticles);
        g_deviceParticles = nullptr;
        g_deviceParticleCount = 0;
    }

    const std::size_t bytes =
        sizeof(Particle) * static_cast<std::size_t>(count);

    if (!checkCuda(
            cudaMalloc(&g_deviceParticles, bytes),
            "cudaMalloc persistent particle buffer"
        )) {
        return false;
    }

    g_deviceParticleCount = count;
    g_frameCounter = 0;

    std::fprintf(
        stderr,
        "[CUDA] Persistent particle buffer allocated: %d particles, %.2f MB\n",
        count,
        static_cast<double>(bytes) / (1024.0 * 1024.0)
    );

    return true;
}

// Live CUDA update entry point called from main.cpp.
//
// Current architecture:
//   host Particle array -> cudaMemcpy H2D -> CUDA kernel update
//   -> cudaDeviceSynchronize -> cudaMemcpy D2H -> host Particle array
//
// This keeps the OpenGL renderer simple because rendering still consumes the
// CPU-side particle vector. The tradeoff is that live CUDA mode includes
// host-device transfer and synchronization overhead each update. This is why
// the project separates visual benchmarks from headless update benchmarks.
extern "C" bool updateParticlesCUDA(
    Particle* particles,
    int count,
    float dt,
    float gForce,
    float flow,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip,
    int phase
) {
    if (particles == nullptr || count <= 0) {
        return false;
    }

    if (g_deviceParticles == nullptr || g_deviceParticleCount != count) {
        if (!initCudaParticleBuffer(count)) {
            return false;
        }
    }

    const std::size_t bytes =
        sizeof(Particle) * static_cast<std::size_t>(count);

    SimParams params{};
    params.count = count;
    params.dt = dt;
    params.time = static_cast<float>(g_frameCounter) * dt;
    params.height = height;
    params.outerBase = outerBase;
    params.outerTip = outerTip;
    params.innerBase = innerBase;
    params.innerTip = innerTip;
    params.gForce = gForce;
    params.flowRate = flow;
    params.phase = clampPhase(phase);

    if (!checkCuda(
            cudaMemcpy(g_deviceParticles, particles, bytes, cudaMemcpyHostToDevice),
            "live H2D particle copy"
        )) {
        return false;
    }

    if (!launchUpdate(g_deviceParticles, params, g_frameCounter)) {
        return false;
    }

    if (!checkCuda(cudaDeviceSynchronize(), "live cudaDeviceSynchronize")) {
        return false;
    }

    if (!checkCuda(
            cudaMemcpy(particles, g_deviceParticles, bytes, cudaMemcpyDeviceToHost),
            "live D2H particle copy"
        )) {
        return false;
    }

    ++g_frameCounter;
    return true;
}

// Releases persistent CUDA state. Called when live CUDA mode exits or when
// the benchmark completes.
extern "C" void shutdownCudaParticleBuffer() {
    if (g_deviceParticles != nullptr) {
        cudaFree(g_deviceParticles);
        g_deviceParticles = nullptr;
        g_deviceParticleCount = 0;
    }

    g_frameCounter = 0;

    std::fprintf(stderr, "[CUDA] Persistent particle buffer released.\n");
}

// =============================================================================
// Standalone CUDA frame generator
// =============================================================================

#ifdef CUDA_FRAMES_STANDALONE

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] != nullptr &&
            (std::strcmp(argv[i], "--help") == 0 ||
             std::strcmp(argv[i], "-h") == 0)) {
            std::printf(
                "Usage: cuda_frames "
                "[--particles N] [--frames N] [--dt F] [--output PATH]\n"
                "  --particles N     particle count, default 5000\n"
                "  --frames N        frame count, default 600\n"
                "  --dt F            timestep, default 0.016\n"
                "  --output PATH     output file, default frames.bin\n"
            );
            return 0;
        }
    }

    const int particleCount = getIntArg(argc, argv, "--particles", 5000);
    const int frameCount = getIntArg(argc, argv, "--frames", 600);
    const float dt = getFloatArg(argc, argv, "--dt", 0.016f);
    const std::string outputPath =
        getStringArg(argc, argv, "--output", "frames.bin");

    // Geometry convention:
    // y = -height/2 is the narrow cone tip / inlet.
    // y = +height/2 is the wider upper chamber.
    constexpr float chamberHeight = 2.5f;
    constexpr float outerBase = 0.28f;
    constexpr float outerTip  = 1.15f;
    constexpr float innerBase = 0.12f;
    constexpr float innerTip  = 0.55f;

    std::printf("Rotea CUDA offline frame generator\n");
    std::printf("  Particles      : %d\n", particleCount);
    std::printf("  Frames         : %d\n", frameCount);
    std::printf("  dt             : %.6f s\n", dt);
    std::printf("  Particle size  : %zu bytes\n", sizeof(Particle));
    std::printf("  Output         : %s\n", outputPath.c_str());
    std::fflush(stdout);

    if (!checkCudaAvailable()) {
        return 1;
    }

    const std::size_t frameBytes =
        sizeof(Particle) * static_cast<std::size_t>(particleCount);

    std::vector<Particle> hostParticles;
    hostParticles.resize(static_cast<std::size_t>(particleCount));

    initializeParticlesHost(
        hostParticles,
        chamberHeight,
        outerBase,
        outerTip,
        innerBase,
        innerTip
    );

    Particle* deviceParticles = nullptr;

    if (!checkCuda(
            cudaMalloc(&deviceParticles, frameBytes),
            "standalone cudaMalloc particles"
        )) {
        return 1;
    }

    if (!checkCuda(
            cudaMemcpy(
                deviceParticles,
                hostParticles.data(),
                frameBytes,
                cudaMemcpyHostToDevice
            ),
            "standalone initial H2D copy"
        )) {
        cudaFree(deviceParticles);
        return 1;
    }

    FILE* out = std::fopen(outputPath.c_str(), "wb");

    if (out == nullptr) {
        std::fprintf(
            stderr,
            "[ERROR] Failed to open output file: %s\n",
            outputPath.c_str()
        );
        cudaFree(deviceParticles);
        return 1;
    }

    if (!writeReplayHeader(out, particleCount, frameCount, dt)) {
        std::fprintf(stderr, "[ERROR] Failed to write replay header.\n");
        std::fclose(out);
        cudaFree(deviceParticles);
        return 1;
    }

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};

    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(startEvent), "cudaEventRecord start");

    constexpr int threads = 256;
    const int blocks = chooseBlocks(particleCount, threads);

    std::fprintf(
        stderr,
        "[CUDA FRAME GEN] grid: %d blocks x %d threads\n",
        blocks,
        threads
    );

    for (int frame = 0; frame < frameCount; ++frame) {
        float gForce = 2500.0f;
        float flowRate = 16.0f;
        int phase = 0;

        protocolForFrame(frame, frameCount, gForce, flowRate, phase);

        SimParams params{};
        params.count = particleCount;
        params.dt = dt;
        params.time = static_cast<float>(frame) * dt;
        params.height = chamberHeight;
        params.outerBase = outerBase;
        params.outerTip = outerTip;
        params.innerBase = innerBase;
        params.innerTip = innerTip;
        params.gForce = gForce;
        params.flowRate = flowRate;
        params.phase = phase;

        if (!launchUpdate(deviceParticles, params, frame)) {
            std::fclose(out);
            cudaFree(deviceParticles);
            return 1;
        }

        if (!checkCuda(cudaDeviceSynchronize(), "standalone frame sync")) {
            std::fclose(out);
            cudaFree(deviceParticles);
            return 1;
        }

        if (!checkCuda(
                cudaMemcpy(
                    hostParticles.data(),
                    deviceParticles,
                    frameBytes,
                    cudaMemcpyDeviceToHost
                ),
                "standalone frame D2H copy"
            )) {
            std::fclose(out);
            cudaFree(deviceParticles);
            return 1;
        }

        if (std::fwrite(hostParticles.data(), frameBytes, 1, out) != 1) {
            std::fprintf(stderr, "[ERROR] Failed to write frame %d\n", frame);
            std::fclose(out);
            cudaFree(deviceParticles);
            return 1;
        }

        if ((frame + 1) % 100 == 0 || frame + 1 == frameCount) {
            std::fprintf(
                stderr,
                "[CUDA FRAME GEN] %d / %d frames written\n",
                frame + 1,
                frameCount
            );
        }
    }

    checkCuda(cudaEventRecord(stopEvent), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stop");

    float elapsedMs = 0.0f;
    checkCuda(
        cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent),
        "cudaEventElapsedTime"
    );

    std::fclose(out);
    cudaFree(deviceParticles);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    const std::size_t totalBytes =
        sizeof(ReplayHeader) +
        static_cast<std::size_t>(frameCount) * frameBytes;

    std::printf("\n========== CUDA Frame Generator Summary ==========\n");
    std::printf("Output file     : %s\n", outputPath.c_str());
    std::printf("Header bytes    : %zu\n", sizeof(ReplayHeader));
    std::printf("Frame bytes     : %zu\n", frameBytes);
    std::printf(
        "Total file size : %zu bytes (%.2f MB)\n",
        totalBytes,
        static_cast<double>(totalBytes) / (1024.0 * 1024.0)
    );
    std::printf("GPU time        : %.4f ms total\n", elapsedMs);
    std::printf(
        "Avg per frame   : %.6f ms/frame\n",
        elapsedMs / static_cast<float>(frameCount)
    );
    std::printf("==================================================\n");

    return 0;
}

#endif // CUDA_FRAMES_STANDALONE