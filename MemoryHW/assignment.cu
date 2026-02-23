// Dylan Renard
// EN.605.617 Introduction to GPU Programming (JHU)
// Professor Chance Pascale
// February 21, 2026
//
// CUDA Memory Hierarchy Assignment
// 1D LUT Image Filter with Optional Shared-Memory Gaussian Blur
//
// Overview:
// This program demonstrates practical usage of the CUDA memory hierarchy by
// applying a selectable 1D color Look-Up Table (LUT) to an input image,
// followed optionally by a 3x3 Gaussian blur implemented using shared memory tiling.
//
// Memory Hierarchy Demonstrated:
//   • Host Memory      - Image loading, LUT construction, and final image storage
//   • Global Memory    - Device image buffers (input, output, and ping-pong blur buffers)
//   • Constant Memory  - Precomputed 1D LUT tables stored in __constant__ memory
//                        and broadcast across threads during color mapping
//   • Shared Memory    - Tile plus halo staging for 3x3 stencil blur kernel
//   • Registers        - Per-thread local variables used inside kernels
//
// Program Flow:
//   1. Input image is converted to PPM (P6) using ImageMagick if necessary.
//   2. LUT tables are built on the host and copied into constant memory.
//   3. Image is copied Host → Device (global memory).
//   4. LUT kernel executes (constant + global memory).
//   5. Optional multi-pass shared-memory blur executes.
//   6. Image is copied Device → Host.
//   7. Result is written and converted back to requested format.
//
// Performance Instrumentation:
//   • Separate timing for Host→Device, Kernel, and Device→Host.
//   • Variable block dimensions via --bx and --by.
//   • Structured RESULT line printed for automated grid search parsing.
//
// Output Location:
//   - Intermediate PPM files written to ./OutputImages/
//   - If --out is filename only, final output is also saved to ./OutputImages/
//
// Command-Line Usage:
//   ./cuda_filter --in <path> --out <path>
//                 [--mode warm|blue|bw]
//                 [--intensity 0..1]
//                 [--blur 0|1] [--blur-passes N]
//                 [--bx N] [--by N]
//                 [--keep-temp 0|1]
//
// Example:
//   ./cuda_filter --in canyon.jpg --out canyon_bw.jpg --mode bw --bx 32 --by 8
//   ./cuda_filter --in input.ppm --out output.ppm --blur 1 --blur-passes 3
//


#include <cuda_runtime.h>

#include <cctype>
#include <cmath>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)


enum class Mode : int { WARM = 0, BLUE = 1, BW = 2 };

// Constant memory LUTs: 3 modes × 3 channels × 256 entries
__constant__ unsigned char c_lutWarmR[256];
__constant__ unsigned char c_lutWarmG[256];
__constant__ unsigned char c_lutWarmB[256];

__constant__ unsigned char c_lutBlueR[256];
__constant__ unsigned char c_lutBlueG[256];
__constant__ unsigned char c_lutBlueB[256];

__constant__ unsigned char c_lutBWR[256];
__constant__ unsigned char c_lutBWG[256];
__constant__ unsigned char c_lutBWB[256];


static float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static unsigned char f2u8(float x) {
  x = clamp01(x);
  return (unsigned char)(x * 255.0f + 0.5f);
}

// Very strong contrast curve (gamma + S-curve)
static float strongCurve(float x) {
  // lift toe a bit
  x = clamp01(x + 0.04f * (1.0f - x));

  // gamma for midtone shaping
  float g = 0.85f;               // <1 brightens mids a bit
  x = powf(x, g);

  // S-curve (very strong)
  float y = x * x * (3.0f - 2.0f * x);  // smoothstep
  // push harder: blend toward y heavily
  return 0.15f * x + 0.85f * y;
}

static void buildWarmLUT(unsigned char R[256], unsigned char G[256], unsigned char B[256]) {
  // EXTREME warm:
  // - heavy red/orange push
  // - heavy blue pull
  // - noticeable contrast bump
  for (int i = 0; i < 256; i++) {
    float x = i / 255.0f;
    float c = strongCurve(x);

    // highlight bias (very aggressive)
    float h = c * c * c; // pushes effect into highlights hard

    // Temperature-style offsets (shift mids/highlights)
    float r = c + 0.28f * h;          // big warm lift
    float g = c + 0.10f * h;          // modest lift to avoid neon reds
    float b = c - 0.32f * h;          // heavy blue suppression

    // Additional channel gain (also aggressive)
    r *= (1.0f + 0.45f * h);          // up to +45%
    g *= (1.0f + 0.15f * h);          // up to +15%
    b *= (1.0f - 0.50f * h);          // down to -50%

    R[i] = f2u8(r);
    G[i] = f2u8(g);
    B[i] = f2u8(b);
  }
}

static void buildBlueLUT(unsigned char R[256], unsigned char G[256], unsigned char B[256]) {
  // EXTREME cool/blue:
  // - heavy blue push
  // - heavy red pull
  // - slight green pull for colder look
  for (int i = 0; i < 256; i++) {
    float x = i / 255.0f;
    float c = strongCurve(x);

    float h = c * c * c;

    float r = c - 0.32f * h;          // heavy red suppression
    float g = c - 0.12f * h;          // slight green suppression
    float b = c + 0.28f * h;          // big blue lift

    r *= (1.0f - 0.50f * h);          // down to -50%
    g *= (1.0f - 0.15f * h);          // down to -15%
    b *= (1.0f + 0.45f * h);          // up to +45%

    R[i] = f2u8(r);
    G[i] = f2u8(g);
    B[i] = f2u8(b);
  }
}

static void buildBWLUT(unsigned char R[256], unsigned char G[256], unsigned char B[256]) {
  // BW: map intensity to grayscale (identity curve); kernel will compute luma and index these.
  for (int i = 0; i < 256; i++) {
    R[i] = (unsigned char)i;
    G[i] = (unsigned char)i;
    B[i] = (unsigned char)i;
  }
}

static void uploadLUTsToConstant() {
  unsigned char warmR[256], warmG[256], warmB[256];
  unsigned char blueR[256], blueG[256], blueB[256];
  unsigned char bwR[256], bwG[256], bwB[256];

  buildWarmLUT(warmR, warmG, warmB);
  buildBlueLUT(blueR, blueG, blueB);
  buildBWLUT(bwR, bwG, bwB);

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmR, warmR, sizeof(warmR)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmG, warmG, sizeof(warmG)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmB, warmB, sizeof(warmB)));

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueR, blueR, sizeof(blueR)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueG, blueG, sizeof(blueG)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueB, blueB, sizeof(blueB)));

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWR, bwR, sizeof(bwR)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWG, bwG, sizeof(bwG)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWB, bwB, sizeof(bwB)));
}   

static std::string toLower(std::string s) {
  for (char &c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static std::string getExtLower(const std::string &path) {
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return "";
  return toLower(path.substr(pos + 1));
}

static std::string getBaseName(const std::string &path) {
  // strip directory
  size_t slash = path.find_last_of("/\\");
  std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
  // strip extension
  size_t dot = name.find_last_of('.');
  if (dot != std::string::npos) name = name.substr(0, dot);
  return name;
}

static bool hasDirComponent(const std::string& path) {
  return path.find('/') != std::string::npos || path.find('\\') != std::string::npos;
}

static void ensureDir(const std::string& dir) {
  try {
    std::filesystem::create_directories(dir);
  } catch (const std::exception& e) {
    std::cerr << "Failed to create directory '" << dir << "': " << e.what() << "\n";
    std::exit(1);
  }
}

static std::string joinPath(const std::string& dir, const std::string& file) {
  std::filesystem::path p = std::filesystem::path(dir) / std::filesystem::path(file);
  return p.string();
}

static bool fileExists(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  return f.good();
}

static int runCmd(const std::string &cmd) {
  // Note: coursework-simple. Quotes in cmd handle spaces.
  return std::system(cmd.c_str());
}

static bool haveMagickCommand(std::string &cmdNameOut) {
  // Prefer `magick` if present; fallback to `convert`.
  // We detect by running "<cmd> -version" and checking return code.
  if (std::system("magick -version > /dev/null 2>&1") == 0) {
    cmdNameOut = "magick";
    return true;
  }
  if (std::system("convert -version > /dev/null 2>&1") == 0) {
    cmdNameOut = "convert";
    return true;
  }
  return false;
}

// Minimal PPM (P6) reader/writer supporting maxval=255.
// Returns rgb bytes in a vector of size w*h*3.
static bool readPPM_P6(const std::string &path, int &w, int &h, std::vector<unsigned char> &rgb) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  auto readToken = [&](std::string &tok) -> bool {
    tok.clear();
    char ch;
    // skip whitespace and comments
    while (in.get(ch)) {
      if (std::isspace((unsigned char)ch)) continue;
      if (ch == '#') {
        std::string dummy;
        std::getline(in, dummy);
        continue;
      }
      tok.push_back(ch);
      break;
    }
    if (tok.empty()) return false;
    while (in.get(ch)) {
      if (std::isspace((unsigned char)ch)) break;
      tok.push_back(ch);
    }
    return true;
  };

  std::string tok;
  if (!readToken(tok)) return false;
  if (tok != "P6") {
    std::cerr << "PPM read error: expected P6, got " << tok << "\n";
    return false;
  }

  if (!readToken(tok)) return false;
  w = std::stoi(tok);
  if (!readToken(tok)) return false;
  h = std::stoi(tok);
  if (!readToken(tok)) return false;
  int maxval = std::stoi(tok);
  if (maxval != 255) {
    std::cerr << "PPM read error: only maxval=255 supported, got " << maxval << "\n";
    return false;
  }

  const size_t bytes = (size_t)w * (size_t)h * 3u;
  rgb.resize(bytes);
  in.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)bytes);
  if ((size_t)in.gcount() != bytes) {
    std::cerr << "PPM read error: expected " << bytes << " bytes, got " << in.gcount() << "\n";
    return false;
  }
  return true;
}

static bool writePPM_P6(const std::string &path, int w, int h, const unsigned char *rgb) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P6\n" << w << " " << h << "\n255\n";
  out.write(reinterpret_cast<const char*>(rgb), (std::streamsize)((size_t)w * (size_t)h * 3u));
  return (bool)out;
}

__device__ __forceinline__ unsigned char lerpU8(unsigned char a, unsigned char b, float t) {
  float out = (1.0f - t) * (float)a + t * (float)b;
  int v = (int)(out + 0.5f);
  return (v < 0) ? 0 : (v > 255 ? 255 : (unsigned char)v);
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ uchar3 loadPixelClampedRGB(
    const unsigned char* in, int width, int height, int gx, int gy) {
  gx = clampi(gx, 0, width - 1);
  gy = clampi(gy, 0, height - 1);
  int idx = (gy * width + gx) * 3;
  return make_uchar3(in[idx + 0], in[idx + 1], in[idx + 2]);
}

// blur3x3Shared
//
// Applies a 3x3 Gaussian blur using shared memory tiling.
//
// Memory Usage:
//   - Global memory: input and output image buffers
//   - Shared memory: per-block tile including 1-pixel halo on all sides
//   - Registers: per-thread accumulators and temporary pixel values
//
// Each thread maps to one output pixel. Threads within a block cooperatively
// load a tile of pixels from global memory into shared memory. The tile is
// padded with a 1-pixel halo so that every thread can compute its 3x3 stencil
// entirely from shared memory after synchronization.
//
// This reduces redundant global memory reads, since neighboring threads reuse
// overlapping pixels from the shared tile instead of reloading them from
// global memory multiple times.
//
// Edge handling is performed via clamped loads in loadPixelClampedRGB().
__global__ void blur3x3Shared(const unsigned char* in, unsigned char* out, int width, int height) {
  int tx = (int)threadIdx.x;
  int ty = (int)threadIdx.y;
  int x  = (int)blockIdx.x * (int)blockDim.x + tx;
  int y  = (int)blockIdx.y * (int)blockDim.y + ty;

  int tileW = (int)blockDim.x + 2;
  extern __shared__ uchar3 sTile[];

  int sx = tx + 1;
  int sy = ty + 1;
  int sCenter = sy * tileW + sx;

  // center
  sTile[sCenter] = loadPixelClampedRGB(in, width, height, x, y);

  // halos (left/right are same row as center)
  if (tx == 0)
    sTile[sCenter - 1] = loadPixelClampedRGB(in, width, height, x - 1, y);

  if (tx == (int)blockDim.x - 1)
    sTile[sCenter + 1] = loadPixelClampedRGB(in, width, height, x + 1, y);

  // halos (top/bottom are +/- tileW from center)
  if (ty == 0)
    sTile[sCenter - tileW] = loadPixelClampedRGB(in, width, height, x, y - 1);

  if (ty == (int)blockDim.y - 1)
    sTile[sCenter + tileW] = loadPixelClampedRGB(in, width, height, x, y + 1);

  // corners (diagonals)
  if (tx == 0 && ty == 0)
    sTile[sCenter - tileW - 1] = loadPixelClampedRGB(in, width, height, x - 1, y - 1);

  if (tx == (int)blockDim.x - 1 && ty == 0)
    sTile[sCenter - tileW + 1] = loadPixelClampedRGB(in, width, height, x + 1, y - 1);

  if (tx == 0 && ty == (int)blockDim.y - 1)
    sTile[sCenter + tileW - 1] = loadPixelClampedRGB(in, width, height, x - 1, y + 1);

  if (tx == (int)blockDim.x - 1 && ty == (int)blockDim.y - 1)
    sTile[sCenter + tileW + 1] = loadPixelClampedRGB(in, width, height, x + 1, y + 1);
  __syncthreads();

  if (x >= width || y >= height) return;

  const int w00 = 1, w01 = 2, w02 = 1;
  const int w10 = 2, w11 = 4, w12 = 2;
  const int w20 = 1, w21 = 2, w22 = 1;

  uchar3 p00 = sTile[(sy - 1) * tileW + (sx - 1)];
  uchar3 p01 = sTile[(sy - 1) * tileW + (sx    )];
  uchar3 p02 = sTile[(sy - 1) * tileW + (sx + 1)];
  uchar3 p10 = sTile[(sy    ) * tileW + (sx - 1)];
  uchar3 p11 = sTile[(sy    ) * tileW + (sx    )];
  uchar3 p12 = sTile[(sy    ) * tileW + (sx + 1)];
  uchar3 p20 = sTile[(sy + 1) * tileW + (sx - 1)];
  uchar3 p21 = sTile[(sy + 1) * tileW + (sx    )];
  uchar3 p22 = sTile[(sy + 1) * tileW + (sx + 1)];

  int r = w00*p00.x + w01*p01.x + w02*p02.x +
          w10*p10.x + w11*p11.x + w12*p12.x +
          w20*p20.x + w21*p21.x + w22*p22.x;

  int g = w00*p00.y + w01*p01.y + w02*p02.y +
          w10*p10.y + w11*p11.y + w12*p12.y +
          w20*p20.y + w21*p21.y + w22*p22.y;

  int b = w00*p00.z + w01*p01.z + w02*p02.z +
          w10*p10.z + w11*p11.z + w12*p12.z +
          w20*p20.z + w21*p21.z + w22*p22.z;

  // divide by 16 with rounding
  r = (r + 8) >> 4;
  g = (g + 8) >> 4;
  b = (b + 8) >> 4;

  int outIdx = (y * width + x) * 3;
  out[outIdx + 0] = (unsigned char)clampi(r, 0, 255);
  out[outIdx + 1] = (unsigned char)clampi(g, 0, 255);
  out[outIdx + 2] = (unsigned char)clampi(b, 0, 255);
} 

// lutKernel
//
// Applies a selectable 1D color Look-Up Table (LUT) to each pixel.
//
// Memory Usage:
//   - Global memory: input and output image buffers
//   - Constant memory: precomputed LUT tables (broadcast across threads)
//   - Registers: per-thread RGB values and temporaries
//
// Each thread processes one pixel. The original RGB values are used as
// indices into LUT tables stored in constant memory. Because many threads
// access LUT values simultaneously, constant memory provides efficient
// broadcast caching behavior.
//
// The final pixel is optionally blended with the original value based on
// the provided intensity parameter.
__global__ void lutKernel(const unsigned char *in, unsigned char *out,
                          int width, int height, int modeInt, float intensity) {
  int x = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  int y = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;
  unsigned char r = in[idx + 0];
  unsigned char g = in[idx + 1];
  unsigned char b = in[idx + 2];

  unsigned char r2 = r, g2 = g, b2 = b;

  // Select LUT from constant memory
  if (modeInt == (int)Mode::WARM) {
    r2 = c_lutWarmR[r]; g2 = c_lutWarmG[g]; b2 = c_lutWarmB[b];
  } else if (modeInt == (int)Mode::BLUE) {
    r2 = c_lutBlueR[r]; g2 = c_lutBlueG[g]; b2 = c_lutBlueB[b];
  } else { // BW
    // Compute luma and map all channels to the same value via BW LUT
    // (BT.601 luma coefficients)
    int yv = (int)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
    yv = (yv < 0) ? 0 : (yv > 255 ? 255 : yv);
    unsigned char yy = (unsigned char)yv;
    r2 = c_lutBWR[yy]; g2 = c_lutBWG[yy]; b2 = c_lutBWB[yy];
  }

  // Blend original with LUT output
  out[idx + 0] = lerpU8(r, r2, intensity);
  out[idx + 1] = lerpU8(g, g2, intensity);
  out[idx + 2] = lerpU8(b, b2, intensity);
}

struct Args {
  std::string inPath;
  std::string outPath;
  std::string mode = "warm";
  float intensity = 1.0f;
  bool blur = false;
  int blurPasses = 1;
  int bx = 16;
  int by = 16;
  bool keepTemp = false;
};

static void usage() {
  std::cerr <<
    "Usage:\n"
    "  ./cuda_filter --in <path> --out <path> [--bx N] [--by N] [--keep-temp 0|1]\n"
    "\nExamples:\n"
    "  ./cuda_filter --in input.jpg --out output.jpg --bx 16 --by 16\n"
    "  ./cuda_filter --in input.ppm --out output.ppm --bx 32 --by 8\n"
    "  ./cuda_filter --in <path> --out <path> --mode warm|blue|bw --intensity <0..1> [--bx N] [--by N]\n"
    "  --blur 0|1   Enable shared-memory 3x3 Gaussian blur after LUT\n"
    "  --blur-passes N   Number of blur passes (default 1). Larger = blurrier.\n";
}

static Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto needVal = [&](const char *flag) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        usage();
        std::exit(1);
      }
    };

    if (k == "--in") {
      needVal("--in");
      a.inPath = argv[++i];
    } else if (k == "--out") {
      needVal("--out");
      a.outPath = argv[++i];
    } else if (k == "--bx") {
      needVal("--bx");
      a.bx = std::stoi(argv[++i]);
    } else if (k == "--by") {
      needVal("--by");
      a.by = std::stoi(argv[++i]);
    } else if (k == "--keep-temp") {
      needVal("--keep-temp");
      a.keepTemp = (std::stoi(argv[++i]) != 0);
    } else if (k == "--help" || k == "-h") {
      usage();
      std::exit(0);
    } else if (k == "--mode") {
      needVal("--mode");
      a.mode = toLower(argv[++i]);
    } else if (k == "--intensity") {
      needVal("--intensity");
      a.intensity = std::stof(argv[++i]);
    } else if (k == "--blur") {
      needVal("--blur");
      a.blur = (std::stoi(argv[++i]) != 0);
    } else if (k == "--blur-passes") {
      needVal("--blur-passes");
      a.blurPasses = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown arg: " << k << "\n";
      usage();
      std::exit(1);
    }
  }

  if (a.inPath.empty() || a.outPath.empty()) {
    usage();
    std::exit(1);
  }
  if (a.bx <= 0 || a.by <= 0 || a.bx > 1024 || a.by > 1024) {
    std::cerr << "Invalid block dims: bx/by must be in (0..1024]\n";
    std::exit(1);
  }

  if (a.intensity < 0.0f) a.intensity = 0.0f;
  if (a.intensity > 1.0f) a.intensity = 1.0f;
  if (a.mode != "warm" && a.mode != "blue" && a.mode != "bw") {
    std::cerr << "Invalid --mode. Use warm|blue|bw\n";
    std::exit(1);
  }

  if (a.blurPasses < 1) a.blurPasses = 1;
  if (a.blurPasses > 20) a.blurPasses = 20; // safety cap

  if (a.blur) {
    if (a.bx > 32) a.bx = 32;
    if (a.by > 32) a.by = 32;
  }

  return a;
}


struct ImagePaths {
  std::string outDir;
  std::string magickCmd;

  std::string ppmInPath;
  std::string ppmOutPath;

  std::string tmpInPPM;
  std::string tmpOutPPM;

  bool usedTmpIn = false;
  bool usedTmpOut = false;
};

static bool ensureMagick(std::string &magickCmd) {
  if (haveMagickCommand(magickCmd)) return true;
  std::cerr
      << "Error: ImageMagick not found. Install it or provide .ppm input/output.\n"
      << "Try: sudo apt install -y imagemagick\n";
  return false;
}

static ImagePaths preparePathsAndConvertInput(const Args &argsIn,
                                             const std::string &outExt,
                                             ImagePaths p) {
  const std::string inExt = getExtLower(argsIn.inPath);
  const std::string base  = getBaseName(argsIn.inPath);

  p.tmpInPPM  = joinPath(p.outDir, base + "_cuda_in.ppm");
  p.tmpOutPPM = joinPath(p.outDir, base + "_cuda_out.ppm");

  p.ppmInPath  = argsIn.inPath;
  p.ppmOutPath = argsIn.outPath;

  // Convert input to PPM if needed
  if (inExt != "ppm") {
    std::ostringstream cmd;
    cmd << p.magickCmd << " \"" << argsIn.inPath
        << "\" -alpha off -depth 8 \"" << p.tmpInPPM << "\"";
    std::cout << "[Convert] " << cmd.str() << "\n";
    if (runCmd(cmd.str()) != 0 || !fileExists(p.tmpInPPM)) {
      std::cerr << "Failed to convert input to PPM.\n";
      std::exit(1);
    }
    p.ppmInPath = p.tmpInPPM;
    p.usedTmpIn = true;
  }

  // If output isn't PPM, write PPM first then convert at end
  if (outExt != "ppm") {
    p.ppmOutPath = p.tmpOutPPM;
    p.usedTmpOut = true;
  }

  return p;
}

static int resolveModeInt(const std::string &mode) {
  return (mode == "warm") ? (int)Mode::WARM :
         (mode == "blue") ? (int)Mode::BLUE :
                            (int)Mode::BW;
}

struct DeviceBuffers {
  unsigned char *d_in = nullptr;
  unsigned char *d_out = nullptr;
  unsigned char *d_tmp = nullptr;
  unsigned char *d_tmp2 = nullptr;
  size_t bytes = 0;

  void alloc(bool needBlurTemps) {
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    if (needBlurTemps) {
      CUDA_CHECK(cudaMalloc((void**)&d_tmp, bytes));
      CUDA_CHECK(cudaMalloc((void**)&d_tmp2, bytes));
    }
  }

  void freeAll() {
    if (d_tmp)  CUDA_CHECK(cudaFree(d_tmp));
    if (d_tmp2) CUDA_CHECK(cudaFree(d_tmp2));
    if (d_in)   CUDA_CHECK(cudaFree(d_in));
    if (d_out)  CUDA_CHECK(cudaFree(d_out));
    d_in = d_out = d_tmp = d_tmp2 = nullptr;
  }

  ~DeviceBuffers() { freeAll(); }
};

static float runPipeline(const Args &args,
                         const DeviceBuffers &buf,
                         int w, int h,
                         dim3 grid, dim3 block,
                         int modeInt) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  if (args.blur) {
    // LUT -> tmp
    lutKernel<<<grid, block>>>(buf.d_in, buf.d_tmp, w, h, modeInt, args.intensity);
    CUDA_CHECK(cudaGetLastError());

    // Blur passes (ping-pong)
    const size_t sharedBytes =
        (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(uchar3);

    unsigned char *src = buf.d_tmp;
    unsigned char *dst = buf.d_tmp2;

    int passes = args.blurPasses < 1 ? 1 : args.blurPasses;

    for (int p = 0; p < passes; p++) {
      if (p == passes - 1) dst = buf.d_out;

      blur3x3Shared<<<grid, block, sharedBytes>>>(src, dst, w, h);
      CUDA_CHECK(cudaGetLastError());

      if (p != passes - 1) {
        unsigned char *t = src;
        src = dst;
        dst = t;
      }
    }
  } else {
    lutKernel<<<grid, block>>>(buf.d_in, buf.d_out, w, h, modeInt, args.intensity);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

static void convertOutputIfNeeded(const ImagePaths &p,
                                 const std::string &outExt,
                                 const std::string &finalOutPath) {
  if (outExt == "ppm") return;

  std::ostringstream cmd;
  cmd << p.magickCmd << " \"" << p.ppmOutPath << "\" \"" << finalOutPath << "\"";
  std::cout << "[Convert] " << cmd.str() << "\n";

  if (runCmd(cmd.str()) != 0 || !fileExists(finalOutPath)) {
    std::cerr << "Failed to convert output PPM to final output.\n";
    std::exit(1);
  }
}

static void cleanupTempsIfNeeded(const Args &args, const ImagePaths &p) {
  if (args.keepTemp) {
    std::cout << "Keeping temp files:\n"
              << "  " << p.tmpInPPM << "\n"
              << "  " << p.tmpOutPPM << "\n";
    return;
  }
  if (p.usedTmpIn)  std::remove(p.tmpInPPM.c_str());
  if (p.usedTmpOut) std::remove(p.tmpOutPPM.c_str());
}

static void printRunSummary(const Args &args,
                            int w, int h,
                            dim3 grid,
                            float htod_ms,
                            float kernel_ms,
                            float dtoh_ms,
                            size_t alloc_bytes,
                            size_t free_before,
                            size_t free_after_alloc,
                            size_t total_mem,
                            const std::string &outExt,
                            const ImagePaths &p) {
  const int tpb = args.bx * args.by;
  const float total_ms = htod_ms + kernel_ms + dtoh_ms;

  auto toMiB = [](size_t bytes) -> double {
    return (double)bytes / (1024.0 * 1024.0);
  };
  auto toGiB = [](size_t bytes) -> double {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
  };

  std::cout << "\n";
  std::cout << "CUDA Run Summary\n";
  std::cout << "Image\n";
  std::cout << "  Size: " << w << " x " << h
            << "  Pixels: " << (size_t)w * (size_t)h << "\n";

  std::cout << "Launch: block " << args.bx << "x" << args.by
          << " tpb " << tpb
          << " grid " << grid.x << "x" << grid.y
          << " blocks " << (size_t)grid.x * (size_t)grid.y << "\n";

  std::cout << "Config\n";
  std::cout << "  Mode: " << args.mode
            << "  Intensity: " << args.intensity << "\n";
  std::cout << "  Blur: " << (args.blur ? "ON" : "OFF");
  if (args.blur) std::cout << "  Passes: " << args.blurPasses;
  std::cout << "\n";

  std::cout << "Timing (ms): HtoD " << htod_ms
          << "  Kernel " << kernel_ms
          << "  DtoH " << dtoh_ms
          << "  Total " << total_ms << "\n";

  std::cout << "Memory\n";
  std::cout << "  Allocated (estimated): " << alloc_bytes
            << " bytes  (" << toMiB(alloc_bytes) << " MiB)\n";
  std::cout << "  GPU Total: " << toGiB(total_mem) << " GiB\n";
  std::cout << "  GPU Free Before Alloc: " << toGiB(free_before) << " GiB\n";
  std::cout << "  GPU Free After  Alloc: " << toGiB(free_after_alloc) << " GiB\n";

  std::cout << "Output\n";
  std::cout << "  " << (outExt == "ppm" ? p.ppmOutPath : args.outPath) << "\n";

  // Parse friendly single line for Python
  std::cout
      << "RESULT "
      << "w=" << w << " "
      << "h=" << h << " "
      << "bx=" << args.bx << " "
      << "by=" << args.by << " "
      << "tpb=" << tpb << " "
      << "gridx=" << grid.x << " "
      << "gridy=" << grid.y << " "
      << "mode=" << args.mode << " "
      << "blur=" << (args.blur ? 1 : 0) << " "
      << "passes=" << (args.blur ? args.blurPasses : 0) << " "
      << "htod_ms=" << htod_ms << " "
      << "kernel_ms=" << kernel_ms << " "
      << "dtoh_ms=" << dtoh_ms << " "
      << "total_ms=" << total_ms << " "
      << "alloc_bytes=" << alloc_bytes << " "
      << "free0=" << free_before << " "
      << "free1=" << free_after_alloc << " "
      << "total_mem=" << total_mem
      << "\n\n";

  std::cout << std::flush;
}


int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);

  ImagePaths p;
  p.outDir = "OutputImages";
  ensureDir(p.outDir);

  if (!hasDirComponent(args.outPath)) {
    args.outPath = joinPath(p.outDir, args.outPath);
  }

  if (!ensureMagick(p.magickCmd)) return 1;

  if (!fileExists(args.inPath)) {
    std::cerr << "Input not found: " << args.inPath << "\n";
    return 1;
  }

  uploadLUTsToConstant();

  const std::string outExt = getExtLower(args.outPath);
  p = preparePathsAndConvertInput(args, outExt, p);

  int w = 0, h = 0;
  std::vector<unsigned char> h_in;
  if (!readPPM_P6(p.ppmInPath, w, h, h_in)) {
    std::cerr << "Failed to read PPM: " << p.ppmInPath << "\n";
    return 1;
  }

  const size_t bytes = (size_t)w * (size_t)h * 3u;
  std::vector<unsigned char> h_out(bytes);

  // Memory snapshot before any cudaMalloc
  size_t free0 = 0, total0 = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

  DeviceBuffers buf;
  buf.bytes = bytes;
  buf.alloc(args.blur);

  // Memory snapshot after allocations (should be lower)
  size_t free1 = 0, total1 = 0;
  CUDA_CHECK(cudaMemGetInfo(&free1, &total1));

  // Estimated alloc size (exact for your buffers)
  const int numBuf = args.blur ? 4 : 2;
  const size_t alloc_bytes = (size_t)numBuf * bytes;

  // Create timing events for copies
  cudaEvent_t e0, e1, e2, e3;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventCreate(&e3));

  float htod_ms = 0.0f;
  float dtoh_ms = 0.0f;

  // HtoD
  CUDA_CHECK(cudaEventRecord(e0));
  CUDA_CHECK(cudaMemcpy(buf.d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&htod_ms, e0, e1));

  // Launch config
  dim3 block((unsigned)args.bx, (unsigned)args.by, 1);
  dim3 grid((unsigned)((w + args.bx - 1) / args.bx),
            (unsigned)((h + args.by - 1) / args.by),
            1);

  // Kernel pipeline timing
  const int modeInt = resolveModeInt(args.mode);
  const float kernel_ms = runPipeline(args, buf, w, h, grid, block, modeInt);

  // DtoH
  CUDA_CHECK(cudaEventRecord(e2));
  CUDA_CHECK(cudaMemcpy(h_out.data(), buf.d_out, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e3));
  CUDA_CHECK(cudaEventSynchronize(e3));
  CUDA_CHECK(cudaEventElapsedTime(&dtoh_ms, e2, e3));

  // Write output PPM
  if (!writePPM_P6(p.ppmOutPath, w, h, h_out.data())) {
    std::cerr << "Failed to write PPM: " << p.ppmOutPath << "\n";
    return 1;
  }

  convertOutputIfNeeded(p, outExt, args.outPath);

  printRunSummary(args, w, h, grid,
                  htod_ms, kernel_ms, dtoh_ms,
                  alloc_bytes, free0, free1, total0,
                  outExt, p);

  cleanupTempsIfNeeded(args, p);

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
  CUDA_CHECK(cudaEventDestroy(e3));

  return 0;
}