// Dylan Renard
// EN.605.617 Introduction to GPU Programming (JHU)
// Professor Chance Pascale
// February 21st 2026
//
// CUDA Memory Assignment — 1D .cube LUT Filter Applier + Optional Shared-Memory Gaussian Blur
//
// Summary:
// This program demonstrates CUDA memory hierarchy usage by applying a selectable color
// “coating” (1D LUT) to an input image, with an optional 3x3 Gaussian blur implemented
// using shared memory tiling. The tool supports standard image inputs (e.g., JPG/PNG)
// via ImageMagick conversion to an intermediate PPM (P6) format, then runs CUDA kernels
// on raw RGB bytes, and converts the result back to the requested output format

// Output location:
//   - All intermediate PPMs are written to ./OutputImages/
//   - If --out is provided as a filename (no directory), the final output is also saved to ./OutputImages/
//
// Command-line usage:
//   ./cuda_filter --in <path> --out <path> [--bx N] [--by N] [--keep-temp 0|1]
//
// Examples:
//   ./cuda_filter --in canyon.jpg --out canyon_out.jpg --bx 16 --by 16
//   ./cuda_filter --in input.ppm --out output.ppm --bx 32 --by 8
//


#include <cuda_runtime.h>

#include <cctype>
#include <cmath>
#include <cstdio>
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

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmR, warmR, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmG, warmG, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutWarmB, warmB, 256));

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueR, blueR, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueG, blueG, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBlueB, blueB, 256));

  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWR, bwR, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWG, bwG, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lutBWB, bwB, 256));
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

__global__ void blur3x3Shared(const unsigned char* in, unsigned char* out, int width, int height) {
  int tx = (int)threadIdx.x;
  int ty = (int)threadIdx.y;
  int x  = (int)blockIdx.x * (int)blockDim.x + tx;
  int y  = (int)blockIdx.y * (int)blockDim.y + ty;

  int tileW = (int)blockDim.x + 2;
  int tileH = (int)blockDim.y + 2;

  extern __shared__ uchar3 sTile[]; // size = tileW * tileH

  // Write my center pixel into shared tile at (+1,+1)
  if (x < width && y < height) {
    sTile[(ty + 1) * tileW + (tx + 1)] = loadPixelClampedRGB(in, width, height, x, y);
  }

  // Halo loads (only some threads do this)
  if (tx == 0) {
    sTile[(ty + 1) * tileW + 0] = loadPixelClampedRGB(in, width, height, x - 1, y);
  }
  if (tx == (int)blockDim.x - 1) {
    sTile[(ty + 1) * tileW + (tx + 2)] = loadPixelClampedRGB(in, width, height, x + 1, y);
  }
  if (ty == 0) {
    sTile[0 * tileW + (tx + 1)] = loadPixelClampedRGB(in, width, height, x, y - 1);
  }
  if (ty == (int)blockDim.y - 1) {
    sTile[(ty + 2) * tileW + (tx + 1)] = loadPixelClampedRGB(in, width, height, x, y + 1);
  }

  // Corners
  if (tx == 0 && ty == 0) {
    sTile[0 * tileW + 0] = loadPixelClampedRGB(in, width, height, x - 1, y - 1);
  }
  if (tx == (int)blockDim.x - 1 && ty == 0) {
    sTile[0 * tileW + (tx + 2)] = loadPixelClampedRGB(in, width, height, x + 1, y - 1);
  }
  if (tx == 0 && ty == (int)blockDim.y - 1) {
    sTile[(ty + 2) * tileW + 0] = loadPixelClampedRGB(in, width, height, x - 1, y + 1);
  }
  if (tx == (int)blockDim.x - 1 && ty == (int)blockDim.y - 1) {
    sTile[(ty + 2) * tileW + (tx + 2)] = loadPixelClampedRGB(in,  width, height, x + 1, y + 1);
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  // Convolution in shared memory
  int sx = tx + 1;
  int sy = ty + 1;

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

  return a;
}

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);

  // Ensure output directory exists
  const std::string outDir = "OutputImages";
  ensureDir(outDir);

  // If user gave only a filename for --out, place it into OutputImages/
  if (!hasDirComponent(args.outPath)) {
    args.outPath = joinPath(outDir, args.outPath);
  }

  std::string magickCmd;
  if (!haveMagickCommand(magickCmd)) {
    std::cerr << "Error: ImageMagick not found. Install it or provide .ppm input/output.\n"
              << "Try: sudo apt install -y imagemagick\n";
    return 1;
  }

  if (!fileExists(args.inPath)) {
    std::cerr << "Input not found: " << args.inPath << "\n";
    return 1;
  }

  // Upload LUTs to constant memory once per run
  uploadLUTsToConstant();

  const std::string inExt  = getExtLower(args.inPath);
  const std::string outExt = getExtLower(args.outPath);
  const std::string base   = getBaseName(args.inPath);

  // Temp PPMs always go to OutputImages/
  const std::string tmpInPPM  = joinPath(outDir, base + "_cuda_in.ppm");
  const std::string tmpOutPPM = joinPath(outDir, base + "_cuda_out.ppm");

  bool usedTmpIn = false;
  bool usedTmpOut = false;

  std::string ppmInPath  = args.inPath;
  std::string ppmOutPath = args.outPath;

  // Convert input to PPM if needed
  if (inExt != "ppm") {
    std::ostringstream cmd;
    cmd << magickCmd << " \"" << args.inPath
        << "\" -alpha off -depth 8 \"" << tmpInPPM << "\"";
    std::cout << "[Convert] " << cmd.str() << "\n";
    if (runCmd(cmd.str()) != 0 || !fileExists(tmpInPPM)) {
      std::cerr << "Failed to convert input to PPM.\n";
      return 1;
    }
    ppmInPath = tmpInPPM;
    usedTmpIn = true;
  }

  // Always write a PPM first if output isn't ppm
  if (outExt != "ppm") {
    ppmOutPath = tmpOutPPM;
    usedTmpOut = true;
  }

  // Read PPM
  int w = 0, h = 0;
  std::vector<unsigned char> h_in;
  if (!readPPM_P6(ppmInPath, w, h, h_in)) {
    std::cerr << "Failed to read PPM: " << ppmInPath << "\n";
    return 1;
  }

  const size_t bytes = (size_t)w * (size_t)h * 3u;
  std::vector<unsigned char> h_out(bytes);

  // Device alloc (global memory arrays)
  unsigned char *d_in = nullptr, *d_out = nullptr;
  unsigned char *d_tmp = nullptr, *d_tmp2 = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

  if (args.blur) {
    CUDA_CHECK(cudaMalloc((void**)&d_tmp, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_tmp2, bytes));
  }

  // Copy host->device
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Launch config (variable blocks/threads)
  dim3 block((unsigned)args.bx, (unsigned)args.by, 1);
  dim3 grid((unsigned)((w + args.bx - 1) / args.bx),
            (unsigned)((h + args.by - 1) / args.by),
            1);

  // Resolve mode
  int modeInt = (args.mode == "warm") ? (int)Mode::WARM :
                (args.mode == "blue") ? (int)Mode::BLUE :
                                        (int)Mode::BW;

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  if (args.blur) {
    // 1) LUT into d_tmp
    lutKernel<<<grid, block>>>(d_in, d_tmp, w, h, modeInt, args.intensity);
    CUDA_CHECK(cudaGetLastError());

    // 2) Multi-pass shared-memory blur (ping-pong)
    size_t sharedBytes = (size_t)(args.bx + 2) * (size_t)(args.by + 2) * sizeof(uchar3);

    unsigned char *src = d_tmp;
    unsigned char *dst = d_tmp2;

    int passes = args.blurPasses;
    if (passes < 1) passes = 1;

    for (int p = 0; p < passes; p++) {
      // Final pass writes to d_out
      if (p == passes - 1) dst = d_out;

      blur3x3Shared<<<grid, block, sharedBytes>>>(src, dst, w, h);
      CUDA_CHECK(cudaGetLastError());

      // Swap for next pass (unless we just wrote to d_out)
      if (p != passes - 1) {
        unsigned char *tmp = src;
        src = dst;
        dst = tmp;
      }
    }
  } else {
    // LUT directly to output
    lutKernel<<<grid, block>>>(d_in, d_out, w, h, modeInt, args.intensity);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy device->host
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // Write output PPM
  if (!writePPM_P6(ppmOutPath, w, h, h_out.data())) {
    std::cerr << "Failed to write PPM: " << ppmOutPath << "\n";
    return 1;
  }

  // Convert output PPM to desired format if needed
  if (outExt != "ppm") {
    std::ostringstream cmd;
    cmd << magickCmd << " \"" << ppmOutPath << "\" \"" << args.outPath << "\"";
    std::cout << "[Convert] " << cmd.str() << "\n";
    if (runCmd(cmd.str()) != 0 || !fileExists(args.outPath)) {
      std::cerr << "Failed to convert output PPM to final output.\n";
      return 1;
    }
  }

  // Print required runtime info for screenshots
  std::cout << "Image: " << w << "x" << h << "\n";
  std::cout << "Block: " << args.bx << "x" << args.by
            << "  Grid: " << grid.x << "x" << grid.y << "\n";
  std::cout << "Mode: " << args.mode << "  Intensity: " << args.intensity << "\n";
  std::cout << "Blur: " << (args.blur ? "ON" : "OFF");
  if (args.blur) std::cout << "  Passes: " << args.blurPasses;
  std::cout << "\n";
  std::cout << "Kernel time (LUT" << (args.blur ? "+Blur" : "") << "): " << ms << " ms\n";
  std::cout << "Output: " << (outExt == "ppm" ? ppmOutPath : args.outPath) << "\n";

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (d_tmp)  CUDA_CHECK(cudaFree(d_tmp));
  if (d_tmp2) CUDA_CHECK(cudaFree(d_tmp2));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (!args.keepTemp) {
    if (usedTmpIn)  std::remove(tmpInPPM.c_str());
    if (usedTmpOut) std::remove(tmpOutPPM.c_str());
  } else {
    std::cout << "Keeping temp files:\n"
              << "  " << tmpInPPM << "\n"
              << "  " << tmpOutPPM << "\n";
  }

  return 0;
}