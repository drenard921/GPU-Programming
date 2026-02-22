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
// Notes:
//   - This Part 1 program focuses on I/O, CLI, variable block sizing, and timing.
//   - Later parts will add constant/shared memory features (LUT + blur).

#include <cuda_runtime.h>

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Identity kernel: copy input pixels to output.
// This is the “pipeline proof” kernel.
__global__ void identityKernel(const unsigned char *in, unsigned char *out, int width, int height) {
  int x = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  int y = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;
  // Registers: idx and channel values are held in registers
  unsigned char r = in[idx + 0];
  unsigned char g = in[idx + 1];
  unsigned char b = in[idx + 2];

  out[idx + 0] = r;
  out[idx + 1] = g;
  out[idx + 2] = b;
}

struct Args {
  std::string inPath;
  std::string outPath;
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
    "  ./cuda_filter --in input.ppm --out output.ppm --bx 32 --by 8\n";
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

  const std::string inExt = getExtLower(args.inPath);
  const std::string outExt = getExtLower(args.outPath);
  const std::string base = getBaseName(args.inPath);

  // Temp PPMs always go to OutputImages/
  const std::string tmpInPPM  = joinPath(outDir, base + "_cuda_in.ppm");
  const std::string tmpOutPPM = joinPath(outDir, base + "_cuda_out.ppm");

  bool usedTmpIn = false;
  bool usedTmpOut = false;

  std::string ppmInPath = args.inPath;
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
  CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

  // Copy host->device
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Launch config (variable blocks/threads)
  dim3 block((unsigned)args.bx, (unsigned)args.by, 1);
  dim3 grid((unsigned)((w + args.bx - 1) / args.bx),
            (unsigned)((h + args.by - 1) / args.by),
            1);

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  identityKernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK(cudaGetLastError());
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
  std::cout << "Kernel time (identity): " << ms << " ms\n";
  std::cout << "Output: " << (outExt == "ppm" ? ppmOutPath : args.outPath) << "\n";

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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