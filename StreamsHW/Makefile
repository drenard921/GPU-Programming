CXX      := g++
NVCC     := /usr/local/cuda/bin/nvcc

CPU_TARGET := streams_hw_libretro
CPU_SRC    := streams_hw.cpp

CUDA_TARGET := streams_hw_cuda
CUDA_SRC    := streams_hw.cu

SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS   := $(shell sdl2-config --libs)
RETRO_INC  := -I./mgba/src/platform/libretro

# Debug toggle
ifeq ($(DEBUG),1)
  CXXFLAGS  := -O0 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter -fno-omit-frame-pointer
  NVCCFLAGS := -O0 -g -G -std=c++17
else
  CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter
  NVCCFLAGS := -O2 -std=c++17
endif

LDFLAGS := $(SDL_LIBS) -ldl

all: $(CPU_TARGET) $(CUDA_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(SDL_CFLAGS) $(RETRO_INC) $(LDFLAGS)

$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(SDL_CFLAGS) $(RETRO_INC) $(LDFLAGS)

clean:
	rm -f $(CPU_TARGET) $(CUDA_TARGET)

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) --scale 3

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET) --scale 3 --threads 256 --frames 600

.PHONY: all clean run_cpu run_cuda