TARGET_NAME := rotea_sim
CUDA_APP_NAME := rotea_sim_cuda
CUDA_TARGET_NAME := cuda_frames

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

CUDA_HOME ?= /usr/local/cuda

PARTICLES ?= 5000
SECONDS ?= 5
CUDA_FRAMES ?= 300
FRAME_FILE ?= frames.bin

CXX := g++
NVCC := nvcc

APP_CPP_SOURCES := \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/camera.cpp \
	$(SRC_DIR)/renderer.cpp \
	$(SRC_DIR)/shader.cpp \
	$(SRC_DIR)/sim_cpu.cpp

CUDA_SOURCE := $(SRC_DIR)/simulation.cu

APP_OBJ_DIR := $(BUILD_DIR)/app
APP_CUDA_OBJ_DIR := $(BUILD_DIR)/app_cuda
CUDA_OBJ_DIR := $(BUILD_DIR)/cuda

APP_TARGET := $(BIN_DIR)/$(TARGET_NAME)
CUDA_APP_TARGET := $(BIN_DIR)/$(CUDA_APP_NAME)
CUDA_TARGET := $(BIN_DIR)/$(CUDA_TARGET_NAME)

APP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(APP_OBJ_DIR)/%.o,$(APP_CPP_SOURCES))
CUDA_APP_CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(APP_CUDA_OBJ_DIR)/%.o,$(APP_CPP_SOURCES))
CUDA_APP_CU_OBJECT := $(APP_CUDA_OBJ_DIR)/simulation.o

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -I$(CUDA_HOME)/include
NVCCFLAGS := -std=c++17 -O2 -I$(CUDA_HOME)/include -DUSE_CUDA

APP_LDFLAGS := -lglfw -lGL -ldl
CUDA_LDFLAGS := -L$(CUDA_HOME)/lib64 \
	-Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
	-lcudart

.PHONY: all clean app app-cuda cuda-frames test-frames replay cuda-replay run-cpu run-cuda info

all: app app-cuda cuda-frames

# -----------------------------
# CPU app
# -----------------------------

app: $(APP_TARGET)

$(APP_TARGET): $(APP_OBJECTS) | $(BIN_DIR)
	$(CXX) $(APP_OBJECTS) -o $@ $(APP_LDFLAGS)

$(APP_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(APP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -----------------------------
# CUDA OpenGL app
# -----------------------------

app-cuda: $(CUDA_APP_TARGET)

$(CUDA_APP_TARGET): $(CUDA_APP_CPP_OBJECTS) $(CUDA_APP_CU_OBJECT) | $(BIN_DIR)
	$(NVCC) $(CUDA_APP_CPP_OBJECTS) $(CUDA_APP_CU_OBJECT) -o $@ $(APP_LDFLAGS) $(CUDA_LDFLAGS)

$(APP_CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(APP_CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -c $< -o $@

$(CUDA_APP_CU_OBJECT): $(CUDA_SOURCE) | $(APP_CUDA_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# -----------------------------
# Standalone CUDA frame generator
# -----------------------------

cuda-frames: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SOURCE) | $(CUDA_OBJ_DIR) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DCUDA_FRAMES_STANDALONE $< -o $@ $(CUDA_LDFLAGS)

# -----------------------------
# Run targets
# -----------------------------

run-cpu: app
	./$(APP_TARGET) --mode cpu --particles $(PARTICLES) --duration $(SECONDS)

run-cuda: app-cuda
	./$(CUDA_APP_TARGET) --mode cuda --particles $(PARTICLES) --duration $(SECONDS)

test-frames: cuda-frames
	./$(CUDA_TARGET) --particles 1 --frames 3 --output frames_one.bin
	./$(CUDA_TARGET) --particles 2 --frames 3 --output frames_two.bin
	./$(CUDA_TARGET) --particles $(PARTICLES) --frames $(CUDA_FRAMES) --output $(FRAME_FILE)
	ls -lh frames_one.bin frames_two.bin $(FRAME_FILE)

replay: app
	./$(APP_TARGET) --mode replay --replay $(FRAME_FILE) --duration $(SECONDS)

cuda-replay: app-cuda cuda-frames
	./$(CUDA_TARGET) --particles $(PARTICLES) --frames $(CUDA_FRAMES) --output $(FRAME_FILE)
	./$(CUDA_APP_TARGET) --mode replay --replay $(FRAME_FILE) --duration $(SECONDS)

# -----------------------------
# Directories
# -----------------------------

$(APP_OBJ_DIR):
	mkdir -p $@

$(APP_CUDA_OBJ_DIR):
	mkdir -p $@

$(CUDA_OBJ_DIR):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

# -----------------------------
# Cleanup / diagnostics
# -----------------------------

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) frames.bin frames_one.bin frames_two.bin frames_large.bin

info:
	@echo "CXX              = $(CXX)"
	@echo "NVCC             = $(NVCC)"
	@echo "CUDA_HOME        = $(CUDA_HOME)"
	@echo "PARTICLES        = $(PARTICLES)"
	@echo "SECONDS          = $(SECONDS)"
	@echo "CUDA_FRAMES      = $(CUDA_FRAMES)"
	@echo "FRAME_FILE       = $(FRAME_FILE)"
	@echo "CXXFLAGS         = $(CXXFLAGS)"
	@echo "NVCCFLAGS        = $(NVCCFLAGS)"
	@echo "APP_LDFLAGS      = $(APP_LDFLAGS)"
	@echo "CUDA_LDFLAGS     = $(CUDA_LDFLAGS)"