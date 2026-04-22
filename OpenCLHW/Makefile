# ===== Compiler =====
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude
LDLIBS   := -lOpenCL

# ===== Directories =====
SRC_DIR  := src
BIN_DIR  := bin
OUT_DIR  := output
DATA_DIR := Data/genres_original

TARGET   := $(BIN_DIR)/spectrogram

# ===== Source Files =====
SRCS := \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/wav_reader.cpp \
	$(SRC_DIR)/spectrogram_cpu.cpp \
	$(SRC_DIR)/spectrogram_opencl.cpp \
	$(SRC_DIR)/opencl_utils.cpp \
	$(SRC_DIR)/image_writer.cpp

OBJS := $(SRCS:.cpp=.o)

# ===== Runtime Arguments =====
INPUT      ?= $(DATA_DIR)/blues/blues.00000.wav
BATCH_DIR  ?= $(DATA_DIR)/blues
MODE       ?= cpu
KERNEL     ?= local
WINDOW     ?= 1024
HOP        ?= 512
BINS       ?= 512

# ===== Targets =====
.PHONY: all build run run_cpu run_gpu run_gpu_naive run_gpu_local run_both \
        batch_cpu batch_gpu batch_gpu_naive batch_gpu_local batch_both \
        clean rebuild help dirs

all: build

build: dirs $(TARGET)

dirs:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OUT_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDLIBS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ===== Single-file modes =====

run: build
	./$(TARGET) "$(INPUT)" "$(MODE)" "$(KERNEL)" "$(WINDOW)" "$(HOP)" "$(BINS)"

run_cpu: build
	./$(TARGET) "$(INPUT)" cpu "$(WINDOW)" "$(HOP)" "$(BINS)"

run_gpu: build
	./$(TARGET) "$(INPUT)" gpu "$(KERNEL)" "$(WINDOW)" "$(HOP)" "$(BINS)"

run_gpu_naive: build
	./$(TARGET) "$(INPUT)" gpu naive "$(WINDOW)" "$(HOP)" "$(BINS)"

run_gpu_local: build
	./$(TARGET) "$(INPUT)" gpu local "$(WINDOW)" "$(HOP)" "$(BINS)"

run_both: build
	./$(TARGET) "$(INPUT)" both "$(KERNEL)" "$(WINDOW)" "$(HOP)" "$(BINS)"

# ===== Directory batch modes =====
# These assume main.cpp treats directory input as batch mode.

batch_cpu: build
	./$(TARGET) "$(BATCH_DIR)" cpu "$(WINDOW)" "$(HOP)" "$(BINS)"

batch_gpu: build
	./$(TARGET) "$(BATCH_DIR)" gpu "$(KERNEL)" "$(WINDOW)" "$(HOP)" "$(BINS)"

batch_gpu_naive: build
	./$(TARGET) "$(BATCH_DIR)" gpu naive "$(WINDOW)" "$(HOP)" "$(BINS)"

batch_gpu_local: build
	./$(TARGET) "$(BATCH_DIR)" gpu local "$(WINDOW)" "$(HOP)" "$(BINS)"

batch_both: build
	./$(TARGET) "$(BATCH_DIR)" both "$(KERNEL)" "$(WINDOW)" "$(HOP)" "$(BINS)"

clean:
	rm -rf $(BIN_DIR) $(OUT_DIR) $(OBJS)

rebuild: clean build

help:
	@echo "Targets:"
	@echo "  make build"
	@echo "  make run"
	@echo "  make run_cpu"
	@echo "  make run_gpu"
	@echo "  make run_gpu_naive"
	@echo "  make run_gpu_local"
	@echo "  make run_both"
	@echo "  make batch_cpu"
	@echo "  make batch_gpu"
	@echo "  make batch_gpu_naive"
	@echo "  make batch_gpu_local"
	@echo "  make batch_both"
	@echo "  make clean"
	@echo ""
	@echo "Configurable variables:"
	@echo "  INPUT=<wav file path>"
	@echo "  BATCH_DIR=<directory of wav files>"
	@echo "  MODE=cpu|gpu|both"
	@echo "  KERNEL=naive|local"
	@echo "  WINDOW=<window size>"
	@echo "  HOP=<hop size>"
	@echo "  BINS=<num bins>"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run MODE=gpu KERNEL=local"
	@echo "  make run MODE=gpu KERNEL=naive"
	@echo "  make run_gpu"
	@echo "  make run_gpu_naive"
	@echo "  make run_gpu_local"
	@echo "  make run_both KERNEL=local"
	@echo "  make run INPUT=Data/genres_original/jazz/jazz.00000.wav MODE=both KERNEL=naive"
	@echo "  make batch_gpu"
	@echo "  make batch_gpu_naive"
	@echo "  make batch_gpu_local"
	@echo "  make batch_both BATCH_DIR=Data/genres_original/blues KERNEL=local"
	@echo "  make run WINDOW=2048 HOP=256 BINS=1024 MODE=gpu KERNEL=local"