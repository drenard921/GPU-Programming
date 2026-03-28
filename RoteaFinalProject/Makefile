TARGET := bin/rotea_sim

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
SDKROOT := $(shell xcrun --sdk macosx --show-sdk-path)
CXXFLAGS += -isysroot $(SDKROOT)

UNAME_S := $(shell uname -s)

# Default source files
CPP_SOURCES := $(SRC_DIR)/main.cpp \
               $(SRC_DIR)/camera.cpp \
               $(SRC_DIR)/renderer.cpp \
               $(SRC_DIR)/shader.cpp \
               $(SRC_DIR)/sim_cpu.cpp

CU_SOURCES := $(SRC_DIR)/simulation.cu

# Defaults
CXXFLAGS := -std=c++17 -O2
LDFLAGS :=
CUDA_ENABLED := 0

# -----------------------------
# OS-specific configuration
# -----------------------------
ifeq ($(UNAME_S),Darwin)
    CXX := clang++
    NVCC :=

    # Homebrew paths (Apple Silicon + Intel Mac fallback)
    CXXFLAGS += -I/opt/homebrew/include -I/usr/local/include
    LDFLAGS += -L/opt/homebrew/lib -L/usr/local/lib -lglfw -framework OpenGL

    # GLEW optional; uncomment if your project uses it
    # LDFLAGS += -lGLEW

    # CUDA is not supported on modern macOS, so force CPU-only build
    CUDA_ENABLED := 0
endif

ifeq ($(UNAME_S),Linux)
    CXX := g++
    NVCC := nvcc

    CUDA_HOME ?= /usr/local/cuda

    CXXFLAGS += -I$(CUDA_HOME)/include
    NVCCFLAGS := -std=c++17 -O2 -I$(CUDA_HOME)/include

    LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 \
               -lglfw -lGL -ldl

    # Enable CUDA only if nvcc exists
    ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
        CUDA_ENABLED := 1
        LDFLAGS += -lcudart
    endif
endif

# -----------------------------
# Source/object selection
# -----------------------------
ifeq ($(CUDA_ENABLED),1)
    CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(filter-out $(SRC_DIR)/sim_cpu.cpp,$(CPP_SOURCES)))
    CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cuo,$(CU_SOURCES))
    OBJECTS := $(CPP_OBJECTS) $(CU_OBJECTS)
else
    CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
    CU_OBJECTS :=
    OBJECTS := $(CPP_OBJECTS)
endif

all: dirs $(TARGET)

dirs:
	mkdir -p $(OBJ_DIR)
	mkdir -p $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cuo: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: all
	./$(TARGET)

info:
	@echo "OS            = $(UNAME_S)"
	@echo "CXX           = $(CXX)"
	@echo "CUDA_ENABLED  = $(CUDA_ENABLED)"
	@echo "CPP_SOURCES   = $(CPP_SOURCES)"
	@echo "CU_SOURCES    = $(CU_SOURCES)"
	@echo "OBJECTS       = $(OBJECTS)"

.PHONY: all clean run dirs info