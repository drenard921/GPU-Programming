TARGET := bin/rotea_sim

CXX := g++
NVCC := nvcc

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

CUDA_HOME := /usr/local/cuda

CPP_SOURCES := $(SRC_DIR)/main.cpp \
               $(SRC_DIR)/camera.cpp \
               $(SRC_DIR)/renderer.cpp \
               $(SRC_DIR)/shader.cpp \
               $(SRC_DIR)/sim_cpu.cpp

CU_SOURCES := $(SRC_DIR)/simulation.cu

CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cuo,$(CU_SOURCES))

OBJECTS := $(CPP_OBJECTS) $(CU_OBJECTS)

CXXFLAGS := -std=c++17 -O2 -I$(CUDA_HOME)/include -I/usr/include
NVCCFLAGS := -std=c++17 -O2 -I$(CUDA_HOME)/include -I/usr/include

LDFLAGS := -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lGL -lglfw -ldl -lcudart

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

.PHONY: all clean run dirs