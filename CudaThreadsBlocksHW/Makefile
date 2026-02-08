# ==============================
# Makefile for CUDA Assignment
# ==============================

# CUDA compiler
NVCC = nvcc

# Executable name (REQUIRED by assignment)
TARGET = assignment.exe

# Source file (REQUIRED location/name)
SRC = assignment.cu

# Compiler flags
NVCC_FLAGS = -O2 -std=c++14

# Default rule (runs when you type `make`)
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean up build artifacts
clean:
	rm -f $(TARGET)
