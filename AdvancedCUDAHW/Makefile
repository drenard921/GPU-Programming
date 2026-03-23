# Makefile for CUDA Dividend Portfolio Allocation Optimizer

NVCC        := nvcc
CXXFLAGS    := -O2 -std=c++17
LDLIBS      := -lcublas -lcusolver
SRC         := solver.cu

# Detect OS
ifeq ($(OS),Windows_NT)
    TARGET      := solver_app.exe
    RM          := -del /Q /F
    RUN_PREFIX  :=
    PYTHON      := python
    LDFLAGS     := -L"$(CUDA_PATH)/lib/x64"
    INCLUDES    := -I"$(CUDA_PATH)/include"
else
    TARGET      := solver_app
    RM          := rm -f
    RUN_PREFIX  := ./
    PYTHON      := python3
    LDFLAGS     :=
    INCLUDES    :=
endif

.PHONY: all run gui clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	$(RUN_PREFIX)$(TARGET) input.csv results.csv

gui: $(TARGET)
	$(PYTHON) gui.py

clean:
	$(RM) $(TARGET) 2>nul || true
	$(RM) results.csv 2>nul || true
	$(RM) results_python.csv 2>nul || true
	$(RM) normalized_import.csv 2>nul || true