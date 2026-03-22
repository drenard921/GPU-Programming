# Makefile for CUDA Dividend Portfolio Allocation Optimizer

NVCC        := nvcc
CXXFLAGS    := -O2 -std=c++17
LDLIBS      := -lcublas -lcusolver

TARGET      := solver_app
SRC         := solver.cu

.PHONY: all run gui clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET) input.csv results.csv

gui: $(TARGET)
	python3 gui.py

clean:
	rm -f $(TARGET) results.csv