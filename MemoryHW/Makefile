# Makefile
NVCC := nvcc
TARGET := cuda_filter
SRC := assignment.cu

CXXFLAGS := -O3 -std=c++17
NVCCFLAGS := $(CXXFLAGS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean