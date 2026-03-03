CXX      := g++
TARGET   := streams_hw_libretro
SRC      := streams_hw.cpp

SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS   := $(shell sdl2-config --libs)
RETRO_INC  := -I./mgba/src/platform/libretro

# Debug toggle
ifeq ($(DEBUG),1)
  CXXFLAGS := -O0 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter -fno-omit-frame-pointer
else
  CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter
endif

LDFLAGS := $(SDL_LIBS) -ldl

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(SDL_CFLAGS) $(RETRO_INC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) --rom "./GBAROMS/Pokemon - FireRed Version (USA, Europe).gba" \
	             --core "./mgba/build/mgba_libretro.so" \
	             --scale 3

.PHONY: all clean run