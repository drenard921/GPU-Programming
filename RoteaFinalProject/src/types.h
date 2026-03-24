#pragma once
#include <string>
#include <array>

enum class LineID { A=0, B, C, D, E, F, G, H };

enum class PhaseType {
    Load,
    Wash,
    Concentrate,
    Harvest
};

struct Particle {
    float x, y, z;
    int type;
};

struct Bag {
    std::string name;
    float volume_ml;
    float pressure_kpa;
};

struct Line {
    LineID id;
    int sourceBag;   // index into bags
    int targetBag;   // index into bags
    float flow_ml_min;
    bool active;
};

struct Chamber {
    float retained_cells;
    float suspended_cells;
};

struct Step {
    std::string name;
    PhaseType phase;

    float duration_s;
    float g_force;

    std::array<float, 8> flow_ml_min; // A–H
};