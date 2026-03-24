#pragma once

#include <vector>
#include <GLFW/glfw3.h>
#include "types.h"

std::vector<Particle> initParticles(
    int nFluid,
    int nCells,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip
);

void updateSystem(
    std::vector<Bag>& bags,
    std::vector<Line>& lines,
    Chamber& chamber,
    const Step& step,
    float dt
);

void updateParticles(
    std::vector<Particle>& particles,
    float dt,
    float gForce,
    float flow,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip
);

void processInput(GLFWwindow* window, float& gForce, float& flow);