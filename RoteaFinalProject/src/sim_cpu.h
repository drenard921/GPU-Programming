#pragma once

#include <vector>
#include <GLFW/glfw3.h>

#include "renderer.h"
#include "types.h"

std::vector<Particle> initParticles(int nFluid, int nCells, float radius, float height);

void updateSystem(
    std::vector<Bag>& bags,
    std::vector<Line>& lines,
    Chamber& chamber,
    const Step& step,
    float dt
);

void updateParticles(std::vector<Particle>& particles, float dt, float gForce, float flow);

void processInput(GLFWwindow* window, float& gForce, float& flow);