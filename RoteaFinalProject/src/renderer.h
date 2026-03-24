#pragma once

#include <vector>
#include "types.h"

void setupProjection(int width, int height);

void drawNestedConeChamber(
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip,
    float height
);

void drawParticles(const std::vector<Particle>& particles);