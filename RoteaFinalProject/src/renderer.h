#pragma once

#include <vector>
#include "types.h"

void setupProjection(int width, int height);
void drawConeChamber(float baseRadius, float tipRadius, float height);
void drawParticles(const std::vector<Particle>& particles);
void drawBag2D(float x, float y, float w, float h, const char* label, float fillFrac);
void drawTubing2D(float x1, float y1, float x2, float y2, bool active);
void drawKitOverlay(const std::vector<Bag>& bags, const std::vector<Line>& lines);