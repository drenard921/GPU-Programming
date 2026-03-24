#include "renderer.h"

#include <GL/gl.h>
#include <cmath>
#include <vector>
#include <cstring>

static void beginOrtho2D(int width, int height) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
}

static void endOrtho2D() {
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

void setupProjection(int width, int height) {
    float aspect = static_cast<float>(width) / static_cast<float>(height == 0 ? 1 : height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float nearp = 0.1f;
    float farp  = 100.0f;
    float fov   = 60.0f;
    float top   = std::tan(fov * 3.14159265f / 360.0f) * nearp;
    float right = top * aspect;

    glFrustum(-right, right, -top, top, nearp, farp);
    glMatrixMode(GL_MODELVIEW);
}

void drawConeChamber(float baseRadius, float tipRadius, float height) {
    glColor4f(0.8f, 0.8f, 0.8f, 0.35f);

    const int rings = 14;
    const int segments = 64;

    for (int j = 0; j <= rings; ++j) {
        float tY = static_cast<float>(j) / static_cast<float>(rings);
        float y = -height * 0.5f + height * tY;
        float r = baseRadius + (tipRadius - baseRadius) * tY;

        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float t = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(segments);
            float x = r * std::cos(t);
            float z = r * std::sin(t);
            glVertex3f(x, y, z);
        }
        glEnd();
    }

    glBegin(GL_LINES);
    for (int i = 0; i < 12; ++i) {
        float t = 2.0f * 3.14159265f * static_cast<float>(i) / 12.0f;

        float xb = baseRadius * std::cos(t);
        float zb = baseRadius * std::sin(t);

        float xt = tipRadius * std::cos(t);
        float zt = tipRadius * std::sin(t);

        glVertex3f(xb, -height * 0.5f, zb);
        glVertex3f(xt,  height * 0.5f, zt);
    }
    glEnd();
}

void drawParticles(const std::vector<Particle>& particles) {
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 0) {
            glColor4f(0.2f, 0.65f, 1.0f, 0.45f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();

    glPointSize(4.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 1) {
            glColor4f(0.95f, 0.8f, 0.2f, 1.0f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();
}

void drawBag2D(float x, float y, float w, float h, const char* label, float fillFrac) {
    fillFrac = std::max(0.0f, std::min(1.0f, fillFrac));

    glColor4f(0.15f, 0.15f, 0.18f, 0.9f);
    glBegin(GL_QUADS);
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
    glEnd();

    glColor4f(0.2f, 0.55f, 0.95f, 0.65f);
    float fillH = h * fillFrac;
    glBegin(GL_QUADS);
    glVertex2f(x + 4,     y + h - fillH - 4);
    glVertex2f(x + w - 4, y + h - fillH - 4);
    glVertex2f(x + w - 4, y + h - 4);
    glVertex2f(x + 4,     y + h - 4);
    glEnd();

    glColor4f(0.9f, 0.9f, 0.95f, 1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
    glEnd();

    // tiny top hanger/tube stub
    glBegin(GL_LINES);
    glVertex2f(x + w * 0.5f, y);
    glVertex2f(x + w * 0.5f, y - 18.0f);
    glEnd();

    // label placeholder bar (real text later)
    glColor4f(0.85f, 0.85f, 0.2f, 1.0f);
    float labelW = 8.0f * std::strlen(label);
    glBegin(GL_LINES);
    glVertex2f(x + 5.0f, y + h + 10.0f);
    glVertex2f(x + 5.0f + labelW, y + h + 10.0f);
    glEnd();
}

void drawTubing2D(float x1, float y1, float x2, float y2, bool active) {
    if (active) glColor4f(0.1f, 0.95f, 0.4f, 1.0f);
    else        glColor4f(0.5f, 0.5f, 0.55f, 0.65f);

    glLineWidth(active ? 2.5f : 1.25f);
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
    glLineWidth(1.0f);
}

void drawKitOverlay(const std::vector<Bag>& bags, const std::vector<Line>& lines) {
    // assumes viewport already set by caller
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int width = viewport[2];
    int height = viewport[3];

    beginOrtho2D(width, height);

    const float bagW = 70.0f;
    const float bagH = 95.0f;

    // left side bags
    const float leftX = 35.0f;
    const float rightX = width - 35.0f - bagW;

    const float yA = 110.0f;
    const float yB = 240.0f;
    const float yG = 370.0f;
    const float yE = 500.0f;

    // right side bags
    const float yC = 110.0f;
    const float yD = 240.0f;
    const float yH = 370.0f;
    const float yF = 500.0f;

    // chamber screen anchor
    const float cx = width * 0.5f;
    const float cy = height * 0.52f;

    // bubble trap
    glColor4f(0.9f, 0.7f, 0.2f, 1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(cx - 25.0f, cy - 220.0f);
    glVertex2f(cx + 25.0f, cy - 220.0f);
    glVertex2f(cx + 20.0f, cy - 170.0f);
    glVertex2f(cx - 20.0f, cy - 170.0f);
    glEnd();

    // chamber connector
    glBegin(GL_LINES);
    glVertex2f(cx, cy - 170.0f);
    glVertex2f(cx, cy - 70.0f);
    glEnd();

    // bag fill fractions, safe defaults
    auto fillFrac = [&](int idx) -> float {
        if (idx < 0 || idx >= (int)bags.size()) return 0.0f;
        float denom = 500.0f;
        return bags[idx].volume_ml / denom;
    };

    // draw a few core bags from actual current sim indices
    // 0 Sample, 1 Buffer, 2 Product, 3 Waste
    drawBag2D(leftX,  yA, bagW, bagH, "Sample",  fillFrac(0));
    drawBag2D(leftX,  yB, bagW, bagH, "Buffer",  fillFrac(1));
    drawBag2D(rightX, yC, bagW, bagH, "Product", fillFrac(2));
    drawBag2D(rightX, yD, bagW, bagH, "Waste",   fillFrac(3));

    // optional placeholders for remaining kit lines
    drawBag2D(leftX,  yG, bagW, bagH, "G", 0.0f);
    drawBag2D(leftX,  yE, bagW, bagH, "E", 0.0f);
    drawBag2D(rightX, yH, bagW, bagH, "H", 0.0f);
    drawBag2D(rightX, yF, bagW, bagH, "F", 0.0f);

    // tubing endpoints from bags toward chamber/bubble trap
    auto bagPortXLeft  = leftX + bagW * 0.5f;
    auto bagPortXRight = rightX + bagW * 0.5f;

    auto activeLine = [&](int idx) -> bool {
        return idx >= 0 && idx < (int)lines.size() && lines[idx].active;
    };

    drawTubing2D(bagPortXLeft,  yA - 18.0f, cx - 40.0f, cy - 190.0f, activeLine(0)); // A
    drawTubing2D(bagPortXLeft,  yB - 18.0f, cx - 20.0f, cy - 150.0f, activeLine(1)); // B
    drawTubing2D(cx + 20.0f,    cy - 20.0f, bagPortXRight, yC - 18.0f, activeLine(2)); // C
    drawTubing2D(cx + 40.0f,    cy + 20.0f, bagPortXRight, yD - 18.0f, activeLine(3)); // D
    drawTubing2D(cx - 25.0f,    cy + 40.0f, bagPortXLeft,  yE - 18.0f, activeLine(4)); // E
    drawTubing2D(cx + 25.0f,    cy + 55.0f, bagPortXRight, yF - 18.0f, activeLine(5)); // F
    drawTubing2D(bagPortXLeft,  yG - 18.0f, cx - 10.0f, cy - 90.0f,  activeLine(6)); // G
    drawTubing2D(cx + 5.0f,     cy + 70.0f, bagPortXRight, yH - 18.0f, activeLine(7)); // H

    endOrtho2D();
}