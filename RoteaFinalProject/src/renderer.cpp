#include "renderer.h"

#include <GL/gl.h>
#include <cmath>
#include <vector>

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

void drawParticles(const std::vector<Particle>& particles) {
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 0) {
            glColor4f(0.20f, 0.65f, 1.0f, 0.35f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();

    glPointSize(4.5f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 1) {
            glColor4f(0.95f, 0.82f, 0.20f, 1.0f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();
}

void drawNestedConeChamber(
    float outerBase, float outerTip,
    float innerBase, float innerTip,
    float height
) {
    const int rings = 16;
    const int segments = 64;

    auto drawConeWire = [&](float baseR, float tipR, float alpha) {
        glColor4f(0.75f, 0.77f, 0.82f, alpha);

        for (int j = 0; j <= rings; ++j) {
            float tY = static_cast<float>(j) / static_cast<float>(rings);
            float y = -height * 0.5f + height * tY;
            float r = baseR + (tipR - baseR) * tY;

            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < segments; ++i) {
                float t = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(segments);
                glVertex3f(r * std::cos(t), y, r * std::sin(t));
            }
            glEnd();
        }

        glBegin(GL_LINES);
        for (int i = 0; i < 12; ++i) {
            float t = 2.0f * 3.14159265f * static_cast<float>(i) / 12.0f;
            glVertex3f(baseR * std::cos(t), -height * 0.5f, baseR * std::sin(t));
            glVertex3f(tipR  * std::cos(t),  height * 0.5f, tipR  * std::sin(t));
        }
        glEnd();
    };

    // outer cone
    drawConeWire(outerBase, outerTip, 0.70f);

    // inner cone
    drawConeWire(innerBase, innerTip, 0.35f);

    // optional retention bands on outer cone
    glColor4f(0.95f, 0.78f, 0.15f, 0.9f);
    for (float yy : {-0.75f * height * 0.5f, -0.15f, 0.35f}) {
        float tY = (yy + height * 0.5f) / height;
        float r = outerBase + (outerTip - outerBase) * tY;

        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float t = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(segments);
            glVertex3f(r * std::cos(t), yy, r * std::sin(t));
        }
        glEnd();
    }
}