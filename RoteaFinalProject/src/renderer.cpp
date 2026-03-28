#include "renderer.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <cmath>
#include <vector>
#include "types.h"

namespace {
    constexpr float PI_F = 3.14159265358979323846f;

    float outerRadiusAtY(float y, float height, float outerBase, float outerTip) {
        float t = (y + height * 0.5f) / height;
        return outerBase + (outerTip - outerBase) * t;
    }

    float innerRadiusAtY(float y, float height, float innerBase, float innerTip) {
        float t = (y + height * 0.5f) / height;
        return innerBase + (innerTip - innerBase) * t;
    }

    void drawCircleXZ(float radius, float y, int segments) {
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float t = 2.0f * PI_F * static_cast<float>(i) / static_cast<float>(segments);
            glVertex3f(radius * std::cos(t), y, radius * std::sin(t));
        }
        glEnd();
    }

    void drawTipMarker(float height, float markerRadius = 0.045f) {
        const float tipY = -height * 0.5f; // flip to +height*0.5f if your chamber is inverted

        glPointSize(9.0f);
        glBegin(GL_POINTS);
        glColor4f(1.0f, 0.35f, 0.18f, 1.0f);
        glVertex3f(0.0f, tipY, 0.0f);
        glEnd();

        glColor4f(1.0f, 0.55f, 0.18f, 0.95f);
        drawCircleXZ(markerRadius, tipY, 28);

        glBegin(GL_LINES);
        glColor4f(1.0f, 0.45f, 0.20f, 0.90f);
        glVertex3f(0.0f, tipY, 0.0f);
        glVertex3f(0.0f, tipY + 0.08f, 0.0f);
        glEnd();
    }

    void drawFlowGuides(
        float outerBase, float outerTip,
        float innerBase, float innerTip,
        float height
    ) {
        const int guides = 8;
        const int segments = 56;
        const float tipY = -height * 0.5f; // flip if needed
        const float tipStartR = std::max(innerBase * 0.15f, 0.02f);

        glLineWidth(1.5f);

        for (int k = 0; k < guides; ++k) {
            float angle = 2.0f * PI_F * static_cast<float>(k) / static_cast<float>(guides);
            float ca = std::cos(angle);
            float sa = std::sin(angle);

            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= segments; ++i) {
                float u = static_cast<float>(i) / static_cast<float>(segments);

                // Rise through chamber, then softly return near the outer wall.
                float y;
                if (u < 0.65f) {
                    float a = u / 0.65f;
                    y = tipY + a * (height * 0.88f);
                } else {
                    float b = (u - 0.65f) / 0.35f;
                    y = tipY + (height * 0.88f) - b * (height * 0.28f);
                }

                float rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
                float rInner = innerRadiusAtY(y, height, innerBase, innerTip);

                // Expand outward from tip, then ride near outer wall.
                float radialFrac;
                if (u < 0.55f) {
                    float a = u / 0.55f;
                    radialFrac = 0.10f + 0.82f * std::pow(a, 0.85f);
                } else {
                    float b = (u - 0.55f) / 0.45f;
                    radialFrac = 0.92f - 0.12f * b;
                }

                float r = (1.0f - u) * tipStartR + u * (rInner + radialFrac * (rOuter - rInner));

                // Soft corkscrew so the guides feel rotational rather than flat.
                float twist = 0.85f * u;
                float ct = std::cos(angle + twist);
                float st = std::sin(angle + twist);

                float alpha = 0.15f + 0.35f * (1.0f - std::abs(0.5f - u) * 2.0f);
                glColor4f(0.22f, 0.82f, 1.0f, alpha);
                glVertex3f(r * ct, y, r * st);
            }
            glEnd();
        }

        glLineWidth(1.0f);
    }

    void drawInnerTipOutletStem(float height, float innerBase) {
        const float tipY = -height * 0.5f; // flip if needed
        const float stemLen = 0.12f;
        const float stemR = std::max(innerBase * 0.10f, 0.012f);

        glColor4f(0.92f, 0.92f, 0.96f, 0.75f);

        for (int i = 0; i < 3; ++i) {
            float a = 2.0f * PI_F * static_cast<float>(i) / 3.0f;
            float x = stemR * std::cos(a);
            float z = stemR * std::sin(a);

            glBegin(GL_LINES);
            glVertex3f(x, tipY - stemLen, z);
            glVertex3f(x, tipY, z);
            glEnd();
        }
    }
}

void setupProjection(int width, int height) {
    float aspect = static_cast<float>(width) / static_cast<float>(height == 0 ? 1 : height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float nearp = 0.1f;
    float farp  = 100.0f;
    float fov   = 60.0f;
    float top   = std::tan(fov * PI_F / 360.0f) * nearp;
    float right = top * aspect;

    glFrustum(-right, right, -top, top, nearp, farp);
    glMatrixMode(GL_MODELVIEW);
}

void drawParticles(const std::vector<Particle>& particles) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 0) {
            // fluid/background tracers
            glColor4f(0.20f, 0.65f, 1.0f, 0.30f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();

    glPointSize(4.8f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.type == 1) {
            // retained/suspended cells
            glColor4f(0.95f, 0.82f, 0.20f, 1.0f);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();

    glDisable(GL_BLEND);
}

void drawNestedConeChamber(
    float outerBase, float outerTip,
    float innerBase, float innerTip,
    float height
) {
    const int rings = 18;
    const int segments = 64;

    auto drawConeWire = [&](float baseR, float tipR, float alpha) {
        glColor4f(0.75f, 0.77f, 0.82f, alpha);

        for (int j = 0; j <= rings; ++j) {
            float tY = static_cast<float>(j) / static_cast<float>(rings);
            float y = -height * 0.5f + height * tY;
            float r = baseR + (tipR - baseR) * tY;

            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < segments; ++i) {
                float t = 2.0f * PI_F * static_cast<float>(i) / static_cast<float>(segments);
                glVertex3f(r * std::cos(t), y, r * std::sin(t));
            }
            glEnd();
        }

        glBegin(GL_LINES);
        for (int i = 0; i < 12; ++i) {
            float t = 2.0f * PI_F * static_cast<float>(i) / 12.0f;
            glVertex3f(baseR * std::cos(t), -height * 0.5f, baseR * std::sin(t));
            glVertex3f(tipR  * std::cos(t),  height * 0.5f, tipR  * std::sin(t));
        }
        glEnd();
    };

    // Outer chamber
    drawConeWire(outerBase, outerTip, 0.72f);

    // Inner cone
    drawConeWire(innerBase, innerTip, 0.42f);

    // Suggested recirculation / flow structure
    drawFlowGuides(outerBase, outerTip, innerBase, innerTip, height);

    // Inlet/tip marker
    drawTipMarker(height);

    // Small stem below inner cone tip so the inlet looks intentional
    drawInnerTipOutletStem(height, innerBase);

    // Retention bands on outer cone
    glColor4f(0.95f, 0.78f, 0.15f, 0.88f);
    for (float yy : {-0.75f * height * 0.5f, -0.15f, 0.35f}) {
        float tY = (yy + height * 0.5f) / height;
        float r = outerBase + (outerTip - outerBase) * tY;

        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float t = 2.0f * PI_F * static_cast<float>(i) / static_cast<float>(segments);
            glVertex3f(r * std::cos(t), yy, r * std::sin(t));
        }
        glEnd();
    }

    // Highlight the inner cone tip ring
    glColor4f(1.0f, 0.58f, 0.22f, 0.90f);
    drawCircleXZ(std::max(innerBase * 0.14f, 0.03f), -height * 0.5f, 28);
}