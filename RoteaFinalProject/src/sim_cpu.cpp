#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <GLFW/glfw3.h>

#include "sim_cpu.h"
#include "types.h"

namespace {
    constexpr int CHAMBER_NODE = -1;

    float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }
}


static float outerRadiusAtY(float y, float height, float outerBase, float outerTip) {
    float t = (y + height * 0.5f) / height;
    return outerBase + (outerTip - outerBase) * t;
}

static float innerRadiusAtY(float y, float height, float innerBase, float innerTip) {
    float t = (y + height * 0.5f) / height;
    return innerBase + (innerTip - innerBase) * t;
}

std::vector<Particle> initParticles(
    int nFluid,
    int nCells,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip
) {
    std::vector<Particle> particles;
    particles.reserve(nFluid + nCells);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uy(-height * 0.5f, height * 0.5f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> uang(0.0f, 2.0f * 3.14159265f);

    auto sampleInShell = [&](int type) {
        while (true) {
            float y = uy(rng);

            float rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
            float rInner = innerRadiusAtY(y, height, innerBase, innerTip);

            // area-aware sampling in annulus
            float rr = std::sqrt(u01(rng) * (rOuter * rOuter - rInner * rInner) + rInner * rInner);
            float a = uang(rng);

            float x = rr * std::cos(a);
            float z = rr * std::sin(a);

            particles.push_back({x, y, z, type});
            return;
        }
    };

    for (int i = 0; i < nFluid; ++i) sampleInShell(0);
    for (int i = 0; i < nCells; ++i) sampleInShell(1);

    return particles;
}

void updateSystem(
    std::vector<Bag>& bags,
    std::vector<Line>& lines,
    Chamber& chamber,
    const Step& step,
    float dt
) {
    const float dt_min = dt / 60.0f;

    // Track only chamber-relevant flow for drag model
    float chamberFlow_ml_min = 0.0f;

    for (int i = 0; i < 8; ++i) {
        float commandedFlow = step.flow_ml_min[i];
        if (commandedFlow <= 0.0f) continue;

        Line& line = lines[i];
        if (!line.active) continue;

        const bool sourceIsChamber = (line.sourceBag == CHAMBER_NODE);
        const bool targetIsChamber = (line.targetBag == CHAMBER_NODE);

        float srcPressure = 0.0f;
        float dstPressure = 0.0f;

        if (!sourceIsChamber) srcPressure = bags[line.sourceBag].pressure_kpa;
        if (!targetIsChamber) dstPressure = bags[line.targetBag].pressure_kpa;

        // Apply pressure BEFORE computing transferred volume
        float dP = srcPressure - dstPressure;
        float effectiveFlow = commandedFlow * clampf(1.0f + 0.01f * dP, 0.25f, 2.0f);

        float volumeTransfer = effectiveFlow * dt_min;

        // BAG -> CHAMBER
        if (!sourceIsChamber && targetIsChamber) {
            Bag& src = bags[line.sourceBag];

            if (src.volume_ml < volumeTransfer)
                volumeTransfer = src.volume_ml;

            src.volume_ml -= volumeTransfer;

            // very simple chamber loading model
            chamber.suspended_cells += volumeTransfer * 0.1f;
            chamberFlow_ml_min += effectiveFlow;
        }

        // CHAMBER -> BAG
        else if (sourceIsChamber && !targetIsChamber) {
            Bag& dst = bags[line.targetBag];

            dst.volume_ml += volumeTransfer;

            // remove suspended cells first, then retained cells if harvest-like conditions
            float suspendedOut = std::min(chamber.suspended_cells, volumeTransfer * 0.08f);
            chamber.suspended_cells -= suspendedOut;

            // during lower g-force or harvest-like output, allow retained cells to leave too
            if (step.phase == PhaseType::Harvest) {
                float retainedOut = std::min(chamber.retained_cells, volumeTransfer * 0.15f);
                chamber.retained_cells -= retainedOut;
            }

            chamberFlow_ml_min += effectiveFlow;
        }

        // BAG -> BAG
        else if (!sourceIsChamber && !targetIsChamber) {
            Bag& src = bags[line.sourceBag];
            Bag& dst = bags[line.targetBag];

            if (src.volume_ml < volumeTransfer)
                volumeTransfer = src.volume_ml;

            src.volume_ml -= volumeTransfer;
            dst.volume_ml += volumeTransfer;
        }

        // CHAMBER -> CHAMBER is nonsense, ignore it
    }

    // === CHAMBER PHYSICS ===
    // Manual model: opposing centrifugation and drag forces govern particle behavior.
    // We keep this simplified but make it depend only on chamber-relevant flow. :contentReference[oaicite:5]{index=5}
    float F_c = step.g_force;
    float F_d = 0.05f * chamberFlow_ml_min;
    float net = F_c - F_d;

    if (net > 0.0f) {
        // retention / bed formation
        float retainFrac = clampf(0.02f + 0.00005f * net, 0.0f, 0.25f);
        float retain = chamber.suspended_cells * retainFrac;
        chamber.retained_cells += retain;
        chamber.suspended_cells -= retain;
    } else {
        // washout / elutriation
        float washFrac = clampf(0.02f + 0.00005f * (-net), 0.0f, 0.25f);
        float loss = chamber.retained_cells * washFrac;
        chamber.retained_cells -= loss;
        chamber.suspended_cells += loss * 0.25f; // some re-entrainment
    }

    chamber.retained_cells = std::max(0.0f, chamber.retained_cells);
    chamber.suspended_cells = std::max(0.0f, chamber.suspended_cells);
}

void constrainToNestedCone(
    Particle& p,
    float height,
    float outerBase,
    float outerTip,
    float innerBase,
    float innerTip
) {
    float halfH = height * 0.5f;

    if (p.y > halfH) p.y = -halfH;
    if (p.y < -halfH) p.y =  halfH;

    float r = std::sqrt(p.x * p.x + p.z * p.z) + 1e-6f;

    float rOuter = outerRadiusAtY(p.y, height, outerBase, outerTip);
    float rInner = innerRadiusAtY(p.y, height, innerBase, innerTip);

    if (r > rOuter) {
        float s = rOuter / r;
        p.x *= s;
        p.z *= s;
    }

    if (r < rInner) {
        float s = rInner / r;
        p.x *= s;
        p.z *= s;
    }
}
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
) {
    for (auto& p : particles) {
        const bool isCell = (p.type == 1);
        const float density = isCell ? 1.08f : 1.00f;

        float x = p.x;
        float y = p.y;
        float z = p.z;

        float r = std::sqrt(x * x + z * z) + 1e-6f;
        float rx = x / r;
        float rz = z / r;

        float rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
        float rInner = innerRadiusAtY(y, height, innerBase, innerTip);

        // gentler centrifugal separation
        float radialPush = gForce * (density - 1.0f) * 0.018f;

        // flow pushes fluid more than cells
        float axial = flow * (isCell ? 0.30f : 0.55f);

        // downward settling
        float settling = isCell ? -0.035f : -0.015f;

        // mild swirl
        float swirl = 0.12f;
        float tx = -rz * swirl;
        float tz =  rx * swirl;

        // flow-aware equilibrium band
        float flowEffect = std::clamp(flow * 0.0025f, 0.0f, 0.18f);
        float targetFrac = isCell ? (0.82f - flowEffect) : (0.42f - 0.35f * flowEffect);
        float targetR = rInner + targetFrac * (rOuter - rInner);

        // softer pull toward band
        float bandCorrection = (targetR - r) * (isCell ? 0.55f : 0.22f);

        // decompose motion
        float radialStepX = (radialPush * rx + bandCorrection * rx) * dt;
        float radialStepZ = (radialPush * rz + bandCorrection * rz) * dt;

        float swirlStepX = tx * dt;
        float swirlStepZ = tz * dt;

        float axialStep = (axial + settling) * dt;

        // realistic drag / damping
        float radialDrag = isCell ? 0.88f : 0.93f;
        float swirlDrag  = isCell ? 0.80f : 0.88f;
        float axialDrag  = isCell ? 0.90f : 0.95f;

        radialStepX *= radialDrag;
        radialStepZ *= radialDrag;

        swirlStepX *= swirlDrag;
        swirlStepZ *= swirlDrag;

        axialStep *= axialDrag;

        x += radialStepX + swirlStepX;
        z += radialStepZ + swirlStepZ;
        y += axialStep;

        // vertical wrap
        float halfH = height * 0.5f;
        if (y > halfH) y = -halfH;
        if (y < -halfH) y =  halfH;

        // recompute bounds at new y
        rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
        rInner = innerRadiusAtY(y, height, innerBase, innerTip);

        float newR = std::sqrt(x * x + z * z) + 1e-6f;

        // clamp to outer wall
        if (newR > rOuter) {
            float s = rOuter / newR;
            x *= s;
            z *= s;
            newR = rOuter;
        }

        // clamp to inner wall
        if (newR < rInner) {
            float s = rInner / newR;
            x *= s;
            z *= s;
        }

        p.x = x;
        p.y = y;
        p.z = z;
    }
}