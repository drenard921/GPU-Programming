#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "sim_cpu.h"
#include "types.h"

namespace {
    constexpr int CHAMBER_NODE = -1;
    constexpr float PI_F = 3.14159265358979323846f;

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
    std::uniform_real_distribution<float> uang(0.0f, 2.0f * PI_F);

    auto sampleInShell = [&](int type) {
        float y = uy(rng);

        float rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
        float rInner = innerRadiusAtY(y, height, innerBase, innerTip);

        // area-aware annulus sampling
        float rr = std::sqrt(u01(rng) * (rOuter * rOuter - rInner * rInner) + rInner * rInner);
        float a = uang(rng);

        Particle p{};
        p.x = rr * std::cos(a);
        p.y = y;
        p.z = rr * std::sin(a);

        p.vx = 0.0f;
        p.vy = 0.0f;
        p.vz = 0.0f;

        p.type = type;

        if (type == 1) {
            // cells
            p.diameter = 0.12f;
            p.density  = 1.08f;
        } else {
            // fluid tracer / background particles
            p.diameter = 0.06f;
            p.density  = 1.00f;
        }

        particles.push_back(p);
    };

    for (int i = 0; i < nFluid; ++i) sampleInShell(0);
    for (int i = 0; i < nCells;  ++i) sampleInShell(1);

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

    // Keep chamber physical properties sane.
    if (chamber.media_density <= 0.0f)   chamber.media_density = 1.00f;
    if (chamber.media_viscosity <= 0.0f) chamber.media_viscosity = 1.00f;
    chamber.omega = 0.06f * step.g_force;

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

        float dP = srcPressure - dstPressure;
        float effectiveFlow = commandedFlow * clampf(1.0f + 0.01f * dP, 0.25f, 2.0f);

        float volumeTransfer = effectiveFlow * dt_min;

        // BAG -> CHAMBER
        if (!sourceIsChamber && targetIsChamber) {
            Bag& src = bags[line.sourceBag];

            if (src.volume_ml < volumeTransfer) {
                volumeTransfer = src.volume_ml;
            }

            src.volume_ml -= volumeTransfer;

            // simple loading model
            chamber.suspended_cells += volumeTransfer * 0.10f;
            chamberFlow_ml_min += effectiveFlow;
        }
        // CHAMBER -> BAG
        else if (sourceIsChamber && !targetIsChamber) {
            Bag& dst = bags[line.targetBag];
            dst.volume_ml += volumeTransfer;

            float suspendedOut = std::min(chamber.suspended_cells, volumeTransfer * 0.08f);
            chamber.suspended_cells -= suspendedOut;

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

            if (src.volume_ml < volumeTransfer) {
                volumeTransfer = src.volume_ml;
            }

            src.volume_ml -= volumeTransfer;
            dst.volume_ml += volumeTransfer;
        }
        // CHAMBER -> CHAMBER ignored
    }

    // Coarse chamber-level retention model.
    // This remains simplified, but now at least tracks the chamber's rotational state.
    float F_c = step.g_force;
    float F_d = 0.05f * chamberFlow_ml_min;
    float net = F_c - F_d;

    if (net > 0.0f) {
        float retainFrac = clampf(0.02f + 0.00005f * net, 0.0f, 0.25f);
        float retain = chamber.suspended_cells * retainFrac;
        chamber.retained_cells += retain;
        chamber.suspended_cells -= retain;
    } else {
        float washFrac = clampf(0.02f + 0.00005f * (-net), 0.0f, 0.25f);
        float loss = chamber.retained_cells * washFrac;
        chamber.retained_cells -= loss;
        chamber.suspended_cells += loss * 0.25f;
    }

    chamber.retained_cells  = std::max(0.0f, chamber.retained_cells);
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
    const float eps = 1e-6f;
    const float halfH = height * 0.5f;

    if (p.y > halfH)  p.y = -halfH;
    if (p.y < -halfH) p.y =  halfH;

    float r = std::sqrt(p.x * p.x + p.z * p.z) + eps;

    float rOuter = outerRadiusAtY(p.y, height, outerBase, outerTip);
    float rInner = innerRadiusAtY(p.y, height, innerBase, innerTip);

    if (r > rOuter) {
        float s = rOuter / r;
        p.x *= s;
        p.z *= s;
        p.vx *= 0.65f;
        p.vz *= 0.65f;
    }

    r = std::sqrt(p.x * p.x + p.z * p.z) + eps;
    if (r < rInner) {
        float s = rInner / r;
        p.x *= s;
        p.z *= s;
        p.vx *= 0.65f;
        p.vz *= 0.65f;
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
    const float eps = 1e-6f;
    const float halfH = height * 0.5f;

    // These are local defaults for now.
    // Later you can route chamber.media_density / chamber.media_viscosity here directly.
    const float mediaDensity   = 1.00f;
    const float mediaViscosity = 1.00f;

    // Control knob -> approximate rotational strength
    const float omega = 0.06f * gForce;

    // Tip-driven inlet jet tuning
    const float jetRadius   = 0.18f;
    const float jetStrength = 0.020f * flow;
    const float jetDecay    = 8.0f;

    // Recirculation and rotational swirl
    const float wallReturnStrength = 0.35f;
    const float swirlStrength      = 0.08f;

    // Global damping
    const float velDamping = 0.965f;

    // Choose which end of the chamber acts like the inner cone tip.
    // Flip this sign if your rendered chamber is oriented the other way.
    const float tipY = -halfH;

    for (auto& p : particles) {
        float x  = p.x;
        float y  = p.y;
        float z  = p.z;
        float vx = p.vx;
        float vy = p.vy;
        float vz = p.vz;

        float r = std::sqrt(x * x + z * z) + eps;
        float rx = x / r;
        float rz = z / r;

        float rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
        float rInner = innerRadiusAtY(y, height, innerBase, innerTip);

        // -----------------------------
        // 1. Tip-driven jet from inner-cone tip
        // -----------------------------
        float dyTip = y - tipY;
        float distToTip = std::sqrt(r * r + dyTip * dyTip) + eps;

        // Localized nozzle weighting: strongest near axis around the tip
        float radialNozzleWeight = std::exp(-(r * r) / std::max(jetRadius * jetRadius, eps));
        float jetFalloff = std::exp(-jetDecay * distToTip) * radialNozzleWeight;

        // Flow leaves tip and expands outward into chamber
        float jetDirX = rx;
        float jetDirY = dyTip / distToTip;
        float jetDirZ = rz;

        float vJetX = jetStrength * jetFalloff * jetDirX;
        float vJetY = jetStrength * jetFalloff * jetDirY;
        float vJetZ = jetStrength * jetFalloff * jetDirZ;

        // -----------------------------
        // 2. Wall-following return flow
        // -----------------------------
        float gap = std::max(rOuter - rInner, eps);
        float wallFrac = (r - rInner) / gap; // 0 near inner wall, 1 near outer wall
        wallFrac = clampf(wallFrac, 0.0f, 1.0f);

        float nearOuter = clampf((wallFrac - 0.65f) / 0.35f, 0.0f, 1.0f);

        // Return flow trends back toward tip along outer wall region
        float vReturnY = -wallReturnStrength * flow * 0.0025f * nearOuter;
        float vReturnX = -rx * wallReturnStrength * flow * 0.0015f * nearOuter;
        float vReturnZ = -rz * wallReturnStrength * flow * 0.0015f * nearOuter;

        // -----------------------------
        // 3. Rotational swirl
        // -----------------------------
        float vSwirlX = -rz * swirlStrength * omega;
        float vSwirlY =  0.0f;
        float vSwirlZ =  rx * swirlStrength * omega;

        // Total local flow field
        float flowVX = vJetX + vReturnX + vSwirlX;
        float flowVY = vJetY + vReturnY + vSwirlY;
        float flowVZ = vJetZ + vReturnZ + vSwirlZ;

        // -----------------------------
        // 4. Stokes-like drag toward local fluid velocity
        // -----------------------------
        float relVX = flowVX - vx;
        float relVY = flowVY - vy;
        float relVZ = flowVZ - vz;

        float d = std::max(p.diameter, 0.02f);
        float dragCoeff = 3.0f * PI_F * mediaViscosity * d;

        float axDrag = dragCoeff * relVX;
        float ayDrag = dragCoeff * relVY;
        float azDrag = dragCoeff * relVZ;

        // -----------------------------
        // 5. Centrifugal outward tendency
        // -----------------------------
        float densityDelta = std::max(p.density - mediaDensity, 0.0f);

        float aCent =
            densityDelta *
            PI_F *
            d * d * d *
            omega * omega *
            r / 6.0f;

        float axCent = rx * aCent;
        float ayCent = 0.0f;
        float azCent = rz * aCent;

        // -----------------------------
        // 6. Integrate
        // -----------------------------
        float ax = axDrag + axCent;
        float ay = ayDrag;
        float az = azDrag + azCent;

        vx += ax * dt;
        vy += ay * dt;
        vz += az * dt;

        vx *= velDamping;
        vy *= velDamping;
        vz *= velDamping;

        x += vx * dt;
        y += vy * dt;
        z += vz * dt;

        // -----------------------------
        // 7. Vertical wrap
        // -----------------------------
        if (y > halfH)  y = -halfH;
        if (y < -halfH) y =  halfH;

        // -----------------------------
        // 8. Recompute local cone bounds and clamp
        // -----------------------------
        rOuter = outerRadiusAtY(y, height, outerBase, outerTip);
        rInner = innerRadiusAtY(y, height, innerBase, innerTip);

        float newR = std::sqrt(x * x + z * z) + eps;

        if (newR > rOuter) {
            float s = rOuter / newR;
            x *= s;
            z *= s;
            vx *= 0.65f;
            vz *= 0.65f;
            newR = rOuter;
        }

        if (newR < rInner) {
            float s = rInner / newR;
            x *= s;
            z *= s;
            vx *= 0.65f;
            vz *= 0.65f;
        }

        p.x  = x;
        p.y  = y;
        p.z  = z;
        p.vx = vx;
        p.vy = vy;
        p.vz = vz;
    }
}