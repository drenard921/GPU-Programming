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


std::vector<Particle> initParticles(int nFluid, int nCells, float radius, float height) {
    std::vector<Particle> particles;
    particles.reserve(nFluid + nCells);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uy(-height * 0.5f, height * 0.5f);

    auto sampleInCylinder = [&](int type) {
        while (true) {
            float x = u(rng) * radius;
            float z = u(rng) * radius;
            if (x * x + z * z <= radius * radius) {
                float y = uy(rng);
                particles.push_back({x, y, z, type});
                return;
            }
        }
    };

    for (int i = 0; i < nFluid; ++i) sampleInCylinder(0);
    for (int i = 0; i < nCells; ++i) sampleInCylinder(1);

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

void updateParticles(std::vector<Particle>& particles, float dt, float gForce, float flow) {
    for (auto& p : particles) {
        bool isCell = (p.type == 1);
        float density = isCell ? 1.056f : 1.000f;

        float x = p.x;
        float y = p.y;
        float z = p.z;

        float r = std::sqrt(x * x + z * z) + 1e-6f;
        float rx = x / r;
        float rz = z / r;

        // simplified centrifugal vs drag behavior
        float radialPush = gForce * (density - 0.98f) * 0.02f;
        float axial = flow * (isCell ? 0.7f : 1.0f);

        float swirl = 0.4f;
        float tx = -rz * swirl;
        float tz =  rx * swirl;

        x += (radialPush * rx + tx) * dt;
        z += (radialPush * rz + tz) * dt;
        y += axial * dt;

        float maxR = 1.0f;
        float newR = std::sqrt(x * x + z * z);
        if (newR > maxR) {
            float s = maxR / newR;
            x *= s;
            z *= s;
        }

        float halfH = 1.0f;
        if (y > halfH) y = -halfH;
        if (y < -halfH) y =  halfH;

        p.x = x;
        p.y = y;
        p.z = z;
    }
}

void processInput(GLFWwindow* window, float& gForce, float& flow) {
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)    gForce += 0.05f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)  gForce -= 0.05f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) flow += 0.01f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)  flow -= 0.01f;

    if (gForce < 0.0f) gForce = 0.0f;
    if (flow < 0.0f) flow = 0.0f;
}