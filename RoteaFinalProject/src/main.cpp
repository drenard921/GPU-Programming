#include <iostream>
#include <vector>

#include <GLFW/glfw3.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "renderer.h"
#include "sim_cpu.h"
#include "types.h"

static constexpr int CHAMBER_NODE = -1;

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error [" << error << "]: " << description << std::endl;
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) return 1;

    GLFWwindow* window = glfwCreateWindow(1200, 800, "Rotea CUDA Sim", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float chamberHeight = 2.5f;

    const float outerBase = 1.15f;
    const float outerTip  = 0.28f;

    const float innerBase = 0.55f;
    const float innerTip  = 0.12f;

    const float dt = 0.016f;

    std::vector<Particle> particles = initParticles(
        15000, 1500,
        chamberHeight,
        outerBase, outerTip,
        innerBase, innerTip
    );

    std::vector<Bag> bags = {
        {"Sample",  500.0f, 0.0f},
        {"Buffer",  500.0f, 0.0f},
        {"Product",   0.0f, 0.0f},
        {"Waste",     0.0f, 0.0f}
    };

    Chamber chamber{
        0.0f,  // retained_cells
        0.0f,  // suspended_cells
        1.00f, // media_density
        1.00f, // media_viscosity
        0.0f   // omega
    };

    std::vector<Line> lines(8);
    lines[0] = {LineID::A, 0,            CHAMBER_NODE, 0.0f, true};
    lines[1] = {LineID::B, 1,            CHAMBER_NODE, 0.0f, true};
    lines[2] = {LineID::C, CHAMBER_NODE, 2,            0.0f, true};
    lines[3] = {LineID::D, CHAMBER_NODE, 3,            0.0f, true};
    lines[4] = {LineID::E, CHAMBER_NODE, 2,            0.0f, true};
    lines[5] = {LineID::F, CHAMBER_NODE, 3,            0.0f, true};
    lines[6] = {LineID::G, 2,            CHAMBER_NODE, 0.0f, true};
    lines[7] = {LineID::H, CHAMBER_NODE, 2,            0.0f, true};

    // std::vector<Step> protocol = {
    //     {
    //         "Bed Stabilize",
    //         PhaseType::Concentrate,
    //         100.0f,
    //         3000.0f,
    //         {0.0f, 12.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    //     }
    // };

    std::vector<Step> protocol = {
        {
            "Load",
            PhaseType::Load,
            8.0f,
            1200.0f,
            {20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
            },
        {
            "Bed Stabilize",
            PhaseType::Concentrate,
            15.0f,
            3000.0f,
            {0.0f, 12.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        },
        {
            "Wash",
            PhaseType::Wash,
            12.0f,
            2600.0f,
            {0.0f, 15.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        },
        {
            "Harvest",
            PhaseType::Harvest,
            10.0f,
            800.0f,
            {0.0f, 0.0f, 18.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        }
    };

    int currentStep = 0;
    float stepTimer = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        Step& step = protocol[currentStep];

        updateSystem(bags, lines, chamber, step, dt);

        float chamberFlow = 0.0f;
        for (const auto& line : lines) {
            if (!line.active) continue;
            if (line.sourceBag == CHAMBER_NODE || line.targetBag == CHAMBER_NODE) {
                chamberFlow += step.flow_ml_min[static_cast<int>(line.id)];
            }
        }

        updateParticles(
            particles,
            dt,
            step.g_force,
            chamberFlow,
            chamberHeight,
            outerBase, outerTip,
            innerBase, innerTip
        );

        stepTimer += dt;
        if (stepTimer >= step.duration_s) {
            std::cout
                << "Phase complete: " << step.name
                << " | retained=" << chamber.retained_cells
                << " | suspended=" << chamber.suspended_cells
                << " | omega=" << chamber.omega
                << std::endl;

            stepTimer = 0.0f;
            currentStep = (currentStep + 1) % static_cast<int>(protocol.size());
        }

        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        setupProjection(width, height);

        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -5.0f);
        glRotatef(20.0f, 1.0f, 0.0f, 0.0f);
        glRotatef(static_cast<float>(glfwGetTime()) * 6.0f, 0.0f, 1.0f, 0.0f);

        drawNestedConeChamber(
            outerBase, outerTip,
            innerBase, innerTip,
            chamberHeight
        );
        drawParticles(particles);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}