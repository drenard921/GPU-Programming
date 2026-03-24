#include <iostream>
#include <vector>

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include "renderer.h"
#include "sim_cpu.h"
#include "types.h"

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

    const float chamberRadius = 1.0f;
    const float chamberHeight = 2.0f;
    const float dt = 0.016f;

    std::vector<Particle> particles = initParticles(15000, 1500, chamberRadius, chamberHeight);

    // Bags: Sample, Buffer, Product, Waste
    std::vector<Bag> bags = {
        {"Sample",  500.0f, 0.0f},
        {"Buffer",  500.0f, 0.0f},
        {"Product",   0.0f, 0.0f},
        {"Waste",     0.0f, 0.0f}
    };

    // Chamber state
    Chamber chamber{0.0f, 0.0f};

    // Line routing: -1 means chamber
    std::vector<Line> lines(8);
    lines[0] = {LineID::A,  0, -1, 0.0f, true}; // Sample -> Chamber
    lines[1] = {LineID::B,  1, -1, 0.0f, true}; // Buffer -> Chamber
    lines[2] = {LineID::C, -1,  2, 0.0f, true}; // Chamber -> Product
    lines[3] = {LineID::D, -1,  3, 0.0f, true}; // Chamber -> Waste
    lines[4] = {LineID::E, -1,  2, 0.0f, true}; // Chamber -> Product alt
    lines[5] = {LineID::F, -1,  3, 0.0f, true}; // Chamber -> Waste alt
    lines[6] = {LineID::G,  2, -1, 0.0f, true}; // Product/intermediate -> Chamber
    lines[7] = {LineID::H, -1,  2, 0.0f, true}; // Chamber -> Product alt

    // Simple protocol
    std::vector<Step> protocol = {
        {
            "Load",
            PhaseType::Load,
            5.0f,
            500.0f,
            {50.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        },
        {
            "Wash",
            PhaseType::Wash,
            5.0f,
            800.0f,
            {0.0f, 50.0f, 0.0f, 50.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        },
        {
            "Concentrate",
            PhaseType::Concentrate,
            5.0f,
            1200.0f,
            {0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        },
        {
            "Harvest",
            PhaseType::Harvest,
            5.0f,
            300.0f,
            {0.0f, 0.0f, 50.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
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
            if (line.sourceBag == -1 || line.targetBag == -1) {
                chamberFlow += step.flow_ml_min[(int)line.id];
            }
        }

        updateParticles(particles, dt, step.g_force, chamberFlow);

        stepTimer += dt;
        if (stepTimer >= step.duration_s) {
            stepTimer = 0.0f;
            currentStep = (currentStep + 1) % protocol.size();
        }

        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        setupProjection(width, height);

        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -5.0f);
        glRotatef(20.0f, 1.0f, 0.0f, 0.0f);
        glRotatef((float)glfwGetTime() * 10.0f, 0.0f, 1.0f, 0.0f);

        drawConeChamber(0.95f, 0.25f, 2.0f);
        drawParticles(particles);
        drawKitOverlay(bags, lines);

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