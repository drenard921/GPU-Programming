#ifndef SIMULATION_H
#define SIMULATION_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

bool initCudaParticleBuffer(int count);

bool updateParticlesCUDA(
    Particle* particles,
    int       count,
    float     dt,
    float     gForce,
    float     flow,
    float     height,
    float     outerBase,
    float     outerTip,
    float     innerBase,
    float     innerTip,
    int       phase
);

void shutdownCudaParticleBuffer();

#ifdef __cplusplus
}
#endif

#endif // SIMULATION_H