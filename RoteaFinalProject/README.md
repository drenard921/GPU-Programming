#### Dylan Renard

#### EN.605.617 Introduction to GPU Programming (JHU)

#### Professor Chance Pascale

#### March 23, 2026

# GPU-Accelerated Counterflow Centrifugation Simulation

## Real-Time Particle-Based Modeling of a Rotea-Inspired CFC System

------------------------------------------------------------------------

# 1. Abstract

This project proposes the development of a real-time, GPU-accelerated
simulation of counterflow centrifugation (CFC) within a rotating conical
chamber inspired by the CTS Rotea platform. The system models particle
transport under coupled centrifugal, drag, and axial flow forces using a
particle-based approximation.

The primary objective is to construct a physically motivated,
computationally efficient framework that enables real-time visualization
of phase separation dynamics (fluid vs. cellular components), while
serving as a foundation for GPU-parallel simulation using CUDA.

------------------------------------------------------------------------

# 2. Problem Context and Significance

Counterflow centrifugation systems are widely used in bioprocessing
workflows, particularly in cell therapy manufacturing, where separation
efficiency directly impacts yield and purity. Despite their importance,
these systems are often treated as black boxes due to the difficulty of
observing internal dynamics.

This project addresses that gap by:

-   Providing an interpretable simulation of particle transport under
    CFC conditions\
-   Enabling rapid prototyping of process parameters (flow rate,
    g-force)\
-   Demonstrating how GPU acceleration can support real-time scientific
    visualization

------------------------------------------------------------------------

# 3. Modeling Approach

## 3.1 System Representation

The CFC chamber is approximated as a truncated conical annulus defined
by inner and outer radial boundaries.

## 3.2 Particle-Based Approximation

Particles represent either fluid (low density) or cellular (high
density) phases.

## 3.3 Force Model

Each particle evolves under:

-   Centrifugal force\
-   Axial flow\
-   Radial equilibrium banding\
-   Tangential swirl\
-   Anisotropic drag\
-   Settling bias

------------------------------------------------------------------------

# 4. Numerical Considerations

The simulation uses discrete-time updates with position-based dynamics,
prioritizing stability and real-time performance.

------------------------------------------------------------------------

# 5. Rendering and Visualization

OpenGL is used for real-time rendering of:

-   Chamber geometry\
-   Particle motion\
-   Phase separation

Color scheme: - Blue: Fluid\
- Yellow: Cells

------------------------------------------------------------------------

# 6. Protocol Simulation

Phases include:

-   Load\
-   Wash\
-   Concentration\
-   Harvest

Each phase modifies flow and rotational parameters.

------------------------------------------------------------------------

# 7. Software Architecture

src/ ├── main.cpp\
├── sim_cpu.cpp\
├── simulation.cu\
├── renderer.cpp\
├── renderer.h\
├── sim_cpu.h\
├── types.h

------------------------------------------------------------------------

# 8. GPU Acceleration Strategy

Particle updates are fully parallelizable:

-   One thread per particle\
-   Scales to large particle counts\
-   Enables real-time performance

------------------------------------------------------------------------

# 9. Current Results

-   Stable separation bands\
-   Distinct particle behaviors\
-   Smooth real-time rendering

------------------------------------------------------------------------

# 10. Limitations

-   Simplified physics\
-   No full fluid simulation\
-   No experimental validation

------------------------------------------------------------------------

# 11. Future Work

-   Full CUDA implementation\
-   Velocity-based dynamics\
-   Parameter tuning UI\
-   Experimental validation

------------------------------------------------------------------------

# 12. Conclusion

This project demonstrates a GPU-ready framework for simulating
biologically relevant centrifugation systems.

------------------------------------------------------------------------

# 13. Acknowledgments

Inspired by real-world bioprocessing systems made by Thermo Fischer Scientific.
