# Active Brownian Particles — Prince Kumar
**B.Tech, Chemical Engineering | IIT Indore**
**Research Group: Prof. Dr. Kailasham Ramalingam**

---

## Overview

This directory contains the computational work developed as part of a pedagogical
study in soft matter physics. Rather than using pre-built "black box" packages,
every equation of motion, rotational diffusion algorithm, and trajectory analysis
tool was coded from scratch using **HOOMD-blue** as the simulation backbone.

The central goal is to validate theoretical predictions for **Active Brownian
Particles (ABPs)** — self-propelled microparticles operating at low Reynolds
number — by directly comparing numerical Mean Squared Displacement (MSD) data
against analytical theory.

---

## Repository Structure
Prince/
├── Active-Brownian-Particles/
│   ├── 2D_Disk/          # 2D reciprocal swimmer simulations
│   ├── 3D_Single_Sphere/ # 3D single ABP with quaternion rotations
│   └── (upcoming) 3D_Multiple_Spheres/  # MIPS in dense swarms
---

## Physics Background

### Active Brownian Particles
Self-propelled entities (microorganisms, synthetic colloids) move in a regime
where inertial forces are negligible. Their orientation undergoes continuous
**rotational Brownian diffusion**, causing their trajectories to be persistent
at short times but diffusive at long times.

### Reciprocal Swimmers
A reciprocal swimmer oscillates with speed `U(t) = U₀ cos(ωt)` — net active
displacement per cycle is zero, yet Lauga (2011) predicted these particles
still show **massively enhanced long-time diffusion** due to the interplay of
oscillation with thermal rotational noise. This counter-intuitive result is
validated here numerically.

### Key Formula (3D effective diffusivity)
The long-time diffusion coefficient for a reciprocal swimmer is:
D_eff = (1/6) * (U₀²/2) * Dr / (Dr² + ω²)
The `1/2` factor arises from the time-average of `cos²(ωt)` — a critical
distinction between steady and reciprocal swimmers that this project validates
computationally.

---

## Computational Methods

| Method | Details |
|--------|---------|
| **Time integration** | Explicit Euler with timestep Δt |
| **2D orientation** | Scalar angle θ with Gaussian rotational noise |
| **3D orientation** | Quaternion algebra (axis-angle → Hamilton product) |
| **Boundary conditions** | Periodic box with MIC unwrapping algorithm |
| **MSD analysis** | Ensemble and time-averaged over multiple trajectories |
| **Platform** | HP Z2 Tower G9 Workstation, HOOMD-blue |

---

## Results Summary

### 2D Disk
Simulation MSD matches the Kailasham & Khair (2023) analytical prediction:
- Short-time: ballistic oscillations visible for reciprocal swimmers
- Long-time: slope-1 diffusion with correct prefactor (1/4 factor validated)

### 3D Single Sphere
Simulation MSD matches the Lauga (2011) prediction for multiple ω values:
- ω = 0 (steady): standard ABP enhanced diffusion
- ω = 0.2, 1.0 (reciprocal): correct frequency-dependent diffusivity

---

## References

- Lauga, E. (2011). Enhanced diffusion by reciprocal swimming. *Physical Review Letters*, 106(17), 178101.
- Kailasham, R., & Khair, A. S. (2023). Effect of speed fluctuations on the collective dynamics of active disks. *Soft Matter*, 19(41), 7764–7774.
- Gonnella, G., et al. (2015). Motility-induced phase separation and coarsening in active matter. *Comptes Rendus Physique*, 16(3), 316–331.
