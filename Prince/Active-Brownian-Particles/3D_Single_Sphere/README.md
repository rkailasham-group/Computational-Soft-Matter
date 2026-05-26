# 3D Single Sphere ABP Simulation

Simulates a single Active Brownian Particle (sphere) in 3D using HOOMD-blue
with quaternion-based rotational diffusion.

## Files
- `simulate.py` — Main simulation script
- `msd.py` — MSD calculation
- `vis.py` — Visualization script
- `box.py` — Box/boundary setup
- `results/` — Output MSD plots

## Physics
Validates enhanced diffusion by reciprocal swimming as predicted by Lauga (2011).
Multiple omega values tested.

## Usage
```bash
python simulate.py
python msd.py
```
