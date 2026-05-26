# 2D Active Brownian Disk Simulation

Simulates a reciprocal swimmer (Active Brownian Particle) in 2D using HOOMD-blue.

## Files
- `simulate.py` — Main simulation script (explicit Euler integration)
- `plot_msd.py` — MSD calculation and plotting
- `results/` — Output MSD plots

## Physics
Validates the effective diffusivity formula from Kailasham & Khair (2023)
for both steady (omega=0) and reciprocal swimmers.

## Usage
```bash
python simulate.py
python plot_msd.py
```
