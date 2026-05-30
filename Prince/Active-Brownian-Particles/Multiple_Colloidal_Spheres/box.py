import gsd.hoomd
import numpy as np
import sys

# --- User-Defined Parameters ---
N_particles = 2000    # Upgraded particle count (MIPS requires a larger ensemble, e.g., 4000+)
phi = 0.40               # Target volume fraction (Packing fraction typical for 3D MIPS)
R_sphere = 1.0           # Radius of the spheres (Must match 'radius' parameter in simulation script)

# --- Dynamic Box Size Calculation ---
# 1. Volume of a single 3D sphere: V = (4/3) * pi * R^3
v_sphere = (4.0 / 3.0) * np.pi * (R_sphere ** 3)
total_particle_volume = N_particles * v_sphere

# 2. Total Box Volume: V_box = total_particle_volume / phi
# 3. Box length: L = (V_box)^(1/3)
L = (total_particle_volume / phi) ** (1.0 / 3.0)

print("--- System Geometry Configurations ---")
print(f"Number of Particles (N): {N_particles}")
print(f"Target Volume Fraction (phi): {phi}")
print(f"Calculated Box Length (L): {L:.4f}\n")

# --- Initialize GSD Snapshot ---
snapshot = gsd.hoomd.Snapshot()
snapshot.configuration.box = [L, L, L, 0, 0, 0]  # Fully 3D Box
snapshot.particles.N = N_particles
snapshot.particles.typeid = np.zeros(N_particles, dtype=int)
snapshot.particles.types = ['A']

# --- Overlap-Free Position Generation (Simple Cubic Lattice) ---
# Find the number of grid positions needed along one axis
M = int(np.ceil(N_particles ** (1.0 / 3.0)))
lattice_spacing = L / M

# Ensure the grid spacing is physically larger than the particle diameter
if lattice_spacing < (2.0 * R_sphere):
    raise ValueError(f"CRITICAL: Lattice spacing ({lattice_spacing:.2f}) is smaller than particle diameter ({2.0*R_sphere}). Decrease phi or check parameters.")

# Generate grid points centered at the origin
grid_points = np.arange(M) * lattice_spacing - L / 2.0 + lattice_spacing / 2.0
x, y, z = np.meshgrid(grid_points, grid_points, grid_points)
lattice_positions = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

# Assign the first N positions to our particles
snapshot.particles.position = lattice_positions[:N_particles]
print(f"Placed {N_particles} particles onto an SC lattice with spacing = {lattice_spacing:.4f}")

# --- Isotropic Orientation Generation (Random 3D Quaternions) ---
# To avoid directional bias at t=0, we generate uniform random unit quaternions [s, x, y, z].
# We do this by sampling from a 4D normal distribution and normalizing the resulting vector.
random_quaternions = np.random.normal(0, 1, (N_particles, 4))
quat_norms = np.linalg.norm(random_quaternions, axis=1, keepdims=True)
random_quaternions /= quat_norms

snapshot.particles.orientation = random_quaternions
print("Assigned uniform random 3D heading orientations to all particles.")

# --- Write To Initial Configuration File ---
op_file_name = f"Initial_3D_Dense_N_{N_particles}_phi_{phi:.2f}.gsd"
with gsd.hoomd.open(name=op_file_name, mode='wb') as f:
    f.append(snapshot)
    print(f"\n[Success] Created initial state file: '{op_file_name}'")