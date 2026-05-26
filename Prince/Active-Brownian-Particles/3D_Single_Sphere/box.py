import gsd.hoomd
import numpy as np

# --- Parameters ---
N_particles = 500  # Upgraded for ensemble averaging
L = 50.0         # Box size

# --- Create Snapshot ---
snapshot = gsd.hoomd.Snapshot()
snapshot.configuration.box = [L, L, L, 0, 0, 0]  # 3D Box

snapshot.particles.N = N_particles

# Scatter particles randomly to avoid overlap crashes later
snapshot.particles.position = np.random.uniform(-L/2.1, L/2.1, (N_particles, 3)) 
snapshot.particles.typeid = np.zeros(N_particles, dtype=int)
snapshot.particles.types = ['A']

# Initialize Orientation: [s, x, y, z]
# Start all facing the X-axis: quaternion = [1, 0, 0, 0]
orientation_array = np.zeros((N_particles, 4))
orientation_array[:, 0] = 1.0
snapshot.particles.orientation = orientation_array

# --- Write to File ---
op_file_name = 'Single_Sphere_3D.gsd'
with gsd.hoomd.open(name=op_file_name, mode='wb') as f:
    f.append(snapshot)
    print(f"Created {op_file_name} with {N_particles} particles scattered randomly.")




# import hoomd
# import gsd.hoomd
# import numpy as np

# # --- Parameters ---
# N_particles = 1  # Just one sphere
# L = 50.0         # Box size (large enough so it doesn't cross boundaries instantly)

# # --- Create Snapshot ---
# snapshot = gsd.hoomd.Snapshot()
# snapshot.configuration.box = [L, L, L, 0, 0, 0]  # 3D Box

# snapshot.particles.N = N_particles
# snapshot.particles.position = [[0, 0, 0]]  # Start at origin
# snapshot.particles.typeid = [0]
# snapshot.particles.types = ['A']

# # Initialize Orientation: [s, x, y, z]
# # Start facing the X-axis: quaternion = [1, 0, 0, 0]
# snapshot.particles.orientation = [[1, 0, 0, 0]]

# # --- Write to File ---
# op_file_name = 'Single_Sphere_3D.gsd'
# with gsd.hoomd.open(name=op_file_name, mode='wb') as f:
#     f.append(snapshot)
#     print(f"Created {op_file_name} with 1 particle at (0,0,0).")
