import gsd.hoomd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt

# Parameters (match simulate_single_disk.py)
U_0 = 0.1
D_r = 1.0
omega_values = [0.0, 0.3]
num_realizations = 50  # Updated to match simulations
dt = 0.005
save_freq = 100


# Theoretical MSD
def theoretical_msd(t, U_0, D_r, omega):
    if omega == 0:
        D_L = U_0**2 / (2 * D_r)  # Eqn (9)
    else:
        D_L = (1/4) * U_0**2 * D_r / (D_r**2 + omega**2)  # Eqn (8)
    return 4 * D_L * t

# Compute MSD
msd = {}
for omega in omega_values:
    msd[omega] = []
    for seed in range(num_realizations):  # 0 to 49
        filename = f"output/N_1_D_r_{D_r}_omega_{omega}_tsteps_100000_dt_{dt}_seed_{seed}.gsd"
        try:
            with gsd.hoomd.open(filename, 'rb') as f:
                traj = f[:]
                positions = np.array([frame.particles.position[0] for frame in traj])  # Single particle
                displacements = (positions - positions[0])[:, :2]  # 2D displacement
                squared_displacements = np.sum(displacements**2, axis=1)
                msd[omega].append(squared_displacements)
        except FileNotFoundError:
            print(f"File {filename} not found, skipping seed {seed}")
            continue
    if msd[omega]:  # Only compute mean if data exists
        msd[omega] = np.mean(msd[omega], axis=0)
    else:
        print(f"No data for omega = {omega}")
        msd[omega] = np.zeros(101)  # 101 frames (0 + 100 saves)

plt.figure(figsize=(8, 6))
for omega in omega_values:
    t = np.arange(len(msd[omega])) * dt * save_freq
    plt.loglog(t, msd[omega], 'o', label=f'Simulation, ω={omega}')
    plt.loglog(t, theoretical_msd(t, U_0, D_r, omega), '-', label=f'Theory, ω={omega}')
plt.xlabel('t')
plt.ylabel('⟐r²⟒')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title("Mean Square Displacement of a Single RS Disk")
plt.savefig('msd_plot2.png')  # Save the plot
