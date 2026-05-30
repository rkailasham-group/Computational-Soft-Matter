import os
import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Global Parameters ---
D_r = 0.1
r_cut = 2.02  # Cutoff for accurate dense fluid distinction
omegas = [0.0, 0.2, 0.5, 1.0, 2.0, 2.50, 10.0, 50.0]

# Configured with separate folders for each particle count configuration
systems = {
    4000: {
        "folder": "output_data",
        "color": "blue",
        "marker": "d",
        "label": r"$N=4000, \phi=0.40$",
    },
     2000: {
        "folder": "output_data2",
        "color": "green",
        "marker": "p",
        "label": r"$N=2000, \phi=0.40$",
    },
    500: {
        "folder": "output_data1",
        "color": "crimson",
        "marker": "o",
        "label": r"$N=500, \phi=0.40$",
    },
   
}

# Initialize the figure canvas
plt.figure(figsize=(8, 6))

# --- 2. Process Data and Plot for Each System Size ---
for N, config in systems.items():
    print(f"\n--- Starting variance analysis for N = {N} ---")

    # Wipes out the memory cache from the previous system size
    cl = freud.cluster.Cluster()

    gamma_values = []
    cluster_variances = []

    for omega in omegas:
        gamma = omega / D_r

        # Uses the specific directory assigned to this system size
        file_path = f"{config['folder']}/N_{N}_3D_hs_1_Dr_0.1_w_{omega:.2f}.gsd"

        if not os.path.exists(file_path):
            print(f"  Warning: File {file_path} not found. Skipping...")
            continue

        print(f"  Processing omega = {omega} (gamma = {gamma})...")
        traj = gsd.hoomd.open(file_path, "rb")
        N_frames = len(traj)

        # Average over the steady-state (the last 50% of the simulation runtime)
        start_frame = N_frames // 2
        steady_state_frames = N_frames - start_frame

        # NOTE: Using absolute size (N_LC) rather than fraction for variance
        max_sizes = np.zeros(steady_state_frames)

        # Process second half of the trajectory frames
        for i, frame_idx in enumerate(range(start_frame, N_frames)):
            frame = traj[frame_idx]
            box = frame.configuration.box

            # Use .copy() to enforce distinct contiguous arrays
            positions = frame.particles.position.copy()

            cl.compute(system=(box, positions), neighbors={"r_max": r_cut})
            _, cluster_sizes = np.unique(cl.cluster_idx, return_counts=True)

            max_size = np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0
            max_sizes[i] = max_size  # Store raw absolute count

        # Close the trajectory file handle
        traj.close()

        # Variance calculation of the absolute cluster size
        variance = np.var(max_sizes)

        gamma_values.append(gamma)
        cluster_variances.append(variance)

    # Sort data by gamma to ensure proper line assembly left-to-right
    sorted_indices = np.argsort(gamma_values)
    gamma_values = np.array(gamma_values)[sorted_indices]
    cluster_variances = np.array(cluster_variances)[sorted_indices]

    # Overlay current system curve onto canvas
    plt.plot(
        gamma_values,
        cluster_variances,
        marker=config["marker"],
        markersize=8,
        linestyle="-",
        linewidth=2,
        color=config["color"],
        label=config["label"],
    )

# --- 3. Finalize and Save the Combined Figure ---

# X-axis symlog handles gamma=0. Y-axis log handles wide variance magnitudes.
plt.xscale("symlog", linthresh=0.5)
plt.yscale("log")

# Axis Decorations and Labels
plt.xlabel(r"$\gamma \equiv \omega / D_r$", fontsize=14)
plt.ylabel(r"$\text{var}(N_{LC})$", fontsize=14)
plt.title(
    "Variance of Largest Cluster Size at Steady State", fontsize=15, fontweight="bold"
)

# Grid and Legend
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=12, loc="best")

plt.tight_layout()
plt.savefig("gamma_vs_cluster_variance.png", dpi=300)
print(
    "\n[Success] Graph generated and saved as 'gamma_vs_cluster_variance.png'"
)
plt.show()