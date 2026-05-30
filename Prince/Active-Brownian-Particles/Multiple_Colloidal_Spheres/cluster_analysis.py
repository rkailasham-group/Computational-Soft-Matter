import gsd.hoomd
import freud
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Define Paths and Details for All Runs ---
simulations = {
    "omega_0.0": {
        "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_0.00.gsd",
        "label": r"Steady Limit ($\omega = 0.0$)",
        "color": "crimson"
    },
    "omega_0.2": {
        "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_0.20.gsd",
        "label": r"Slow Oscillation ($\omega = 0.2$)",
        "color": "teal"
    },
    "omega_2": {
        "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_2.00.gsd",
        "label": r"Intermediate Breathing ($\omega = 2$)",
        "color": "darkorange"
    },
    "omega_50": {
        "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_50.00.gsd",
        "label": r"Fully Suppressed ($\omega = 50$)",
        "color": "darkred"
    }
}

r_cut = 2.02  # Cutoff distance for 3D hard-sphere contact
cl = freud.cluster.Cluster()

# --- 2. Process and Store Data First ---
processed_data = {}

for key, sim_info in simulations.items():
    file_path = sim_info["path"]
    
    if not os.path.exists(file_path):
        print(f"Warning: Could not find {file_path}. Skipping this dataset.")
        continue
        
    print(f"Processing: {file_path}...")
    traj = gsd.hoomd.open(file_path, 'rb')
    N_frames = len(traj)
    N_particles = traj[0].particles.N
    
    timesteps = np.zeros(N_frames)
    largest_cluster_fractions = np.zeros(N_frames)
    
    for i, frame in enumerate(traj):
        box = frame.configuration.box
        positions = frame.particles.position
        timesteps[i] = frame.configuration.step
        
        cl.compute(system=(box, positions), neighbors={'r_max': r_cut})
        _, cluster_sizes = np.unique(cl.cluster_idx, return_counts=True)
        
        max_size = np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0
        largest_cluster_fractions[i] = max_size / N_particles

    # Save arrays for plotting
    processed_data[key] = {
        "timesteps": timesteps,
        "fractions": largest_cluster_fractions,
        "label": sim_info["label"],
        "color": sim_info["color"]
    }

# --- 3. Setup 2x2 Grid Plotting Canvas ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

# --- 4. Plot Custom Panelled Regimes ---

# Panel A: MIPS Phase (Steady & Slow Reversals)
# if "omega_0.0" in processed_data:
#     ax1.plot(processed_data["omega_0.0"]["timesteps"], processed_data["omega_0.0"]["fractions"], 
#              color=processed_data["omega_0.0"]["color"], label=processed_data["omega_0.0"]["label"], linewidth=1.5)
# if "omega_0.2" in processed_data:
#     ax1.plot(processed_data["omega_0.2"]["timesteps"], processed_data["omega_0.2"]["fractions"], 
#              color=processed_data["omega_0.2"]["color"], label=processed_data["omega_0.2"]["label"], linewidth=1.5)
# ax1.set_title("(a) MIPS Active Regimes", fontsize=12, fontweight='bold')
# ax1.set_ylim(0.65, 1.02)  # Zoomed to emphasize fine details near the top
# ax1.legend(loc="lower right", fontsize=9)
# --- ADD THIS IMPORT AT THE VERY TOP OF YOUR SCRIPT ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =====================================================================
# REPLACED PANEL A BLOCK BELOW:
# =====================================================================

# Panel A: MIPS Phase (Steady & Slow Reversals)
if "omega_0.0" in processed_data:
    ax1.plot(processed_data["omega_0.0"]["timesteps"], processed_data["omega_0.0"]["fractions"], 
             color=processed_data["omega_0.0"]["color"], label=processed_data["omega_0.0"]["label"], linewidth=1.5)
if "omega_0.2" in processed_data:
    ax1.plot(processed_data["omega_0.2"]["timesteps"], processed_data["omega_0.2"]["fractions"], 
             color=processed_data["omega_0.2"]["color"], label=processed_data["omega_0.2"]["label"], linewidth=1.5)
ax1.set_title("(a) MIPS Active Regimes", fontsize=12, fontweight='bold')
ax1.set_ylim(0.65, 1.02)  # Main plot stays wide to capture the full spikes
ax1.legend(loc="lower right", fontsize=9)

# --- APPLYING METHOD A: INSET MAGNIFYING WINDOW ---
# Creates an inset box in the 'lower left' covering 45% width and 35% height of ax1
axins = inset_axes(ax1, width="45%", height="35%", loc="lower left", borderpad=2)

# Re-plot the data inside the magnifying window
if "omega_0.0" in processed_data:
    axins.plot(processed_data["omega_0.0"]["timesteps"], processed_data["omega_0.0"]["fractions"], 
               color=processed_data["omega_0.0"]["color"], linewidth=1.5)
if "omega_0.2" in processed_data:
    axins.plot(processed_data["omega_0.2"]["timesteps"], processed_data["omega_0.2"]["fractions"], 
               color=processed_data["omega_0.2"]["color"], linewidth=1.5)

# Ultra-zoom the Y-axis to stretch the 0.95 to 1.0 gap cleanly
axins.set_ylim(0.95, 1.005)

# Focus on the final steady-state time window (from 1.5M to 2.0M timesteps)
axins.set_xlim(1.5e6, 2.0e6)

# Clean up the inset styling so it looks crisp and readable
axins.tick_params(axis='both', labelsize=8)
axins.ticklabel_format(axis='x', style='sci', scilimits=(0,0)) # Scientific notation for time
axins.grid(True, linestyle=':', alpha=0.6)

# Draw the visual dashed connector lines from the main plot to the zoom window
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.4", linestyle="--", linewidth=1)

# Panel B: Intermediate Breathing Phase (Massive Volatility)
if "omega_2" in processed_data:
    ax2.plot(processed_data["omega_2"]["timesteps"], processed_data["omega_2"]["fractions"], 
             color=processed_data["omega_2"]["color"], label=processed_data["omega_2"]["label"], linewidth=1.0)
ax2.set_title(r"(b) Tipping Point Phase ($\omega = 2$)", fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.05)  # Full scale to fit the entire structural oscillation cleanly
ax2.legend(loc="lower right", fontsize=9)

# Panel C: Fully Suppressed Fluid Phase (Ultra-Zoomed Baseline)
if "omega_50" in processed_data:
    ax3.plot(processed_data["omega_50"]["timesteps"], processed_data["omega_50"]["fractions"], 
             color=processed_data["omega_50"]["color"], label=processed_data["omega_50"]["label"], linewidth=2.0)
ax3.set_title(r"(c) Suppressed Fluid Phase ($\omega = 50$)", fontsize=12, fontweight='bold')
ax3.set_ylim(-0.001, 0.006)  # Ultra-zoomed to show residual collisional fluctuations clearly
ax3.legend(loc="upper right", fontsize=9)

# Panel D: Combined Macro-Comparison (All Curves)
for key, data in processed_data.items():
    ax4.plot(data["timesteps"], data["fractions"], color=data["color"], label=data["label"], linewidth=1.2)
ax4.set_title("(d) Macroscopic System Comparison", fontsize=12, fontweight='bold')
ax4.set_ylim(-0.05, 1.05)
ax4.legend(loc="lower right", fontsize=9)

# --- 5. Global Polish and Aesthetics ---
fig.suptitle('Effect of Speed Oscillations on Phase Separation Kinetics', fontsize=15, fontweight='bold', y=0.98)

for ax in axs.flat:
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel(r'Largest Cluster Fraction ($N_c/N$)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    # Convert scientific x-axis labels to a cleaner format
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.tight_layout()
plt.savefig("omega_comparison_MIPS_panelled.png", dpi=300)
print("\nAnalysis complete! Presentation-ready multi-panel plot saved as 'omega_comparison_MIPS_panelled.png'.")
plt.show()


# import gsd.hoomd
# import freud
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys

# # --- 1. Define Paths and Details for Both Runs ---
# simulations = {
#     "omega_0.0": {
#         "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_0.00.gsd",
#         "label": r"Steady Limit ($\omega = 0.0$)",
#         "color": "crimson"
#     },
#     "omega_0.2": {
#         "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_0.20.gsd",
#         "label": r"Oscillatory ($\omega = 0.2$)",
#         "color": "teal"
#     },
#     "omega_2": {
#         "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_2.00.gsd",
#         "label": r"Oscillatory ($\omega = 2$)",
#         "color": "darkorange"
#     },
#     "omega_50": {
#         "path": "output_data/N_4000_3D_hs_1_Dr_0.1_w_50.00.gsd",
#         "label": r"Oscillatory ($\omega = 50$)",
#         "color": "darkred"
#     }
# }

# # --- 2. Setup Plotting Canvas (Single Graph) ---
# plt.figure(figsize=(12, 6))

# r_cut = 2.02  # Cutoff distance for 3D hard-sphere contact
# cl = freud.cluster.Cluster()

# # --- 3. Process Each Simulation ---
# for key, sim_info in simulations.items():
#     file_path = sim_info["path"]
    
#     if not os.path.exists(file_path):
#         print(f"Warning: Could not find {file_path}. Skipping this dataset.")
#         continue
        
#     print(f"Processing: {file_path}...")
#     traj = gsd.hoomd.open(file_path, 'rb')
#     N_frames = len(traj)
#     N_particles = traj[0].particles.N
    
#     timesteps = np.zeros(N_frames)
#     largest_cluster_fractions = np.zeros(N_frames)
    
#     # Run clustering algorithm over all frames
#     for i, frame in enumerate(traj):
#         box = frame.configuration.box
#         positions = frame.particles.position
#         timesteps[i] = frame.configuration.step
        
#         # Compute clusters
#         cl.compute(system=(box, positions), neighbors={'r_max': r_cut})
        
#         # Safe version-independent size extraction using cl.cluster_idx
#         # np.unique returns (unique_ids, counts). We grab counts with [1]
#         _, cluster_sizes = np.unique(cl.cluster_idx, return_counts=True)
        
#         # Fraction of particles in the largest cluster
#         max_size = np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0
#         largest_cluster_fractions[i] = max_size / N_particles
        
#     # Plot this simulation's curve on the shared graph
#     plt.plot(timesteps, largest_cluster_fractions, 
#              label=sim_info["label"], 
#              color=sim_info["color"], 
#              linewidth=2.5)

# # --- 4. Finalize and Save the Graph ---
# plt.xlabel('Timestep', fontsize=12)
# plt.ylabel(r'Largest Cluster Fraction ($N_c/N$)', fontsize=12)
# plt.title('Effect of Speed Oscillations on Phase Separation', fontsize=13, fontweight='bold')
# plt.ylim(0.0, 1)  # Constrain y-axis safely from 0 to 1
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(fontsize=11, loc='lower right')

# plt.tight_layout()
# plt.savefig("omega_comparison_MIPS.png", dpi=300)
# print("\nAnalysis complete! Combined plot saved as 'omega_comparison_MIPS.png'.")
# plt.show()