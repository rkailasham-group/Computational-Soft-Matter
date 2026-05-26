import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = 'msd_results'
plt.figure(figsize=(10, 8))

for filename in os.listdir(results_dir):
    if filename.endswith(".dat"):
        # Format: msd_Dr_0.1_w_0.2.dat
        parts = filename.replace('.dat', '').split('_')
        dr_val = float(parts[2])
        w_val = float(parts[4])

        data = np.loadtxt(os.path.join(results_dir, filename))
        t = data[:, 0]
        msd = data[:, 1]

        # Plot simulation data
        label_str = 'Steady Swimmer' if w_val == 0.0 else f'Reciprocal (w={w_val})'
        p = plt.loglog(t, msd, label=label_str, linewidth=2)
        
       # --- Lauga Reciprocal Theory ---
        dt = 0.001
        v_val = 0.1  # Your active force magnitude
        tau = 1.0 / (2.0 * dr_val) # Orientation relaxation time
        t_real = t * dt 
        
        # The Professor's Correction: Time-Averaged Velocity Factor
        if w_val == 0.0:
            # Steady Swimmer (v^2)
            msd_theory = ((2.0 * v_val**2 * tau)) * t_real
        else:
            # Reciprocal Swimmer (Time-averaged cos^2 gives 1/2, removing the 2.0)
            msd_theory = ((v_val**2 * tau) / (1.0 + (w_val**2 * tau**2))) * t_real
        
        # Plot theory line (dashed)
        plt.loglog(t[10:], msd_theory[10:], color=p[0].get_color(), linestyle='--', alpha=0.7)

plt.xlabel('Time Steps')
plt.ylabel('MSD')
plt.title('Enhanced Diffusion by Reciprocal Swimming (Lauga 2011)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.savefig('Reciprocal_Plot.png', dpi=300)
print("Graph saved as Reciprocal_Plot.png")