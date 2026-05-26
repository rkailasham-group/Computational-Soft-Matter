#!/usr/bin/python3

import gsd.hoomd
import numpy as np
import sys
import os
import re

inp_dir = sys.argv[1]
op_dir = sys.argv[2]

full_path_inp_dir = os.path.abspath(inp_dir)
full_path_op_dir = os.path.abspath(op_dir)

if not os.path.exists(full_path_op_dir):
    os.makedirs(full_path_op_dir)

print(f"Reading GSD files from: {full_path_inp_dir}")

for file in os.listdir(full_path_inp_dir):
    filename = os.fsdecode(file)

    if filename.endswith(".gsd"):
        full_filename = os.path.join(full_path_inp_dir, filename)
        
        # Extract Dr and w from the filename safely
        # Format: N_500_3D_hs_flag_0_Dr_0.1_w_1.000000.gsd
        try:
            name_no_ext = filename.replace('.gsd', '')
            parts = name_no_ext.split('_')
            
            d_r = float(parts[7])
            w_val = float(parts[9])
        except (ValueError, IndexError):
            print(f"  [Warning] Could not parse parameters from {filename}. Skipping.")
            continue

        print(f"\nProcessing: Dr = {d_r}, w = {w_val}")

        traj = gsd.hoomd.open(full_filename, 'r')
        times = []
        msd_values = []

        frame0 = traj[0]
        box_L = frame0.configuration.box[0]  
        
        r0 = frame0.particles.position.copy()
        r_unwrapped = r0.copy()
        prev_pos = r0.copy()

        times.append(frame0.configuration.step)
        msd_values.append(0.0)

        for i in range(1, len(traj)):
            frame = traj[i]
            t = frame.configuration.step
            pos = frame.particles.position 

            # Custom Pac-Man Unwrapping
            delta = pos - prev_pos
            delta = delta - box_L * np.round(delta / box_L)
            r_unwrapped += delta
            prev_pos = pos.copy()

            # Squared Displacement & Ensemble Average
            disp_vector = r_unwrapped - r0
            sq_disp = np.sum(disp_vector**2, axis=1)
            mean_msd = np.mean(sq_disp)

            times.append(t)
            msd_values.append(mean_msd)

        output_data = np.column_stack((times, msd_values))
        
        # Output filename includes both Dr and w
        out_name = f"msd_Dr_{d_r}_w_{w_val}.dat"
        out_path = os.path.join(full_path_op_dir, out_name)

        np.savetxt(out_path, output_data, fmt=['%d', '%.6f'],
                   header="Timestep\tSquared_Displacement")