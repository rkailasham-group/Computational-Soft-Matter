import multiprocessing as mp
import gsd.hoomd
import hoomd
import numpy as np
import os
import sys

# --- Input Handling ---
l = len(sys.argv)
if l < 10:
    print("\n Correct syntax is: python script.py [path to input.gsd] [restart? 0/1] [dt] [timesteps] [Dr] [radius] [output folder] [save_freq] [omega]\n", flush=True)
    exit(0)

inp_file_path = sys.argv[1]
restart_param = int(float(sys.argv[2]))
time_step_width = float(sys.argv[3])        # dt
sim_length = int(float(sys.argv[4]))        # Total timesteps for this run
rot_diff_const = float(sys.argv[5])         # Dr
rad_sphere = float(sys.argv[6])             # Radius of sphere (R)
op_folder_path = sys.argv[7]                # Output folder directory
save_freq = int(sys.argv[8])                # GSD frame save frequency
omega_param = float(sys.argv[9])            # Speed oscillation frequency (w)

# --- Global Configurations ---
hs_flag = 1  
d_r = rot_diff_const
w = omega_param

# --- Helper Class to Print Real-Time Progress ---
class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(f" Timestep {timestep} computed successfully.", flush=True)

# --- Custom Updater for 3D Rotational Brownian Motion ---
class RotationalDiffusionUpdater(hoomd.custom.Action):
    def __init__(self, rotation_diff, dt):
        super().__init__()
        self.rotation_diff = rotation_diff
        self.dt = dt

    def act(self, timestep):
        with self._state.cpu_local_snapshot as data:
            quati = data.particles.orientation
            N = len(quati)
            if N == 0:
                return

            # --- True 3D Rotational Brownian Motion Math ---
            noise_vecs = np.random.normal(0, 1, (N, 3)).astype(np.float32)
            sigma = np.sqrt(2 * self.rotation_diff * self.dt)
            d_theta_vecs = noise_vecs * sigma
            
            thetas = np.linalg.norm(d_theta_vecs, axis=1, keepdims=True)
            thetas[thetas == 0] = 1.0e-10 
            u_vecs = d_theta_vecs / thetas
            
            sin_half = np.sin(thetas / 2.0)
            cos_half = np.cos(thetas / 2.0)
            
            dq_s = cos_half
            dq_vec = sin_half * u_vecs 

            old_s = quati[:, 0]
            old_v = quati[:, 1:]
            
            new_s = dq_s.flatten() * old_s - np.sum(dq_vec * old_v, axis=1)
            new_v = (dq_s * old_v) + (old_s[:, None] * dq_vec) + np.cross(dq_vec, old_v)
            
            new_quat = np.hstack((new_s[:, None], new_v))

            # Normalize to eliminate cumulative numerical drift
            q_norms = np.linalg.norm(new_quat, axis=1, keepdims=True)
            new_quat /= q_norms

            # Write back orientations securely during the Updater phase
            data.particles.orientation[:] = new_quat


# --- 3D Active Force Class ---
class CustomActiveForce_3D(hoomd.md.force.Custom):
    def __init__(self, f_a, freq, dt):
        super().__init__(aniso=False)
        self.f_a = f_a
        self.freq = freq
        self.dt = dt

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as data:
            quati = data.particles.orientation.copy()
            N = len(quati)

        if N == 0:
            return

        # Compute Time-Oscillating Active Force Vectors
        f_mag = self.f_a * np.cos(self.freq * timestep * self.dt)
        
        q_s = quati[:, 0]
        q_vec = quati[:, 1:] 
        
        vec_term1 = (q_s**2 - np.sum(q_vec**2, axis=1))[:, None] * np.stack((f_mag * np.ones(N), np.zeros(N), np.zeros(N)), axis=1)
        f_full = np.stack((f_mag * np.ones(N), np.zeros(N), np.zeros(N)), axis=1)
        vec_term2 = 2 * q_s[:, None] * np.cross(q_vec, f_full)
        vec_term3 = 2 * np.sum(q_vec * f_full, axis=1)[:, None] * q_vec

        active_fi = vec_term1 + vec_term2 + vec_term3

        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] = active_fi


# --- Main Simulation Function ---
def simulate(fname, op_dir, w_val):
    print(f"Initializing HOOMD Simulation Instance (w = {w_val})...", flush=True)
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    
    sim.create_state_from_gsd(fname)
    
    # FIX FOR HOOMD v3 DIRECT CLOCK OVERRIDE:
    if restart_param == 0:
        sim._timestep = 0  
        file_mode = 'wb'
        print("Starting simulation from timestep 0.", flush=True)
    else:
        file_mode = 'ab'
        print(f"Restarting simulation workflow. Resuming from snapshot timestep: {sim.timestep}", flush=True)

    dt = time_step_width

    # Integrator and neighbor list configuration
    integrator = hoomd.md.Integrator(dt, integrate_rotational_dof=False)
    cell = hoomd.md.nlist.Cell(buffer=0.4)

    # Attach Custom Active Force Engine
    f_a_base = 1.0  
    custom_active = CustomActiveForce_3D(f_a=f_a_base, freq=w_val, dt=dt)
    integrator.forces.append(custom_active)

    # --- Steric Repulsion Implementation (Heyes-Melrose Soft Potentials) ---
    gamma = 1
    rad = rad_sphere
    
    r_hs_min = 0.0          
    r_hs_cut = 2. * rad       
    
    r = np.linspace(r_hs_min, r_hs_cut, 500)
    U_hs = (gamma / (4.0 * dt)) * (r - 2.0 * rad)**2
    F_hs = -(gamma / (2.0 * dt)) * (r - 2.0 * rad)
    
    hard_sphere = hoomd.md.pair.Table(nlist=cell, default_r_cut=r_hs_cut)
    hard_sphere.params[('A', 'A')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)

    print("Steric Repulsion: ENABLED", flush=True)
    integrator.forces.append(hard_sphere)

    # Overdamped Viscous Mechanics Integration Method
    odv = hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    odv.gamma.default = gamma
    odv.gamma_r.default = [0, 0, 0] 
    integrator.methods.append(odv)
    
    sim.operations.integrator = integrator

    # Attach the Custom Updater for handling orientations at every single step
    rot_updater = hoomd.update.CustomUpdater(action=RotationalDiffusionUpdater(rotation_diff=d_r, dt=dt), trigger=hoomd.trigger.Periodic(1))
    sim.operations.updaters.append(rot_updater)

    # Periodic Console Logger Writer (Every 10k steps)
    custom_action = PrintTimestep()
    custom_op = hoomd.write.CustomWriter(action=custom_action, trigger=hoomd.trigger.Periodic(10000))
    sim.operations.writers.append(custom_op)

    # Trajectory Exporter Writer
    N = sim.state.N_particles
    op_file_name = f"N_{N}_3D_hs_{hs_flag}_Dr_{d_r}_w_{w_val:.2f}.gsd"
    gsd_writer = hoomd.write.GSD(filename=os.path.join(op_dir, op_file_name), trigger=hoomd.trigger.Periodic(save_freq), mode=file_mode)
    sim.operations.writers.append(gsd_writer)

    # Launch Simulation Run
    print(f"Beginning execution run for {sim_length} steps...", flush=True)
    sim.run(sim_length)


# --- Directory Validation and Pipeline Invocation ---
resdir = os.path.abspath(op_folder_path)
if not os.path.exists(resdir):
    try:
        os.makedirs(resdir)
        print(f"Created output target folder: {resdir}", flush=True)
    except FileExistsError:
        pass

simulate(inp_file_path, resdir, w)
print("Simulation Routine Completed Successfully.", flush=True)