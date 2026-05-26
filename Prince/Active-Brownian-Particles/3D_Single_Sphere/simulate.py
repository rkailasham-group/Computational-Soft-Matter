import multiprocessing as mp
import gsd.hoomd
import hoomd
import numpy as np
import os
import sys

# --- Input Handling ---
l = len(sys.argv)
if l < 9:
    print("\n Correct syntax is [script.py] [path to input.gsd] [restart? 0/1] [dt] [timesteps] [Dr] [radius] [output folder] [save_freq] \n")
    exit(0)

inp_file_path = sys.argv[1]
restart_param = int(float(sys.argv[2]))
time_step_width = float(sys.argv[3])        # dt
sim_length = int(float(sys.argv[4]))        # Total timesteps
rot_diff_const = float(sys.argv[5])         # Dr
rad_sphere = float(sys.argv[6])             # Radius of sphere (was disk)
op_folder_path = sys.argv[7]                # Output folder
save_freq = int(sys.argv[8])                # Save frequency

# --- Helper Class to Print Progress ---
class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep, " timesteps computed")

# --- 3D Active Force Class (The Core Math) ---
class CustomActiveForce_3D(hoomd.md.force.Custom):
    def __init__(self, f_array, freq, rotation_diff, N, dt):
        super().__init__(aniso=False)
        self.f_array = f_array  # Magnitude of force
        self.rotation_diff = rotation_diff
        self.active_fi = np.zeros((N, 3))
        self.freq = freq
        self.dt = dt
        self.N = N

    def update_force(self, timestep):
        # 1. Get current particle data
        with self._state.cpu_local_snapshot as data:
            quati = np.array(data.particles.orientation.copy()) # Shape (N, 4) -> [s, x, y, z]

        # 2. Extract Current Heading Vector (n) Using Full 3x3 Matrix
        p_x = 1.0
        p_y = 0.0
        p_z = 0.0

        q_s = quati[:, 0]
        q_x = quati[:, 1]
        q_y = quati[:, 2]
        q_z = quati[:, 3]

        # Column 1 
        R11 = 1.0 - 2.0 * (q_y**2 + q_z**2)
        R21 = 2.0 * (q_x * q_y + q_s * q_z)
        R31 = 2.0 * (q_x * q_z - q_s * q_y)

        # Column 2
        R12 = 2.0 * (q_x * q_y - q_s * q_z)
        R22 = 1.0 - 2.0 * (q_x**2 + q_z**2)
        R32 = 2.0 * (q_y * q_z + q_s * q_x)

        # Column 3
        R13 = 2.0 * (q_x * q_z + q_s * q_y)
        R23 = 2.0 * (q_y * q_z - q_s * q_x)
        R33 = 1.0 - 2.0 * (q_x**2 + q_y**2)

        # Calculate Global Heading (Matrix Multiplication: R * p_local)
        n_x = (R11 * p_x) + (R12 * p_y) + (R13 * p_z)
        n_y = (R21 * p_x) + (R22 * p_y) + (R23 * p_z)
        n_z = (R31 * p_x) + (R32 * p_y) + (R33 * p_z)
        
        # 'n_vecs' is the current direction of every particle
        n_vecs = np.stack((n_x, n_y, n_z), axis=1) # Shape (N, 3)

        # --- True 3D Rotational Brownian Motion ---
        
        # 3. Generate Random Angular Displacement Vector
        noise_vecs = np.random.normal(0, 1, (self.N, 3))
        sigma = np.sqrt(2 * self.rotation_diff * self.dt)
        d_theta_vecs = noise_vecs * sigma
        
        # 4. Calculate Rotation Angle and Axis
        thetas = np.linalg.norm(d_theta_vecs, axis=1, keepdims=True)
        thetas[thetas == 0] = 1.0e-10 
        u_vecs = d_theta_vecs / thetas
        
        # 5. Create Delta Quaternion (dq)
        sin_half = np.sin(thetas / 2.0)
        cos_half = np.cos(thetas / 2.0)
        
        dq_s = cos_half
        dq_vec = sin_half * u_vecs 

        # 6. Apply Rotation (Hamilton Product: dq * q_old)
        old_s = quati[:, 0]
        old_v = quati[:, 1:]
        
        new_s = dq_s.flatten() * old_s - np.sum(dq_vec * old_v, axis=1)
        new_v = (dq_s * old_v) + (old_s[:, None] * dq_vec) + np.cross(dq_vec, old_v)
        
        quati = np.hstack((new_s[:, None], new_v))

        # Normalize quaternion to prevent numerical drift over time
        q_norms = np.linalg.norm(quati, axis=1, keepdims=True)
        quati /= q_norms

        # 7. Calculate Active Force Vector
        # Magnitude oscillates with time
        f_mag = self.f_array[:, 0] * np.cos(self.freq * timestep * self.dt)
        
        # We effectively rotate the vector [f_mag, 0, 0] by the NEW quaternion.
        q_s = quati[:, 0]
        q_vec = quati[:, 1:] # x, y, z parts
        
        # Term 1: (s^2 - v.v) * f
        vec_term1 = (q_s**2 - np.sum(q_vec**2, axis=1))[:, None] * np.stack((f_mag, np.zeros(self.N), np.zeros(self.N)), axis=1)
        
        # Term 2: 2*s*(v x f)
        f_full = np.stack((f_mag, np.zeros(self.N), np.zeros(self.N)), axis=1)
        vec_term2 = 2 * q_s[:, None] * np.cross(q_vec, f_full)
        
        # Term 3: 2*(v . f)*v
        vec_term3 = 2 * np.sum(q_vec * f_full, axis=1)[:, None] * q_vec

        self.active_fi = vec_term1 + vec_term2 + vec_term3

        # 8. Update Orientation in HOOMD State
        with self._state.cpu_local_snapshot as data:
            data.particles.orientation = quati

    def set_forces(self, timestep):
        self.update_force(timestep)
        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] = self.active_fi
        pass


# --- Main Simulation Function ---
def simulate(fname, op_dir, w):
    print("Initializing Simulation...")
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    
    if restart_param == 0:
        sim.timestep = 0

    # Load the GSD file
    sim.create_state_from_gsd(fname)

    N = sim.state.N_particles
    tsteps = sim_length
    dt = time_step_width

    # Integrator setup
    integrator = hoomd.md.Integrator(dt, integrate_rotational_dof=False)
    cell = hoomd.md.nlist.Cell(buffer=0.4)

    # Active Force Setup
    f_a = 0.1
    f_array = f_a * np.ones((N, 3)) 
    f_array[:, 1:3] = 0. # Force is along local X

    # Attach the NEW 3D Force Class
    custom_active = CustomActiveForce_3D(f_array=f_array, freq=w, rotation_diff=d_r, N=N, dt=dt)
    integrator.forces.append(custom_active)

    # --- Steric Repulsion (Heyes-Melrose) ---
    gamma = 1
    rad = rad_sphere
    r_hs_min = rad
    r_hs_cut = 2. * rad
    
    # Using tabulated potential for soft spheres
    r = np.linspace(r_hs_min, r_hs_cut, 2, endpoint=False)
    U_hs = gamma / (4 * dt) * (r - 2 * rad)**2
    F_hs = -gamma / (2 * dt) * (r - 2 * rad)
    
    hard_sphere = hoomd.md.pair.Table(nlist=cell, default_r_cut=r_hs_cut)
    hard_sphere.params[('A', 'A')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)

    if hs_flag > 0:
        print("Steric Repulsion: ON")
        integrator.forces.append(hard_sphere)
    else:
        print("Steric Repulsion: OFF (Phantom Spheres)")

    # Integrator Method
    odv = hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    odv.gamma.default = gamma
    odv.gamma_r.default = [0, 0, 0] # No rotational drag from fluid (handled by us)
    integrator.methods.append(odv)
    
    sim.operations.integrator = integrator

    # Writers
    custom_action = PrintTimestep()
    custom_op = hoomd.write.CustomWriter(action=custom_action, trigger=hoomd.trigger.Periodic(10000))
    sim.operations.writers.append(custom_op)

    op_file_name = 'N_%s_3D_hs_flag_%s_Dr_%s_w_%f.gsd' % (N, hs_flag, d_r, w)
    gsd_writer = hoomd.write.GSD(filename=op_dir + op_file_name, trigger=hoomd.trigger.Periodic(save_freq), mode='wb')
    sim.operations.writers.append(gsd_writer)

    # Run
    print("Running for", tsteps, "steps...")
    sim.run(tsteps)


# --- Global Variables & Execution ---
hs_flag = 0  # <--- SET TO 0 TO TURN OFF REPULSION (PHANTOM SPHERES)
d_r = rot_diff_const

# Frequency list for active force oscillation (Set to 0.0 for Steady Swimmer)
# Frequency list for active force oscillation (Reciprocal Swimming)
omega_list = np.array([0.0, 0.2, 1.0]) 

# Directory Setup
curdir = os.getcwd()
resdir = op_folder_path
try:
    os.mkdir(resdir)
    print("Directory made")
except FileExistsError:
    pass # Removed redundant print to keep output clean

# Execution Loop
for i, w in enumerate(omega_list):
    print("Now running w = ", w)
    simulate(inp_file_path, resdir, w)

print("All Simulations Completed.")