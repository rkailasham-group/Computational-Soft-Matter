import hoomd
import numpy as np

device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=1)

L = 100.0

snap = hoomd.Snapshot()
snap.particles.N = 1
snap.particles.types = ['A']
snap.particles.position[:] = [[0.0, 0.0, 0.0]]
snap.particles.orientation[:] = [[1.0, 0.0, 0.0, 0.0]]
snap.configuration.box = [L, L, 0, 0, 0, 0]

sim.create_state_from_snapshot(snap)

gsd_writer = hoomd.write.GSD(
    filename='N1_L100.gsd',
    trigger=hoomd.trigger.Periodic(1),
    mode='wb'
)

sim.operations.writers.append(gsd_writer)

# MUST run at least 1 step
sim.run(1)

print("SUCCESS: N1_L100.gsd written with 1 frame")
