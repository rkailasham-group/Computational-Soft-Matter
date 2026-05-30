import gsd.hoomd

t4000 = gsd.hoomd.open("output_data/N_4000_3D_hs_1_Dr_0.1_w_2.50.gsd", "rb")
t500 = gsd.hoomd.open("output_data1/N_500_3D_hs_1_Dr_0.1_w_2.50.gsd", "rb")

print("--- Frame 0, First 2 Particles Positions ---")
print("From output_data (N=4000):")
print(t4000[0].particles.position[:2])

print("\nFrom output_data1 (N=500):")
print(t500[0].particles.position[:2])