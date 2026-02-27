import numpy as np
from svp import svp


# Constants
Rho_w = 1000.0 # Density of liquid water (kg m^-3)
Rv = 462.0  # Gas constant of water vapour (J kg^-1 K^-1)
Lv = 2.5e6 # Latent heat of vaporisation (J kg^-1)
k = 0.024  # Thermal conductivity of air (J m^-1 s^-1 K^-1)
Kv = 2.21e-5 # Diffusivity of water vapour (m^2 s^-1)


# Initial conditions
r0 = 1.0e-6 # Initial droplet radius (m)
s = 0.003 # Supersaturation ratio
T = 283.0 # Temperature (K)
total_time = 40 * 60 # Total time (s)
dt = 1.0 # Time step (s)


# SVP at 283K
es = svp([0, T])


# Calculate thermodynamic factor A1
term1 = Lv * Rho_w / (k * Rv * T)
term2 = Rho_w * Rv * T / (Kv * es)
A1 = 1.0 / (term1 + term2)


# Initial conditions for Forward Euler
t = 0.0 # Initial time
r = r0 # Initial radius

# Forward Euler loop
while t < total_time:
    dr_dt = A1 * s / r
    r = r + dt * dr_dt
    t = t + dt

print(f"Radius after {total_time} seconds: {r*1e6:.4f} μm")
print(f"Growth: {(r-r0)*1e6:.4f} μm")